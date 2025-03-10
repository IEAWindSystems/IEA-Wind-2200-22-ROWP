import numpy as np
from py_wake.rotor_avg_models import RotorCenter
import time
import matplotlib.pyplot as plt
import xarray as xr
import os
import utm
import pickle
from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake import NOJ, BastankhahGaussian
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasySGDDriver
from OptPlotBathy import XYPlotCompBathym
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint, InclusionZone
from py_wake.utils.gradients import fd, autograd
from topfarm.constraint_components.constraint_aggregation import DistanceConstraintAggregation
from py_wake.turbulence_models import CrespoHernandez
from windIO.utils.yml_utils import load_yaml
from scipy.interpolate import RegularGridInterpolator
from ssms.CalculateMass import CalculateMass
from ed_win.wind_farm_network import WindFarmNetwork
from shapely.geometry import Point, Polygon
from topfarm.constraint_components.boundary import MultiWFPolygonBoundaryConstraint
np.random.seed(2)
#
#%% INPUTS
# farm layout
Mode = 'competitive'    # 'cooperative' or 'competitive'
Sequence = ['north','mid','south']
Model = 'gauss'
plot_conv = True
tur_nr = [33,33,34]             # Desired turbine number in optimized farm, from north to south!

# monopile optimization
MP_ref = 1          # reference turbine type for monopile mass scaling. 0 = 10MW, 1 = 15MW, 2 = 3.4MW
# lcoe parameters
d = 0.05             # [-] discount rate
life = 25            # years, lifetime
RP = 22              # MW
D = 284              # m
HH = 170             # m
HTrans = 15          # m
WaveHeight = 2.52    # m
WavePeriod = 5.45    # s
WindSpeed = 9.924    # m/s ToDo: verfiy it is average wind speed
capex = 9.6712e7 * 0.924        # €2024 per turbine, excl. Monopile and cabling, from NREL COE Report 2024, converted to € with 2024 average exchange rate
OpexAnnual = 2.97e6 * 0.924     # €2024 per turbine, annual OPEX,from NREL COE Report 2024, converted to € with 2024 average exchange rate
LP = 0               # $2010 per turbine, liquidation proceeds, from DETECT for HKN scaled (22MW turbines)
#
# cable data [cross section, capacity, price]
# 110kV
cables = np.array([[185,3,368.9], [400,5,428.9], [1000,7,737.1]])
# cables = np.array([[500, 3, 393], [800, 5, 522.4], [1000, 7, 615.5]])
# inflation correction 2017€ to 2024€
# Inf = [2.1, 2.4, 1.8, 1.2, 4.7, 8, 4.1]     # 2017-2023
# cables[:,2] *= np.prod(np.array(Inf) / 100 + 1)
#
#%% Load data and setup pywake
# system_dat = sys.argv[1]
system_dat = load_yaml(os.sep.join(['..', 'inputs', 'wind_energy_system.yaml']))
farm_dat = system_dat['wind_farm']
resource_dat = system_dat['site']['energy_resource']

if 'timeseries' in resource_dat['wind_resource'].keys():
   timeseries = True
   wind_resource_timeseries = resource_dat['wind_resource']['timeseries']
   times = [d['time'] for d in wind_resource_timeseries]
   ws = [d['speed'] for d in wind_resource_timeseries]
   wd = [d['direction'] for d in wind_resource_timeseries]
   assert(len(times) == len(ws))
   assert(len(wd) == len(ws))
   site = Hornsrev1Site()
   TI = None
elif 'weibull_k' in resource_dat['wind_resource'].keys():
   A = resource_dat['wind_resource']['weibull_a']
   k = resource_dat['wind_resource']['weibull_k']
   freq = resource_dat['wind_resource']['sector_probability']
   wd = resource_dat['wind_resource']['wind_direction']
   # ws = resource_dat['wind_resource']['wind_speed']
   site = XRSite(
          ds=xr.Dataset(data_vars=
                           {'Sector_frequency': ('wd', freq['data']), 
                            'Weibull_A': ('wd', A['data']), 
                            'Weibull_k': ('wd', k['data']),
                            'TI': resource_dat['wind_resource']['turbulence_intensity']['data']
                            },
                         coords={'wd': wd}))
   timeseries = False
   TI =  resource_dat['wind_resource']['turbulence_intensity']['data']
else:
   timeseries = False
   ws = np.unique(resource_dat['wind_resource']['wind_speed'])
   wd = np.unique(resource_dat['wind_resource']['wind_direction'])
   P = np.array(resource_dat['wind_resource']['probability']['data'])
   site = XRSite(ds=xr.Dataset(data_vars={'P': (['wd', 'ws'], P)}, coords = {'ws': ws, 'wd': wd, 'TI': resource_dat['wind_resource']['turbulence_intensity']['data']}))
   TI = resource_dat['wind_resource']['turbulence_intensity']['data']

# define turbine
hh = farm_dat['turbines']['hub_height']
rd = farm_dat['turbines']['rotor_diameter']
rp = farm_dat['turbines']['performance']['rated_power']
cut_in = farm_dat['turbines']['performance']['cutin_wind_speed']
cut_out = farm_dat['turbines']['performance']['cutout_wind_speed']
# cp = farm_dat['turbines']['performance']['Cp_curve']['Cp_values']
# cp_ws = farm_dat['turbines']['performance']['Cp_curve']['Cp_wind_speeds']
p = farm_dat['turbines']['performance']['power_curve']['power_values']
p_ws = farm_dat['turbines']['performance']['power_curve']['power_wind_speeds']
ct = farm_dat['turbines']['performance']['Ct_curve']['Ct_values']
ct_ws = farm_dat['turbines']['performance']['Ct_curve']['Ct_wind_speeds']
int_speeds = np.linspace(np.min(np.min([p_ws, ct_ws])), np.max(np.max([p_ws, ct_ws])), 10000)
ps_int = np.interp(int_speeds, p_ws, p)
cts_int = np.interp(int_speeds, ct_ws, ct)
windTurbines = WindTurbine(name=farm_dat['turbines']['name'], diameter=rd, hub_height=hh, 
                      powerCtFunction=PowerCtTabular(int_speeds, ps_int, power_unit='W', ct=cts_int))

if Model == 'jensen':
    wake_model = NOJ(site, windTurbines, k=0.05, rotorAvgModel=RotorCenter())
elif Model == 'gauss':
    wake_model = BastankhahGaussian(site, windTurbines, rotorAvgModel=RotorCenter(), turbulenceModel=CrespoHernandez())

# wind resource
dirs = np.arange(0, 360, 1) #wind directions
speeds = np.arange(cut_in, cut_out+1, 1) # wind speeds
freqs = site.local_wind([0], [0], wd=dirs, ws=speeds).P_ilk[0, :, :]     # all frequencies
ws = np.arange(cut_in, cut_out+1, 1)

Subs_x = system_dat['wind_farm']['electrical_substations']['electrical_substation']['coordinates']['x']
Subs_y = system_dat['wind_farm']['electrical_substations']['electrical_substation']['coordinates']['y']

# Mapping for wind farms to indices
wf = {
    "north": 0,
    "mid": 1,
    "south": 2
}

# Boundaries and exclusion zones
b = system_dat['site']
boundaries = {
    name: np.array([
        b['boundaries']['polygons'][index]['x'],
        b['boundaries']['polygons'][index]['y']
    ]).T
    for name, index in wf.items()
}

# bathymetry
X = np.array(system_dat['site']['Bathymetry']['latitude'])
Y = np.array(system_dat['site']['Bathymetry']['longitude'])
Z = np.array(system_dat['site']['Bathymetry']['elevation']['data'])

# Transfer from LongLat to UTM (km)
X_utm = utm.from_latlon(np.ones(len(Y))*X[0],Y)
Y_utm = utm.from_latlon(X,np.ones(len(X))*Y[0])

# Create grids
lon_grid, lat_grid = np.meshgrid(Y, X)

# Convert to UTM
Easting, Northing, _, _ = utm.from_latlon(lat_grid, lon_grid)

# Flip arrays if necessary
if not np.all(np.diff(Easting[0, :]) > 0):
    Easting = np.fliplr(Easting)
    Z = np.fliplr(Z)
if not np.all(np.diff(Northing[:, 0]) > 0):
    Northing = np.flipud(Northing)
    Z = np.flipud(Z)

# Extract coordinate arrays
northing_values = Northing[:, 0]  # Corresponds to axis 0 of Z
easting_values = Easting[0, :]    # Corresponds to axis 1 of Z

# Ensure arrays are sorted
assert np.all(np.diff(northing_values) > 0), "Northing values are not strictly increasing."
assert np.all(np.diff(easting_values) > 0), "Easting values are not strictly increasing."

# Create the interpolator
interpolator = RegularGridInterpolator((northing_values, easting_values), -Z, method='cubic')
def depth_interp(x, y):
    return interpolator(np.array([y, x]).T)

# Calculate monopile mass for different water depth
depths = np.linspace(np.min(-Z),np.max(-Z),20)
ph = 15     # Platform height [m]
swh = 2.52  # Significant Wave Height [m]
swp = 5.45  # Significiant Wave Period [s]
# P_interpolator = interp1d(np.cumsum(sum(P)), ws, kind='linear')  # interpolator to get mean wind sped (@50% probability)
# V_ave = P_interpolator(0.5).tolist()
import scipy.special as sp
def mean_wind_speed(A, k):
    return A * sp.gamma(1 + 1/k)
V_ave = []
for i in range(len(wd)):
    V_ave.append(mean_wind_speed(A['data'][i],k['data'][i]))
V_ave = np.sum(np.array(V_ave) * np.array(freq['data']))
masses = []
for z in depths:
   cur_mass = CalculateMass(RP=rp/1e6, D=rd, HTrans=ph, HHub_Ratio=hh/rd, WaterDepth=z, WaveHeight=swh, WavePeriod=swp, WindSpeed=V_ave, IP_item=MP_ref)
   masses.append(cur_mass[0][0])
# add transition piece (100t, from 22MW report)
masses = [x + 100000 for x in masses]

# Fit a polynomial of degree 2
# depthmass = np.genfromtxt('depth.mass', delimiter=',')
depthmass = np.column_stack((masses,depths))
coefficients = np.polyfit(depthmass[:, 1], depthmass[:, 0], 2)
polynomial = np.poly1d(coefficients)
polynomial_gradients = np.polyder(polynomial)

# objective function and gradient function
samps = 50    #number of samples 
site.interp_method = 'linear'

# defaults
# neighbour wind farm with turbine coordinates and costs to consider
nf = False
xn = []
yn = []
cable_cost_n = [0,0]
cable_u_n = [[],[]]
cable_v_n = [[],[]]
cable_type_n = [[],[]]
mp_cost_n = [0,0]
SepCabling = False
opt_nr = 1

#%% Objective function
def lcoe_func(x, y, **kwargs):
    global metrics_recorder, aep, cable_cost, dcable_cost, mp_cost, dmp_cost, cable_cost_n, cable_u_n, cable_v_n, cable_type_n, mp_cost_n
    # 1.) aep
    wd = np.arange(0, 360, 1)
    if nf:
        aep = wake_model(x=np.concatenate((x,xn)), y=np.concatenate((y,yn)), wd=wd, ws=ws, TI=TI).aep() * 1e3
    else:
        aep = wake_model(x=x, y=y, wd=wd, ws=ws, TI=TI).aep() * 1e3
        # aep = xr.DataArray(np.sum(aep, axis=(1, 2)), dims=["wt"])
    #
    # 2.) monopile costs
    depths = depth_interp(x, y)
    masses = []
    dmasses = []
    for water_depth in depths:
       dmasses.append(polynomial_gradients(water_depth))
       masses.append(polynomial(water_depth))
    mp_cost = 3 * np.array(masses)  # from ORBIT 2025 monopile_steel_cost default
    # gradients (for function later, to avoid double calculus)
    dmasses = (np.array(dmasses) * get_depth_grads(x, y)[:, 0, :].T)
    dmp_cost = 3 * dmasses          # from ORBIT 2025 tp_steel_cost default
    #
    # 3.) Cable costs
    if SepCabling:
        xc1 = x[:tur_nr[0]]
        xc2 = x[tur_nr[0]:sum(tur_nr[0:2])]
        xc3 = x[sum(tur_nr[0:2]):]
        yc1 = y[:tur_nr[0]]
        yc2 = y[tur_nr[0]:sum(tur_nr[0:2])]
        yc3 = y[sum(tur_nr[0:2]):]
        wfn1 = WindFarmNetwork(wt_x=xc1, wt_y=yc1, ss_x=Sx[0], ss_y=Sy[0], cables=cables)
        wfn2 = WindFarmNetwork(wt_x=xc2, wt_y=yc2, ss_x=Sx[1], ss_y=Sy[1], cables=cables)
        wfn3 = WindFarmNetwork(wt_x=xc3, wt_y=yc3, ss_x=Sx[2], ss_y=Sy[2], cables=cables)
        G1 = wfn1.optimize()
        G2 = wfn2.optimize()
        G3 = wfn3.optimize()
        cable_cost = G1.cost + G2.cost + G3.cost     # in Euro
        # for recorder
        cab_data1 = G1.get_table()
        cab_data2 = G2.get_table()
        cab_data3 = G3.get_table()
        metrics_recorder["cable_u"].append([x+len(Sx)-1 if x != 1 else x for x in cab_data1['u'].tolist()] + [x+len(Sx)-1+tur_nr[0] if x != 1 else x+1 for x in cab_data2['u'].tolist()] + [x+len(Sx)-1+sum(tur_nr[0:2]) if x != 1 else x+2 for x in cab_data3['u'].tolist()])
        metrics_recorder["cable_v"].append([y+len(Sx)-1 if y != 1 else y for y in cab_data1['v'].tolist()] + [y+len(Sx)-1+tur_nr[0] if y != 1 else y++1 for y in cab_data2['v'].tolist()] + [y+len(Sx)-1+sum(tur_nr[0:2]) if y != 1 else y+2 for y in cab_data3['v'].tolist()])
        metrics_recorder["cable_type"].append(cab_data1['cable'].tolist() + cab_data2['cable'].tolist() + cab_data3['cable'].tolist())
        # gradients (for function later, to avoid double calculus)
        dcable_length1, dcable_cost1 = wfn1.gradient(node_type='wind_turbines')
        dcable_length2, dcable_cost2 = wfn2.gradient(node_type='wind_turbines')
        dcable_length3, dcable_cost3 = wfn3.gradient(node_type='wind_turbines')
        # dcabel_length = np.vstack((dcable_length1,dcable_length2,dcable_length3))
        dcable_cost = np.vstack((dcable_cost1,dcable_cost2,dcable_cost3))
    else:
        wfn = WindFarmNetwork(wt_x=x, wt_y=y, ss_x=Sx, ss_y=Sy, cables=cables)
        # Optimize cable layout with the given data
        G = wfn.optimize()
        # Costs
        cable_cost = G.cost     # in Euro
        # for recorder
        cab_data = G.get_table()
        metrics_recorder["cable_u"].append(cab_data['u'].tolist())
        metrics_recorder["cable_v"].append(cab_data['v'].tolist())
        metrics_recorder["cable_type"].append(cab_data['cable'].tolist())
        # gradients (for function later, to avoid double calculus)
        dcable_length, dcable_cost = wfn.gradient(node_type='wind_turbines')
    # !! ToDo: Scale costs to same currency and year of reference
    #
    # 4.) lcoe
    CRF = d / (1 - (1 + d) ** -life)
    npv = (capex*len(x) + np.sum(mp_cost) + cable_cost + LP*len(x)) * CRF + OpexAnnual*len(x)
    # lcoe = npv / np.sum(aep).item()
    lcoe = npv / aep.isel(wt=slice(0,len(x))).sum().item()
    #
    # 5. Record the missing metrics
    metrics_recorder["iteration"].append(kwargs.get("iteration", len(metrics_recorder["iteration"]) + 1))
    metrics_recorder["opt_nr"].append(opt_nr)
    metrics_recorder["aep"].append(aep.isel(wt=slice(0,len(x))).sum().item())
    metrics_recorder["mp_cost"].append(sum(mp_cost))
    metrics_recorder["cable_cost"].append(cable_cost)
    metrics_recorder["lcoe"].append(lcoe)
    metrics_recorder["x"].append(x)
    metrics_recorder["y"].append(y)
    # performance of individual zones
    if Mode == 'cooperative':
        # store the values of the individual zones
        mp_cost1 = np.sum(mp_cost[:tur_nr[0]])
        mp_cost2 = np.sum(mp_cost[tur_nr[0]:sum(tur_nr[0:2])])
        mp_cost3 = np.sum(mp_cost[sum(tur_nr[0:2]):])
        cable_cost1 = G1.cost
        cable_cost2 = G2.cost 
        cable_cost3 = G3.cost
        npv1 = (capex*tur_nr[0] + mp_cost1 + cable_cost1 + LP*tur_nr[0]) * CRF + OpexAnnual*tur_nr[0]
        npv2 = (capex*tur_nr[1] + mp_cost2 + cable_cost2 + LP*tur_nr[1]) * CRF + OpexAnnual*tur_nr[1]
        npv3 = (capex*tur_nr[2] + mp_cost3 + cable_cost3 + LP*tur_nr[2]) * CRF + OpexAnnual*tur_nr[2]
        aep1 = aep.isel(wt=slice(0,tur_nr[0])).sum().item()
        aep2 = aep.isel(wt=slice(tur_nr[0],sum(tur_nr[0:2]))).sum().item()
        aep3 = aep.isel(wt=slice(sum(tur_nr[0:2]),None)).sum().item()
        lcoe1 = npv1 / aep1
        lcoe2 = npv2 / aep2
        lcoe3 = npv3 / aep3
        metrics_recorder["aep_north"].append(aep1)
        metrics_recorder["aep_mid"].append(aep2)
        metrics_recorder["aep_south"].append(aep3)
        metrics_recorder["aep_all"].append(np.sum(aep).item())
        metrics_recorder["cable_cost_north"].append(cable_cost1)
        metrics_recorder["cable_cost_mid"].append(cable_cost2)
        metrics_recorder["cable_cost_south"].append(cable_cost3)
        metrics_recorder["cable_cost_all"].append(cable_cost)
        metrics_recorder["mp_cost_north"].append(mp_cost1)
        metrics_recorder["mp_cost_mid"].append(mp_cost2)
        metrics_recorder["mp_cost_south"].append(mp_cost3)
        metrics_recorder["mp_cost_all"].append(sum(mp_cost))
        metrics_recorder["lcoe_north"].append(lcoe1)
        metrics_recorder["lcoe_mid"].append(lcoe2)
        metrics_recorder["lcoe_south"].append(lcoe3)
        metrics_recorder["lcoe_all"].append(lcoe)
    elif Mode == 'competitive':
        # current zone
        metrics_recorder["aep_" + curzone].append(aep.isel(wt=slice(0,tur_nr[wf[curzone]])).sum().item())
        metrics_recorder["cable_cost_" + curzone].append(cable_cost)
        metrics_recorder["mp_cost_" + curzone].append(sum(mp_cost))
        metrics_recorder["lcoe_" + curzone].append(lcoe)
        npv_all = [npv]
        
        # 0 for non-neighbours
        for n in range(len(nnb)):
            metrics_recorder["aep_" + nnb[n]].append(0)
            metrics_recorder["cable_cost_" + nnb[n]].append(0)
            metrics_recorder["mp_cost_" + nnb[n]].append(0)
            metrics_recorder["lcoe_" + nnb[n]].append(0)
            
        # neighbours
        for n in range(len(nb)):
            if n == 0:
                aep_n = aep.isel(wt=slice(tur_nr[wf[curzone]],tur_nr[wf[curzone]] + tur_nr[wf[nb[n]]])).sum().item()
            elif n == 1:
                aep_n = aep.isel(wt=slice(tur_nr[wf[curzone]] + tur_nr[wf[nb[n-1]]],None)).sum().item()
            metrics_recorder["aep_" + nb[n]].append(aep_n)
            metrics_recorder["cable_cost_" + nb[n]].append(cable_cost_n[wf[nb[n]]])
            metrics_recorder["mp_cost_" + nb[n]].append(mp_cost_n[wf[nb[n]]])
            npv_n = (capex*tur_nr[wf[nb[n]]] + mp_cost_n[wf[nb[n]]] + cable_cost_n[wf[nb[n]]] + LP*tur_nr[wf[nb[n]]]) * CRF + OpexAnnual*tur_nr[wf[nb[n]]]
            metrics_recorder["lcoe_" + nb[n]].append(npv_n / aep_n)
            npv_all.append(npv_n)
            
        # total
        metrics_recorder["aep_all"].append(np.sum(aep).item())
        metrics_recorder["cable_cost_all"].append(sum(cable_cost_n) + cable_cost)
        metrics_recorder["mp_cost_all"].append(sum(mp_cost_n) + sum(mp_cost))
        metrics_recorder["lcoe_all"].append(sum(npv_all)/np.sum(aep).item())
        
        # for cabling plan, add cabling of foregoing optimization
        if len(nb) == 1:
            metrics_recorder["cable_u"][-1] = [x+1+tur_nr[wf[curzone]] if x != 1 else x for x in cable_u_n[0]] + [x+1 for x in metrics_recorder["cable_u"][-1]]
            metrics_recorder["cable_v"][-1] = [y+1+tur_nr[wf[curzone]] if y != 1 else y for y in cable_v_n[0]] + [y+1 for y in metrics_recorder["cable_v"][-1]]
            metrics_recorder["cable_type"][-1] = cable_type_n[0] + metrics_recorder["cable_type"][-1]
        elif len(nb) == 2:
            metrics_recorder["cable_u"][-1] = [x+2+tur_nr[wf[curzone]] if x != 1 else x for x in cable_u_n[0]] + [x+2+tur_nr[wf[curzone]]+tur_nr[wf[nb[0]]] if x != 1 else x+1 for x in cable_u_n[1]] + [x+2 for x in metrics_recorder["cable_u"][-1]]
            metrics_recorder["cable_v"][-1] = [y+2+tur_nr[wf[curzone]] if y != 1 else y for y in cable_v_n[0]] + [y+2+tur_nr[wf[curzone]]+tur_nr[wf[nb[0]]] if y != 1 else y+1 for y in cable_v_n[1]] + [y+2 for y in metrics_recorder["cable_v"][-1]]
            metrics_recorder["cable_type"][-1] = cable_type_n[0] + cable_type_n[1] + metrics_recorder["cable_type"][-1]
    # for global variable
    aep = aep.isel(wt=slice(0,len(x))).sum().item()
    return lcoe # $/MWh

#%% Objective gradient function
def wrap_depth(s): 
    return depth_interp(*np.split(s, 2))

depth_grad = fd(wrap_depth, step=0.01)

def get_depth_grads(x, y):
    grads = []
    for ii in range(len(x)):
       grads.append(depth_grad([x[ii], y[ii]]))
    return np.array(grads)

def lcoe_jac(x, y, **kwargs):
    global aep, cable_cost, dcable_cost, mp_cost, dmp_cost
    # 1.) aep
    wd = np.arange(0, 360, 1)
    if nf:
        daep = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=np.concatenate((x,xn)), y=np.concatenate((y,yn)), ws=ws, TI=TI, wd=wd)[:tur_nr[wf[curzone]],:tur_nr[wf[curzone]]] * 1e3
    else:
        daep = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=x, y=y, ws=ws, TI=TI, wd=wd) * 1e3
    #
    # 2.) monopile costs
    # have been calculated earlier (global variable)
    #
    # 3.) Cable costs
    # have been calculated earlier (global variable)
    #
    # 4.) lcoe
    CRF = d / (1 - (1 + d) ** -life)
    dlcoe = (CRF*(dmp_cost+dcable_cost.T)*aep - ((capex*len(x)+LP*len(x)+np.sum(mp_cost)+cable_cost)*CRF+OpexAnnual*len(x))*daep) / (aep ** 2)
    return dlcoe

# Verify gradients
# lcoe = lcoe_func(x0, y0)
# lcoe_grad = lcoe_jac(x0, y0)
# def wrap_lcoe(s): return lcoe_func(*np.split(s, 2))
# grad = fd(wrap_lcoe, 0.000001)(np.append(x0, y0))
# print('difference between fd and analytic grads: ')
# print(grad - np.array(lcoe_grad).flatten())

#%% Prepare initial layout
def random_points_in_polygon(polygon, nr_turbines):
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds  # Bounding box of the polygon
    
    while len(points) < nr_turbines:
        # Generate a random point within the bounding box
        random_point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
        
        # Check if the point is within the polygon
        if polygon.contains(random_point):
            points.append((random_point.x, random_point.y))
    
    return points

# Function to generate random points in polygons
def get_random_points(wind_farm_name, nr_turbines):
    polygon = polygons[wind_farm_name]
    return random_points_in_polygon(polygon, nr_turbines)

# Create polygons dynamically
polygons = {
    name: Polygon(list(zip(boundary[:, 0], boundary[:, 1])))
    for name, boundary in boundaries.items()
}
# Generate initial layouts for each wind farm
points = {name: get_random_points(name, tur_nr[wf[name]]) for name in wf}

#%% General constraints
# Min spacing
min_spacing_m = 2 * windTurbines.diameter()  #minimum inter-turbine spacing in meters

#%% Recorder
metrics_recorder = {
    "iteration": [],
    "opt_nr": [],
    "aep": [],
    "mp_cost": [],
    "cable_cost": [],
    "lcoe": [],
    "cable_u": [],
    "cable_v": [],
    "cable_type": [],
    "aep_north": [],
    "aep_mid": [],
    "aep_south": [],
    "aep_all": [],
    "mp_cost_north": [],
    "mp_cost_mid": [],
    "mp_cost_south": [],
    "mp_cost_all": [],
    "cable_cost_north": [],
    "cable_cost_mid": [],
    "cable_cost_south": [],
    "cable_cost_all": [],
    "lcoe_north": [],
    "lcoe_mid": [],
    "lcoe_south": [],
    "lcoe_all": [],
    "x": [],
    "y": [],
    "x_final": [],
    "y_final": [],
    "lcoe_final": []
}

#%% Convergence plotting script
def plot_convergence(mr=None,item=None,plotstr=None,obj=1,overall=0):
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item+'_north']], label='North', linewidth = 1)
    plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item+'_mid']], label='Mid', linewidth = 1)
    plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item+'_south']], label='South', linewidth = 1)
    if obj:
        plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item]], label='Overall', linewidth = 1)
    if overall:
        plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item+'_all']], label='Overall', linewidth = 1)
    FS = 9
    plt.legend(fontsize=FS)
    plt.grid()
    plt.xlabel('Iteration',fontsize=FS)
    plt.ylabel(plotstr,fontsize=FS)
    plt.title(plotstr,fontsize=FS+2)
    plt.xticks(fontsize=FS-1)
    plt.yticks(fontsize=FS-1)

#%% Cooperative design
if Mode == 'cooperative':
    # Initital Layout
    x0 = np.concatenate([np.array(points[name])[:, 0] for name in wf])
    y0 = np.concatenate([np.array(points[name])[:, 1] for name in wf])
    
    # Constraint
    joint_boundaries = MultiWFPolygonBoundaryConstraint(
        {idx: boundaries[name] for idx, name in enumerate(wf)},  # Boundary mapping
        turbine_groups={idx: np.arange(sum(tur_nr[:idx]), sum(tur_nr[:idx + 1])) for idx in range(len(wf))}  # Turbine groups
    )

    # Options
    SepCabling = True       # Indicate if cabling is only allowed within each zone or cross-zonal
    Sx = Subs_x
    Sy = Subs_y
    
    # Optimization setup
    tf = TopFarmProblem(
            design_vars = {'x':x0, 'y':y0},         
            cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=sum(tur_nr), cost_function=lcoe_func, objective=True, cost_gradient_function=lcoe_jac, maximize=False),
            constraints = DistanceConstraintAggregation([SpacingConstraint(min_spacing_m), joint_boundaries], sum(tur_nr), min_spacing_m, windTurbines), 
            driver = EasySGDDriver(maxiter=3000, learning_rate=windTurbines.diameter(), max_time=1008000, gamma_min_factor=0.1, speedupSGD=True, sgd_thresh=0.12),
            plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sx, Sy=Sy, cables=cables, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=[]),
            )
    
    # Run
    tic = time.time()
    cost, state, recorder = tf.optimize()
    toc = time.time()
    print('Optimization with SGD took: {:.0f}s'.format(toc-tic), ' with a total constraint violation of ', recorder['sgd_constraint'][-1])

    # Store
    metrics_recorder["lcoe_final"].append([cost])
    metrics_recorder["x_final"].append(state['x'].tolist())
    metrics_recorder["y_final"].append(state['y'].tolist())

    # Plot history
    if plot_conv:
        # overall metrics (relative)
        plt.figure(figsize=(5, 3))
        plt.plot(np.arange(metrics_recorder['iteration'][-1]), [(x - metrics_recorder['lcoe'][0]) / metrics_recorder['lcoe'][0] * 100 for x in metrics_recorder['lcoe']], label='lcoe', linewidth = 1)
        plt.plot(np.arange(metrics_recorder['iteration'][-1]), [(x - metrics_recorder['aep'][0]) / metrics_recorder['aep'][0] * 100 for x in metrics_recorder['aep']], label='aep', linewidth = 1)
        plt.plot(np.arange(metrics_recorder['iteration'][-1]), [(x - metrics_recorder['cable_cost'][0]) / metrics_recorder['cable_cost'][0] * 100 for x in metrics_recorder['cable_cost']], label='cable cost',linewidth = 1)
        plt.plot(np.arange(metrics_recorder['iteration'][-1]), [(x - metrics_recorder['mp_cost'][0]) / metrics_recorder['mp_cost'][0] * 100 for x in metrics_recorder['mp_cost']], label='monopile cost',linewidth = 1)
        plt.legend(fontsize=7)
        plt.grid()
        plt.xlabel('Iteration',fontsize=6)
        plt.ylabel('Rel. Diff. wrt baseline (%)',fontsize=6)
        plt.title('Convergence behaviour',fontsize=7)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        #
        # lcoe
        plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=1,overall=0)
        # aep
        plot_convergence(mr=metrics_recorder,item='aep',plotstr='AEP (GWh)',obj=0,overall=0)
        # cable cost
        plot_convergence(mr=metrics_recorder,item='cable_cost',plotstr='Cable Cost (€)',obj=0,overall=0)
        # monopile cost
        plot_convergence(mr=metrics_recorder,item='mp_cost',plotstr='Monopile Cost (€)',obj=0,overall=0)
    
    # Save to a file
    with open("metric_recorder_cooperative.pkl", "wb") as file:
        pickle.dump({"metrics_recorder": metrics_recorder, "state": state}, file)

#%% Competitive design
elif Mode == 'competitive':
    # general options
    SepCabling = False
    boundplot = list(boundaries.values())
    
    # dicts to store values
    res_cable_cost = {}
    res_cable_u = {}
    res_cable_v = {}
    res_cable_type = {}
    res_mp_cost ={}
    res_x = {}
    res_y = {}
    
    # go through each zone as specified in Sequence
    for i in range(len(Sequence)):
        opt_nr = i+1
        # Initial Layout
        x0 = np.array(points[Sequence[i]]).T[0]
        y0 = np.array(points[Sequence[i]]).T[1]
    
        # Constraint
        constraint_comp = XYBoundaryConstraint([InclusionZone(boundaries[Sequence[i]])], 'multi_polygon')
        
        # Options
        Sx = [Subs_x[wf[Sequence[i]]]]
        Sy = [Subs_y[wf[Sequence[i]]]]
        nb = Sequence[:i]
        curzone = Sequence[i]
        nnb = Sequence[i+1:]
        
        if nb:
            nf = True
            xn = np.array([])
            yn = np.array([])
            for j in range(len(nb)):
                xn = np.concatenate([xn, res_x[nb[j]].tolist()])
                yn = np.concatenate([yn, res_y[nb[j]].tolist()])
                cable_cost_n[j] = res_cable_cost[nb[j]]
                cable_u_n[j] = res_cable_u[nb[j]]
                cable_v_n[j] = res_cable_v[nb[j]]
                cable_type_n[j] = res_cable_type[nb[j]]
                mp_cost_n [j]= res_mp_cost[nb[j]]
            
        # Optimization Setup
        tf = TopFarmProblem(
                design_vars = {'x':x0, 'y':y0},         
                cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=tur_nr[wf[Sequence[i]]], cost_function=lcoe_func, objective=True, cost_gradient_function=lcoe_jac, maximize=False),
                constraints = DistanceConstraintAggregation([SpacingConstraint(min_spacing_m), constraint_comp],tur_nr[wf[Sequence[i]]], min_spacing_m, windTurbines), 
                driver = EasySGDDriver(maxiter=3000, learning_rate=windTurbines.diameter(), max_time=1008000, gamma_min_factor=0.1, speedupSGD=True, sgd_thresh=0.12),
                plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Subs_x[:len(nb)+1], Sy=Subs_y[:len(nb)+1], cables=cables, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot, opt_nr=opt_nr)
                )
        
        # Run
        tic = time.time()
        cost, state, recorder = tf.optimize()
        toc = time.time()
        print('Optimization with SGD took: {:.0f}s'.format(toc-tic), ' with a total constraint violation of ', recorder['sgd_constraint'][-1])
        
        # Store
        metrics_recorder["lcoe_final"].append([cost])
        metrics_recorder["x_final"].append(state['x'].tolist())
        metrics_recorder["y_final"].append(state['y'].tolist())
        res_x[Sequence[i]] = state['x']
        res_y[Sequence[i]] = state['y']
        res_cable_cost[Sequence[i]] = metrics_recorder['cable_cost'][-1]
        res_cable_u[Sequence[i]] = [x-i for x in metrics_recorder["cable_u"][-1][-tur_nr[wf[Sequence[i]]]:]]
        res_cable_v[Sequence[i]] = [y-i for y in metrics_recorder["cable_v"][-1][-tur_nr[wf[Sequence[i]]]:]]
        res_cable_type[Sequence[i]] = metrics_recorder['cable_type'][-1][-tur_nr[wf[Sequence[i]]]:]
        res_mp_cost[Sequence[i]] = metrics_recorder['mp_cost'][-1]
        
        # Save recorder to file
        with open("metric_recorder_sequential_" + ''.join([entry[0] for entry in Sequence]) + ".pkl", "wb") as file:
            pickle.dump({"metrics_recorder": metrics_recorder, "state": state}, file)
     
    # Plot history
    if plot_conv:
        # lcoe
        plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=1)
        # plt.ylim([52,57])
        # plt.savefig("plot_lcoe2.svg", format="svg", bbox_inches="tight")
        # aep
        plot_convergence(mr=metrics_recorder,item='aep',plotstr='AEP (GWh)',obj=0,overall=0)
        # cable cost
        plot_convergence(mr=metrics_recorder,item='cable_cost',plotstr='Cable Cost (€)',obj=0,overall=0)
        # monopile cost
        plot_convergence(mr=metrics_recorder,item='mp_cost',plotstr='Monopile Cost (€)',obj=0,overall=0)
        