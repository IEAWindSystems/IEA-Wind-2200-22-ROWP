import numpy as np
from py_wake.rotor_avg_models import RotorCenter
import time
import matplotlib.pyplot as plt
import xarray as xr
import os
import utm
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
from scipy.interpolate import RegularGridInterpolator, interp1d
from ssms.CalculateMass import CalculateMass
from ed_win.wind_farm_network import WindFarmNetwork
from shapely.geometry import Point, Polygon
from topfarm.constraint_components.boundary import MultiWFPolygonBoundaryConstraint
np.random.seed(2)
#
#%% INPUTS
# farm layout
Mode = 'competitive'    # 'cooperative' or 'competitive'
tur_nr = 34             # Desired turbine number in optimized farm
Model = 'gauss'
plot_conv = True

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
capex = 5.5258e7        # $2010 per turbine, excl. Monopile, from DETECT for HKN scaled (22MW turbines)
OpexAnnual = 1.3564e6   # $2010 per turbine, annual OPEX, from DETECT for HKN scaled (22MW turbines)
LP = 2.6447e+06         # $2010 per turbine, liquidation proceeds, from DETECT for HKN scaled (22MW turbines)
#
# cable data [cross section, capacity, price]
cables = np.array([[500, 3, 206], [800, 5, 287], [1000, 7, 406]])
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
   ws = resource_dat['wind_resource']['wind_speed']
   site = XRSite(
          ds=xr.Dataset(data_vars=
                           {'Sector_frequency': ('wd', freq['data']), 
                            'Weibull_A': ('wd', A['data']), 
                            'Weibull_k': ('wd', k['data']), 
                            'TI': (resource_dat['wind_resource']['turbulence_intensity']['dims'][0], resource_dat['wind_resource']['turbulence_intensity']['data'])
                            },
                         coords={'wd': wd, 'ws': ws}))
   
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

Subs_x = system_dat['wind_farm']['electrical_substations']['electrical_substation']['coordinates']['x']
Subs_y = system_dat['wind_farm']['electrical_substations']['electrical_substation']['coordinates']['y']

# Boundaries and exclusion zones
b = system_dat['site']
north_boundary = np.array([b['boundaries']['polygons'][0]['x'], b['boundaries']['polygons'][0]['y']]).T
mid_boundary = np.array([b['boundaries']['polygons'][1]['x'], b['boundaries']['polygons'][1]['y']]).T
south_boundary = np.array([b['boundaries']['polygons'][2]['x'], b['boundaries']['polygons'][2]['y']]).T

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
P_interpolator = interp1d(np.cumsum(sum(P)), ws, kind='linear')  # interpolator to get mean wind sped (@50% probability)
V_ave = P_interpolator(0.5).tolist()
masses = []
for z in depths:
   cur_mass = CalculateMass(RP=rp/1e6, D=rd, HTrans=ph, HHub_Ratio=hh/rd, WaterDepth=z, WaveHeight=swh, WavePeriod=swp, WindSpeed=V_ave)
   masses.append(cur_mass[0][0])

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

#%% Objective function
def lcoe_func(x, y, **kwargs):
    global metrics_recorder, aep, cable_cost, dcable_cost, mp_cost, dmp_cost, cable_cost_n, cable_u_n, cable_v_n, cable_type_n, mp_cost_n
    #
    # 1.) aep
    wd = np.arange(0, 360, 1)
    if nf:
        aep = wake_model(x=np.concatenate((x,xn)), y=np.concatenate((y,yn)), wd=wd, ws=ws, TI=TI).aep() * 1e3
    else:
        aep = wake_model(x=x, y=y, wd=wd, ws=ws, TI=TI).aep() * 1e3
    #
    # 2.) monopile costs
    depths = depth_interp(x, y)
    masses = []
    dmasses = []
    for water_depth in depths:
       dmasses.append(polynomial_gradients(water_depth))
       masses.append(polynomial(water_depth))
    mp_cost = 2.25 * np.array(masses)
    # gradients (for function later, to avoid double calculus)
    dmasses = (np.array(dmasses) * get_depth_grads(x, y)[:, 0, :].T)
    dmp_cost = 2.25 * dmasses
    #
    # 3.) Cable costs
    # initialize
    if SepCabling:
        xc1 = x[:tur_nr]
        xc2 = x[tur_nr:2*tur_nr]
        xc3 = x[2*tur_nr:]
        yc1 = y[:tur_nr]
        yc2 = y[tur_nr:2*tur_nr]
        yc3 = y[2*tur_nr:]
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
        metrics_recorder["cable_u"].append([x+len(Sx)-1 if x != 1 else x for x in cab_data1['u'].tolist()] + [x+len(Sx)-1+tur_nr if x != 1 else x+1 for x in cab_data2['u'].tolist()] + [x+len(Sx)-1+tur_nr*2 if x != 1 else x+2 for x in cab_data3['u'].tolist()])
        metrics_recorder["cable_v"].append([y+len(Sx)-1 if y != 1 else y for y in cab_data1['v'].tolist()] + [y+len(Sx)-1+tur_nr if y != 1 else y++1 for y in cab_data2['v'].tolist()] + [y+len(Sx)-1+tur_nr*2 if y != 1 else y+2 for y in cab_data3['v'].tolist()])
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
    metrics_recorder["aep"].append(aep.isel(wt=slice(0,len(x))).sum().item())
    metrics_recorder["mp_cost"].append(sum(mp_cost))
    metrics_recorder["cable_cost"].append(cable_cost)
    metrics_recorder["lcoe"].append(lcoe)
    metrics_recorder["x"].append(x)
    metrics_recorder["y"].append(y)
    # performance of individual zones
    if Mode == 'cooperative':
        # store the values of the individual zones
        mp_cost1 = np.sum(mp_cost[:tur_nr])
        mp_cost2 = np.sum(mp_cost[tur_nr:tur_nr*2])
        mp_cost3 = np.sum(mp_cost[tur_nr*2:])
        cable_cost1 = G1.cost
        cable_cost2 = G2.cost 
        cable_cost3 = G3.cost
        npv1 = (capex*tur_nr + mp_cost1 + cable_cost1 + LP*tur_nr) * CRF + OpexAnnual*tur_nr
        npv2 = (capex*tur_nr + mp_cost2 + cable_cost2 + LP*tur_nr) * CRF + OpexAnnual*tur_nr
        npv3 = (capex*tur_nr + mp_cost3 + cable_cost3 + LP*tur_nr) * CRF + OpexAnnual*tur_nr
        aep1 = aep.isel(wt=slice(0,tur_nr)).sum().item()
        aep2 = aep.isel(wt=slice(tur_nr,2*tur_nr)).sum().item()
        aep3 = aep.isel(wt=slice(2*tur_nr,None)).sum().item()
        lcoe1 = npv1 / aep1
        lcoe2 = npv2 / aep2
        lcoe3 = npv3 / aep3
        metrics_recorder["aep1"].append(aep1)
        metrics_recorder["aep2"].append(aep2)
        metrics_recorder["aep3"].append(aep3)
        metrics_recorder["aep_all"].append(np.sum(aep).item())
        metrics_recorder["cable_cost1"].append(cable_cost1)
        metrics_recorder["cable_cost2"].append(cable_cost2)
        metrics_recorder["cable_cost3"].append(cable_cost3)
        metrics_recorder["cable_cost_all"].append(cable_cost)
        metrics_recorder["mp_cost1"].append(mp_cost1)
        metrics_recorder["mp_cost2"].append(mp_cost2)
        metrics_recorder["mp_cost3"].append(mp_cost3)
        metrics_recorder["mp_cost_all"].append(sum(mp_cost))
        metrics_recorder["lcoe1"].append(lcoe1)
        metrics_recorder["lcoe2"].append(lcoe2)
        metrics_recorder["lcoe3"].append(lcoe3)
        metrics_recorder["lcoe_all"].append(lcoe)
    elif Mode == 'competitive':
        if len(xn) == 0:
            metrics_recorder["aep1"].append(np.sum(aep).item())
            metrics_recorder["aep2"].append(0)
            metrics_recorder["aep3"].append(0)
            metrics_recorder["aep_all"].append(np.sum(aep).item())
            metrics_recorder["cable_cost1"].append(cable_cost)
            metrics_recorder["cable_cost2"].append(0)
            metrics_recorder["cable_cost3"].append(0)
            metrics_recorder["cable_cost_all"].append(cable_cost)
            metrics_recorder["mp_cost1"].append(sum(mp_cost))
            metrics_recorder["mp_cost2"].append(0)
            metrics_recorder["mp_cost3"].append(0)
            metrics_recorder["mp_cost_all"].append(sum(mp_cost))
            metrics_recorder["lcoe1"].append(lcoe)
            metrics_recorder["lcoe2"].append(0)
            metrics_recorder["lcoe3"].append(0)
            metrics_recorder["lcoe_all"].append(lcoe)
        elif len(xn) == tur_nr:
            aep1 = aep.isel(wt=slice(tur_nr,2*tur_nr)).sum().item()
            aep2 = aep.isel(wt=slice(0,tur_nr)).sum().item()
            npv1 = (capex*tur_nr + mp_cost_n[0] + cable_cost_n[0] + LP*tur_nr) * CRF + OpexAnnual*tur_nr
            lcoe_all = (npv1+npv) / (aep1+aep2)
            metrics_recorder["aep1"].append(aep1)
            metrics_recorder["aep2"].append(aep2)
            metrics_recorder["aep3"].append(0)
            metrics_recorder["aep_all"].append(aep1 + aep2)
            metrics_recorder["cable_cost1"].append(cable_cost_n[0])
            metrics_recorder["cable_cost2"].append(cable_cost)
            metrics_recorder["cable_cost3"].append(0)
            metrics_recorder["cable_cost_all"].append(sum(cable_cost_n) + cable_cost)
            metrics_recorder["mp_cost1"].append(mp_cost_n[0])
            metrics_recorder["mp_cost2"].append(sum(mp_cost))
            metrics_recorder["mp_cost3"].append(0)
            metrics_recorder["mp_cost_all"].append(sum(mp_cost_n) + sum(mp_cost))
            metrics_recorder["lcoe1"].append(npv1 / aep1)
            metrics_recorder["lcoe2"].append(npv / aep2)
            metrics_recorder["lcoe3"].append(0)
            metrics_recorder["lcoe_all"].append(lcoe_all)
            # add cabling of foregoing optimization
            metrics_recorder["cable_u"][-1] = [x+1+tur_nr if x != 1 else x for x in cable_u_n[0]] + [x+1 for x in metrics_recorder["cable_u"][-1]]
            metrics_recorder["cable_v"][-1] = [y+1+tur_nr if y != 1 else y for y in cable_v_n[0]] + [y+1 for y in metrics_recorder["cable_v"][-1]]
            metrics_recorder["cable_type"][-1] = cable_type_n[0] + metrics_recorder["cable_type"][-1]
        elif len(xn) == tur_nr*2:
            aep1 = aep.isel(wt=slice(tur_nr,2*tur_nr)).sum().item()
            aep2 = aep.isel(wt=slice(2*tur_nr,None)).sum().item()
            aep3 = aep.isel(wt=slice(0,tur_nr)).sum().item()
            npv1 = (capex*tur_nr + mp_cost_n[0] + cable_cost_n[0] + LP*tur_nr) * CRF + OpexAnnual*tur_nr
            npv2 = (capex*tur_nr + mp_cost_n[1] + cable_cost_n[1] + LP*tur_nr) * CRF + OpexAnnual*tur_nr
            lcoe_all = (npv1+npv2+npv) / (aep1+aep2+aep3)
            metrics_recorder["aep1"].append(aep1)
            metrics_recorder["aep2"].append(aep2)
            metrics_recorder["aep3"].append(aep3)
            metrics_recorder["aep_all"].append(aep1+aep2+aep3)
            metrics_recorder["cable_cost1"].append(cable_cost_n[0])
            metrics_recorder["cable_cost2"].append(cable_cost_n[1])
            metrics_recorder["cable_cost3"].append(cable_cost)
            metrics_recorder["cable_cost_all"].append(sum(cable_cost_n) + cable_cost)
            metrics_recorder["mp_cost1"].append(mp_cost_n[0])
            metrics_recorder["mp_cost2"].append(mp_cost_n[1])
            metrics_recorder["mp_cost3"].append(sum(mp_cost))
            metrics_recorder["mp_cost_all"].append(sum(mp_cost_n) + sum(mp_cost))
            metrics_recorder["lcoe1"].append(npv1 / aep1)
            metrics_recorder["lcoe2"].append(npv2 / aep2)
            metrics_recorder["lcoe3"].append(npv / aep3)
            metrics_recorder["lcoe_all"].append(lcoe_all)
            # add cabling of foregoing optimization
            metrics_recorder["cable_u"][-1] = [x+2+tur_nr if x != 1 else x for x in cable_u_n[0]] + [x+2+tur_nr*2 if x != 1 else x+1 for x in cable_u_n[1]] + [x+2 for x in metrics_recorder["cable_u"][-1]]
            metrics_recorder["cable_v"][-1] = [y+2+tur_nr if y != 1 else y for y in cable_v_n[0]] + [y+2+tur_nr*2 if y != 1 else y+1 for y in cable_v_n[1]] + [y+2 for y in metrics_recorder["cable_v"][-1]]
            metrics_recorder["cable_type"][-1] = cable_type_n[0] + cable_type_n[1] + metrics_recorder["cable_type"][-1]
    # for global variable
    aep = np.sum(aep).item()
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
        daep = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=np.concatenate((x,xn)), y=np.concatenate((y,yn)), ws=ws, TI=TI, wd=wd)[:tur_nr,:tur_nr] * 1e3
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
def random_points_in_polygon(polygon, tur_nr):
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds  # Bounding box of the polygon
    
    while len(points) < tur_nr:
        # Generate a random point within the bounding box
        random_point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
        
        # Check if the point is within the polygon
        if polygon.contains(random_point):
            points.append((random_point.x, random_point.y))
    
    return points

# Generate random points within polygon
polygon_north = Polygon(list(zip(north_boundary[:,0], north_boundary[:,1])))
polygon_middle = Polygon(list(zip(mid_boundary[:,0], mid_boundary[:,1])))
polygon_south = Polygon(list(zip(south_boundary[:,0], south_boundary[:,1])))
points_n = random_points_in_polygon(polygon_north, tur_nr)
points_m = random_points_in_polygon(polygon_middle, tur_nr)
points_s = random_points_in_polygon(polygon_south, tur_nr)

#%% General constraints
# Min spacing
min_spacing_m = 2 * windTurbines.diameter()  #minimum inter-turbine spacing in meters

#%% Recorder
metrics_recorder = {
    "iteration": [],
    "aep": [],
    "mp_cost": [],
    "cable_cost": [],
    "lcoe": [],
    "cable_u": [],
    "cable_v": [],
    "cable_type": [],
    "aep1": [],
    "aep2": [],
    "aep3": [],
    "aep_all": [],
    "mp_cost1": [],
    "mp_cost2": [],
    "mp_cost3": [],
    "mp_cost_all": [],
    "cable_cost1": [],
    "cable_cost2": [],
    "cable_cost3": [],
    "cable_cost_all": [],
    "lcoe1": [],
    "lcoe2": [],
    "lcoe3": [],
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
    plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item+'1']], label='Zone 1', linewidth = 1)
    plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item+'2']], label='Zone 2', linewidth = 1)
    plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item+'3']], label='Zone 3', linewidth = 1)
    if obj:
        plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item]], label='Overall', linewidth = 1)
    if overall:
        plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item+'_all']], label='Overall', linewidth = 1)
    plt.legend(fontsize=7)
    plt.grid()
    plt.xlabel('Iteration',fontsize=6)
    plt.ylabel(plotstr,fontsize=6)
    plt.title(plotstr,fontsize=7)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

#%% Cooperative design
if Mode == 'cooperative':
    # Initital Layout
    x0 = np.append(np.array(points_n).T[0], np.append(np.array(points_m).T[0], np.array(points_s).T[0]))
    y0 = np.append(np.array(points_n).T[1], np.append(np.array(points_m).T[1], np.array(points_s).T[1]))
    
    # Constraint
    joint_boundaries = MultiWFPolygonBoundaryConstraint({0: north_boundary, 1: mid_boundary, 2: south_boundary}, turbine_groups={0: np.arange(tur_nr), 1: np.arange(tur_nr, 2 * tur_nr), 2: np.arange(2 * tur_nr, 3 * tur_nr)})

    # Options
    SepCabling = True       # Indicate if cabling is only allowed within each zone or cross-zonal
    Sx = Subs_x
    Sy = Subs_y
    
    # Optimization setup
    tf = TopFarmProblem(
            design_vars = {'x':x0, 'y':y0},         
            cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=3 * tur_nr, cost_function=lcoe_func, objective=True, cost_gradient_function=lcoe_jac, maximize=False),
            constraints = DistanceConstraintAggregation([SpacingConstraint(min_spacing_m), joint_boundaries], 3 * tur_nr, min_spacing_m, windTurbines), 
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
    import pickle
    with open("metric_recorder_cooperative.pkl", "wb") as file:
        pickle.dump({"metrics_recorder": metrics_recorder, "state": state}, file)

#%% Competitive design
elif Mode == 'competitive':
    # ----------------------------------------
    # 1.) Optimize Northern Zone
    #
    # Initital layout
    x0 = np.array(points_n).T[0]
    y0 = np.array(points_n).T[1]
    
    # Constraint
    constraint_comp = XYBoundaryConstraint([InclusionZone(north_boundary)], 'multi_polygon')
    
    # Options
    SepCabling = False       # Indicate if cabling is only allowed within each zone or cross-zonal
    Sx = [Subs_x[0]]
    Sy = [Subs_y[0]]
    boundplot = [north_boundary,mid_boundary,south_boundary]
    
    # Optimization Setup
    tf = TopFarmProblem(
            design_vars = {'x':x0, 'y':y0},         
            cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=tur_nr, cost_function=lcoe_func, objective=True, cost_gradient_function=lcoe_jac, maximize=False),
            constraints = DistanceConstraintAggregation([SpacingConstraint(min_spacing_m), constraint_comp],tur_nr, min_spacing_m, windTurbines), 
            driver = EasySGDDriver(maxiter=3000, learning_rate=windTurbines.diameter(), max_time=1008000, gamma_min_factor=0.1, speedupSGD=True, sgd_thresh=0.12),
            plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sx, Sy=Sy, cables=cables, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot),
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
    
    # ----------------------------------------
    # 2.) Optimize Mid Zone
    #
    # Initital layout
    x0 = np.array(points_m).T[0]
    y0 = np.array(points_m).T[1]
    
    # Constraint
    constraint_comp = XYBoundaryConstraint([InclusionZone(mid_boundary)], 'multi_polygon')
    
    # Options
    Sx = [Subs_x[1]]
    Sy = [Subs_y[1]]
    nf = True      # Neighbour wind farm coordinates = results from foregoing optimization
    xn = state['x']
    yn = state['y']
    cable_cost_n[0] = metrics_recorder['cable_cost'][-1]
    cable_u_n[0] = metrics_recorder['cable_u'][-1]
    cable_v_n[0] = metrics_recorder['cable_v'][-1]
    cable_type_n[0] = metrics_recorder['cable_type'][-1]
    mp_cost_n [0]= metrics_recorder['mp_cost'][-1]
    
    # Optimization Setup
    tf = TopFarmProblem(
            design_vars = {'x':x0, 'y':y0},         
            cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=tur_nr, cost_function=lcoe_func, objective=True, cost_gradient_function=lcoe_jac, maximize=False),
            constraints = DistanceConstraintAggregation([SpacingConstraint(min_spacing_m), constraint_comp],tur_nr, min_spacing_m, windTurbines), 
            driver = EasySGDDriver(maxiter=3000, learning_rate=windTurbines.diameter(), max_time=1008000, gamma_min_factor=0.1, speedupSGD=True, sgd_thresh=0.12),
            plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Subs_x[:2], Sy=Subs_y[:2], cables=cables, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot),
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

    # ----------------------------------------
    # 3.) Optimize South Zone
    #
    # Initital layout
    x0 = np.array(points_s).T[0]
    y0 = np.array(points_s).T[1]
    
    # Constraint
    constraint_comp = XYBoundaryConstraint([InclusionZone(south_boundary)], 'multi_polygon')
    
    # Options
    Sx = [Subs_x[2]]
    Sy = [Subs_y[2]]
    nf = True      # Neighbour wind farm coordinates = results from foregoing optimization
    xn = np.concatenate([xn, state['x']])
    yn = np.concatenate([yn, state['y']])
    cable_cost_n[1] = metrics_recorder['cable_cost'][-1]
    cable_u_n[1] = [x-1 for x in metrics_recorder["cable_u"][-1][tur_nr:]]
    cable_v_n[1] = [y-1 for y in metrics_recorder["cable_v"][-1][tur_nr:]]
    cable_type_n[1] = metrics_recorder['cable_type'][-1][tur_nr:]
    mp_cost_n[1]= metrics_recorder['mp_cost'][-1]
    
    # Optimization Setup
    tf = TopFarmProblem(
            design_vars = {'x':x0, 'y':y0},         
            cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=tur_nr, cost_function=lcoe_func, objective=True, cost_gradient_function=lcoe_jac, maximize=False),
            constraints = DistanceConstraintAggregation([SpacingConstraint(min_spacing_m), constraint_comp],tur_nr, min_spacing_m, windTurbines), 
            driver = EasySGDDriver(maxiter=3000, learning_rate=windTurbines.diameter(), max_time=1008000, gamma_min_factor=0.1, speedupSGD=True, sgd_thresh=0.12),
            plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Subs_x, Sy=Subs_y, cables=cables, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot),
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
    
    # ----------------------------------------
    # 4. Plot history
    #
    if plot_conv:
        # lcoe
        plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=1)
        # aep
        plot_convergence(mr=metrics_recorder,item='aep',plotstr='AEP (GWh)',obj=0,overall=0)
        # cable cost
        plot_convergence(mr=metrics_recorder,item='cable_cost',plotstr='Cable Cost (€)',obj=0,overall=0)
        # monopile cost
        plot_convergence(mr=metrics_recorder,item='mp_cost',plotstr='Monopile Cost (€)',obj=0,overall=0)
        
    # ----------------------------------------
    # 5. Save to a file
    import pickle
    with open("metric_recorder_sequential.pkl", "wb") as file:
        pickle.dump({"metrics_recorder": metrics_recorder, "state": state}, file)