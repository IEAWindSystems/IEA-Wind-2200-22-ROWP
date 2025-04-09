import numpy as np
import pandas as pd
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
Mode = 'competitive'    # 'cooperative' or 'competitive' or 'evaluate_sequ' or 'evaluate_multiter'
Sequence = ['north','mid','south','north','mid','south','north','mid','south']
Model = 'gauss'
plot_conv = True
tur_nr = [33,33,34]     # Desired turbine number in optimized farm, from north to south!
obj = 'lcoe'            # 'lcoe' or 'aep'
plot_iter = True
plot_each = 10          # define in which interval a plot should be made

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

# cable data [cross section, capacity, price]
cables = np.array([[185,3,368.9], [400,5,428.9], [1000,7,737.1]])       # 110kV
# cables = np.array([[500, 3, 393], [800, 5, 522.4], [1000, 7, 615.5]]) # 66kV

#%% Load data and setup pywake
# system_dat = sys.argv[1]
system_dat = load_yaml(os.sep.join(['..', 'inputs', 'wind_energy_system.yaml']))
farm_dat = system_dat['wind_farm']
resource_dat = system_dat['site']['energy_resource']

# set up site
A = resource_dat['wind_resource']['weibull_a']
k = resource_dat['wind_resource']['weibull_k']
freq = resource_dat['wind_resource']['sector_probability']
wd = resource_dat['wind_resource']['wind_direction']
# ws = resource_dat['wind_resource']['wind_speed']
TI =  resource_dat['wind_resource']['turbulence_intensity']['data']
site = XRSite(
       ds=xr.Dataset(data_vars=
                        {'Sector_frequency': ('wd', freq['data']), 
                         'Weibull_A': ('wd', A['data']), 
                         'Weibull_k': ('wd', k['data']),
                         'TI': TI
                         },
                      coords={'wd': wd}))

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

# wake model
if Model == 'jensen':
    wake_model = NOJ(site, windTurbines, k=0.05, rotorAvgModel=RotorCenter())
elif Model == 'gauss':
    wake_model = BastankhahGaussian(site, windTurbines, rotorAvgModel=RotorCenter(), turbulenceModel=CrespoHernandez())

# wind resource
dirs = np.arange(0, 360, 1) #wind directions
ws = np.arange(cut_in, cut_out+1, 1)
freqs = site.local_wind([0], [0], wd=dirs, ws=ws).P_ilk[0, :, :].sum(1)     # all frequencies
freqs = freqs / freqs.sum()
Subs_x = system_dat['wind_farm']['electrical_substations']['electrical_substation']['coordinates']['x']
Subs_y = system_dat['wind_farm']['electrical_substations']['electrical_substation']['coordinates']['y']
Sub_x = {'north': Subs_x[0], 'mid': Subs_x[1], 'south': Subs_x[2]}
Sub_y = {'north': Subs_y[0], 'mid': Subs_y[1], 'south': Subs_y[2]}
# for sampling:
As = site.local_wind([0], [0], wd=dirs, ws=ws).Weibull_A_ilk[0, :, 0]               #weibull A
ks = site.local_wind([0], [0], wd=dirs, ws=ws).Weibull_k_ilk[0, :, 0]               #weibull k

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

# Bathymetry
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

# SGD sampling
sample = False
samps = 100    #number of samples 
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

#%% Function to create the random sampling of wind speed and wind directions
def sampling():
    idx = np.random.choice(np.arange(dirs.size), samps, p=freqs)
    wd = dirs[idx]
    A = As[idx]
    k = ks[idx]
    ws = A * np.random.weibull(k)
    return wd, ws

#%% Objective function
def lcoe_func(x, y, **kwargs):
    global metrics_recorder, aep, cable_cost, dcable_cost, mp_cost, dmp_cost, cable_cost_n, cable_u_n, cable_v_n, cable_type_n, mp_cost_n, wd_current, ws_current, Time
    # 1.) aep
    if sample:
        wd_current, ws_current = sampling()
        Time = True
    else:
        wd_current = np.arange(0, 360, 1)
        ws_current = ws
        Time = False
    if nf:
        aep = wake_model(x=np.concatenate((x,xn)), y=np.concatenate((y,yn)), wd=wd_current, ws=ws_current, TI=TI, time=Time).aep() * 1e3
    else:
        aep = wake_model(x=x, y=y, wd=wd_current, ws=ws_current, TI=TI, time=Time).aep() * 1e3
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
        metrics_recorder["cable_u_" + Sequence[0]].append(cab_data1['u'].tolist())
        metrics_recorder["cable_v_" + Sequence[0]].append(cab_data1['v'].tolist())
        metrics_recorder["cable_u_" + Sequence[1]].append(cab_data2['u'].tolist())
        metrics_recorder["cable_v_" + Sequence[1]].append(cab_data2['v'].tolist())
        metrics_recorder["cable_u_" + Sequence[2]].append(cab_data3['u'].tolist())
        metrics_recorder["cable_v_" + Sequence[2]].append(cab_data3['v'].tolist())
        metrics_recorder["cable_type_" + Sequence[0]].append(cab_data1['cable'].tolist())
        metrics_recorder["cable_type_" + Sequence[1]].append(cab_data2['cable'].tolist())
        metrics_recorder["cable_type_" + Sequence[2]].append(cab_data3['cable'].tolist())
        #
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
        metrics_recorder["cable_u_" + curzone].append(cab_data['u'].tolist())
        metrics_recorder["cable_v_" + curzone].append(cab_data['v'].tolist())
        metrics_recorder["cable_type_" + curzone].append(cab_data['cable'].tolist())
        for _, zone in enumerate(nnb):
            metrics_recorder["cable_u_" + zone].append([])
            metrics_recorder["cable_v_" + zone].append([])
            metrics_recorder["cable_type_" + zone].append([])
        for _, zone in enumerate(nb):
            metrics_recorder["cable_u_" + zone].append(metrics_recorder['cable_u_' + zone][-1])
            metrics_recorder["cable_v_" + zone].append(metrics_recorder['cable_v_' + zone][-1])
            metrics_recorder["cable_type_" + zone].append(metrics_recorder['cable_type_' + zone][-1])
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
    # performance of individual zones
    if Mode == 'cooperative':
        # store the values of the individual zones
        metrics_recorder["x_north"].append(x[:tur_nr[0]].flatten().tolist())
        metrics_recorder["y_north"].append(y[:tur_nr[0]].flatten().tolist())
        metrics_recorder["x_mid"].append(x[tur_nr[0]:sum(tur_nr[0:2])].flatten().tolist())
        metrics_recorder["y_mid"].append(y[tur_nr[0]:sum(tur_nr[0:2])].flatten().tolist())
        metrics_recorder["x_south"].append(x[sum(tur_nr[0:2]):].flatten().tolist())
        metrics_recorder["y_south"].append(y[sum(tur_nr[0:2]):].flatten().tolist())
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
    elif Mode == 'competitive' or Mode == 'evaluate_sequ' or Mode == 'evaluate_multiter':
        # current zone
        metrics_recorder["cur_zone"].append([curzone])
        metrics_recorder["neighbours"].append(nb)
        metrics_recorder["x_" + curzone].append(x.flatten().tolist())
        metrics_recorder["y_" + curzone].append(y.flatten().tolist())
        metrics_recorder["aep_" + curzone].append(aep.isel(wt=slice(0,tur_nr[wf[curzone]])).sum().item())
        metrics_recorder["cable_cost_" + curzone].append(cable_cost)
        metrics_recorder["mp_cost_" + curzone].append(sum(mp_cost))
        metrics_recorder["lcoe_" + curzone].append(lcoe)
        npv_all = [npv]
        
        # 0 for non-neighbours
        for n, zone in enumerate(nnb):
            metrics_recorder["x_" + zone].append([])
            metrics_recorder["y_" + zone].append([])
            metrics_recorder["aep_" + zone].append(0)
            metrics_recorder["cable_cost_" + zone].append(0)
            metrics_recorder["mp_cost_" + zone].append(0)
            metrics_recorder["lcoe_" + zone].append(0)
            
        # neighbours
        for n, zone in enumerate(nb):
            metrics_recorder["x_" + zone].append(metrics_recorder["x_" + zone][-1])
            metrics_recorder["y_" + zone].append(metrics_recorder["y_" + zone][-1])
            if n == 0:
                aep_n = aep.isel(wt=slice(tur_nr[wf[curzone]],tur_nr[wf[curzone]] + tur_nr[wf[zone]])).sum().item()
            elif n == 1:
                aep_n = aep.isel(wt=slice(tur_nr[wf[curzone]] + tur_nr[wf[nb[n-1]]],None)).sum().item()
            metrics_recorder["aep_" + zone].append(aep_n)
            metrics_recorder["cable_cost_" + zone].append(cable_cost_n[n])
            metrics_recorder["mp_cost_" + zone].append(mp_cost_n[n])
            npv_n = (capex*tur_nr[wf[zone]] + mp_cost_n[n] + cable_cost_n[n] + LP*tur_nr[wf[zone]]) * CRF + OpexAnnual*tur_nr[wf[zone]]
            metrics_recorder["lcoe_" + zone].append(npv_n / aep_n)
            npv_all.append(npv_n)
            
        # total
        metrics_recorder["aep_all"].append(np.sum(aep).item())
        metrics_recorder["cable_cost_all"].append(sum(cable_cost_n) + cable_cost)
        metrics_recorder["mp_cost_all"].append(sum(mp_cost_n) + sum(mp_cost))
        metrics_recorder["lcoe_all"].append(sum(npv_all)/np.sum(aep).item())
    # for global variable
    aep = aep.isel(wt=slice(0,len(x))).sum().item()
    if obj == 'lcoe':
        return lcoe # $/MWh
    elif obj == 'aep':
        return aep

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
    global aep, cable_cost, dcable_cost, mp_cost, dmp_cost, wd_current, ws_current, Time
    # 1.) aep
    # wd = np.arange(0, 360, 1)
    if nf:
        daep = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=np.concatenate((x,xn)), y=np.concatenate((y,yn)), ws=ws_current, TI=TI, wd=wd_current, time=Time)[:tur_nr[wf[curzone]],:tur_nr[wf[curzone]]] * 1e3
    else:
        daep = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=x, y=y, ws=ws_current, TI=TI, wd=wd_current, time=Time) * 1e3
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
    if obj == 'lcoe':
        return dlcoe # $/MWh
    elif obj == 'aep':
        return daep

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
    "sequence": Sequence,
    "iteration": [],
    "opt_nr": [],
    "cur_zone": [],
    "neighbours": [],
    "aep": [],
    "mp_cost": [],
    "cable_cost": [],
    "lcoe": [],
    "cable_u": [],
    "cable_v": [],
    "cable_u_north": [],
    "cable_u_mid": [],
    "cable_u_south": [],
    "cable_v_north": [],
    "cable_v_mid": [],
    "cable_v_south": [],
    "cable_type": [],
    "cable_type_north": [],
    "cable_type_mid": [],
    "cable_type_south": [],
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
    "x_north": [],
    "y_north": [],
    "x_mid": [],
    "y_mid": [],
    "x_south": [],
    "y_south": [],
    "x_final": [],
    "y_final": [],
    "lcoe_final": []
}

#%% Convergence plotting script
def plot_convergence(mr=None,item=None,plotstr=None,obj=1,overall=0,optfat=0):
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item+'_north']], label='North', linewidth = 1)
    plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item+'_mid']], label='Mid', linewidth = 1)
    plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item+'_south']], label='South', linewidth = 1)
    if obj:
        plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item]], label='Overall', linewidth = 1)
    if overall:
        plt.plot(np.arange(mr['iteration'][-1]),[x if x!=0 else np.NaN for x in mr[item+'_all']], label='Overall', linewidth = 1)
    if optfat:
        for i in range(len(list(set(Sequence)))-1,len(mr['lcoe_all'])-1):
            curzone
            plt.plot(np.array(mr['iteration'][i:i+2])-1,
                np.array(mr[item+'_'+mr['cur_zone'][i+1][0]][i:i+2]), linestyle="--", linewidth = 0.8, color="black")
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
    sample = True
    SepCabling = True       # Indicate if cabling is only allowed within each zone or cross-zonal
    Sx = Subs_x
    Sy = Subs_y
    
    # Optimization setup
    tf = TopFarmProblem(
            design_vars = {'x':x0, 'y':y0},         
            cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=sum(tur_nr), cost_function=lcoe_func, objective=True, cost_gradient_function=lcoe_jac, maximize=False),
            constraints = DistanceConstraintAggregation([SpacingConstraint(min_spacing_m), joint_boundaries], sum(tur_nr), min_spacing_m, windTurbines), 
            driver = EasySGDDriver(maxiter=3000, learning_rate=windTurbines.diameter()*0.2, speedupSGD=True, sgd_thresh=0.02),
            plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sub_x, Sy=Sub_y, cables=cables, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=[]),
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
    sample = True
    SepCabling = False
    boundplot = list(boundaries.values())
    plot_folder = "Figures_" + ''.join([entry[0] for entry in Sequence])
        
    # go through each zone as specified in Sequence
    for i in range(len(Sequence)):
        opt_nr = i+1
        curzone = Sequence[i]
        # Initial Layout
        if any(metrics_recorder['x_'+curzone]):
            x0 = np.array(metrics_recorder['x_'+curzone][-1])
            y0 = np.array(metrics_recorder['y_'+curzone][-1])
        else:
            x0 = np.array(points[curzone]).T[0]
            y0 = np.array(points[curzone]).T[1]
    
        # Constraint
        constraint_comp = XYBoundaryConstraint([InclusionZone(boundaries[Sequence[i]])], 'multi_polygon')
        
        # Options
        Sx = [Subs_x[wf[curzone]]]
        Sy = [Subs_y[wf[curzone]]]
        
        nb = Sequence[:i]
        nb = list(dict.fromkeys(nb[::-1]))[::-1]    # only keep latest ones in case of multiple entries
        nb = [x for x in nb if x not in curzone]    # kick out if current zone countained
        nnb = list(dict.fromkeys([x for x in Sequence if x not in nb and x not in curzone]))
        
        if nb:
            nf = True
            xn = np.array([])
            yn = np.array([])
            for j, zone in enumerate(nb):
                xn = np.concatenate([xn, metrics_recorder['x_' + zone][-1]])
                yn = np.concatenate([yn, metrics_recorder['y_' + zone][-1]])
                cable_cost_n[j] = metrics_recorder['cable_cost_' + zone][-1]
                mp_cost_n[j]= metrics_recorder['mp_cost_' + zone][-1]
        
        # Plot or not
        if plot_iter:
            plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sub_x, Sy=Sub_y, cables=cables, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot, opt_nr=opt_nr, folder=plot_folder, sampling=sample, obj=obj, ploteach=plot_each)
        else:
            plot_comp = None
        
        # Max or min
        if obj == 'lcoe':
            maximize = False
        elif obj == 'aep':
            maximize = True
        
        # change initial learning rate if iterative sequential optimization
        if i < len(list(set(Sequence))):
            learning_rate = windTurbines.diameter()*0.2
        elif i < 2*len(list(set(Sequence))):
            learning_rate = windTurbines.diameter()*0.01
        else:
            learning_rate = windTurbines.diameter()*0.005
        
        # Optimization Setup
        tf = TopFarmProblem(
                design_vars = {'x':x0, 'y':y0},         
                cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=tur_nr[wf[Sequence[i]]], cost_function=lcoe_func, objective=True, cost_gradient_function=lcoe_jac, maximize=maximize),
                constraints = DistanceConstraintAggregation([SpacingConstraint(min_spacing_m), constraint_comp],tur_nr[wf[Sequence[i]]], min_spacing_m, windTurbines), 
                driver = EasySGDDriver(maxiter=3000, learning_rate=learning_rate, speedupSGD=True, sgd_thresh=0.02),
                plot_comp = plot_comp)
        
        # Run
        tic = time.time()
        cost, state, recorder = tf.optimize()
        toc = time.time()
        print('Optimization with SGD took: {:.0f}s'.format(toc-tic), ' with a total constraint violation of ', recorder['sgd_constraint'][-1])
        
        # Store
        metrics_recorder["lcoe_final"].append([cost])
        metrics_recorder["x_final"].append(state['x'].tolist())
        metrics_recorder["y_final"].append(state['y'].tolist())
        
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
        
#%% Evaluate existing design
elif Mode == 'evaluate_sequ':
    # specify file you want to load
    with open("C:\Software\IEA-Wind-2200-22-ROWP\examples\metric_recorder_sequential_nms_lcoe.pkl", "rb") as file:
        data = pickle.load(file)
    data = data['metrics_recorder']
    # update these!
    curzone = 'mid'
    with open("C:\Software\IEA-Wind-2200-22-ROWP\examples\metric_recorder_sequential_m_lcoe.pkl", "rb") as file:
        data2 = pickle.load(file)
    data2 = data2['metrics_recorder']
    x0 = np.array(data2['x_' + curzone][-1])
    y0 = np.array(data2['y_' + curzone][-1])
    #
    # defaults
    nb = Sequence.copy()
    nb.remove(curzone)
    nnb = []
    # nb = ['north']
    # nnb = ['south']
    SepCabling = False
    sample = False
    if nb:
        nf = True
        xn = np.array([])
        yn = np.array([])
        for j, zone in enumerate(nb):
            xn = np.concatenate([xn, data['x_' + zone][-1]])
            yn = np.concatenate([yn, data['y_' + zone][-1]])
            cable_cost_n[j] = data['cable_cost_' + zone][-1]
            mp_cost_n [j]= data['mp_cost_' + zone][-1]
            
            metrics_recorder['x_' + zone] = [data['x_' + zone][-1]]
            metrics_recorder['y_' + zone] = [data['y_' + zone][-1]]
            metrics_recorder['cable_u_' + zone] = [data['cable_u_' + zone][-1]]
            metrics_recorder['cable_v_' + zone] = [data['cable_v_' + zone][-1]]
            metrics_recorder['cable_type_' + zone] = [data['cable_type_' + zone][-1]]
            
    boundplot = list(boundaries.values())
    plot_folder = "EvalFigures"
    opt_nr = None  
    Sx = [Subs_x[wf[curzone]]]
    Sy = [Subs_y[wf[curzone]]]
    #
    # run
    lcoe = lcoe_func(x0,y0)
    print(lcoe)
    # plot
    plt.figure()
    plot = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sub_x, Sy=Sub_y, cables=cables, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot, opt_nr=opt_nr, folder=plot_folder, sampling=sample, obj=obj)
    inputs = {}
    inputs['x'] = np.array(x0)
    inputs['y'] = np.array(y0)
    plot.compute(inputs,[])
#%%
elif Mode == 'evaluate_multiter':
    # specify file you want to load
    with open("C:\Software\IEA-Wind-2200-22-ROWP\examples\metric_recorder_sequential_nmsnmsnms_rand.pkl", "rb") as file:
        data = pickle.load(file)
    data = data['metrics_recorder']
    
    # get the index in metrics recorder of each last iteration step
    vec = pd.Series(data['opt_nr'])
    eva_indices = vec.groupby(vec).apply(lambda x: x.index[-1]).values
    
    
    for idx in eva_indices:
        curzone = data['cur_zone'][idx][0]
        x0 = np.array(data['x_' + curzone][idx])
        y0 = np.array(data['y_' + curzone][idx])
        #
        # defaults
        nb = data['neighbours'][idx]
        nnb = list(set(Sequence))
        nnb.remove(curzone)
        nnb = [x for x in nnb if x not in nb]
        # nb = ['north']
        # nnb = ['south']
        SepCabling = False
        sample = False
        if nb:
            nf = True
            xn = np.array([])
            yn = np.array([])
            for j, zone in enumerate(nb):
                xn = np.concatenate([xn, data['x_' + zone][idx]])
                yn = np.concatenate([yn, data['y_' + zone][idx]])
                cable_cost_n[j] = data['cable_cost_' + zone][idx]
                mp_cost_n [j]= data['mp_cost_' + zone][idx]
                
                # metrics_recorder['x_' + zone] = [data['x_' + zone][-1]]
                # metrics_recorder['y_' + zone] = [data['y_' + zone][-1]]
                # metrics_recorder['cable_u_' + zone] = [data['cable_u_' + zone][-1]]
                # metrics_recorder['cable_v_' + zone] = [data['cable_v_' + zone][-1]]
                # metrics_recorder['cable_type_' + zone] = [data['cable_type_' + zone][-1]]
                
        boundplot = list(boundaries.values())
        plot_folder = "EvalFigures"
        opt_nr = data['opt_nr'][idx]  
        Sx = [Subs_x[wf[curzone]]]
        Sy = [Subs_y[wf[curzone]]]
        #
        # run
        lcoe = lcoe_func(x0,y0)
        # plot
        plt.figure()
        plot = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sub_x, Sy=Sub_y, cables=cables, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot, opt_nr=opt_nr, folder=plot_folder, sampling=sample, obj=obj, optimize=False)
        inputs = {}
        inputs['x'] = np.array(x0)
        inputs['y'] = np.array(y0)
        plot.compute(inputs,[])
        
    plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=0,optfat=1)
    plot_convergence(mr=metrics_recorder,item='aep',plotstr='AEP (GWh)',obj=0,overall=0)