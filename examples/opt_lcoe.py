import numpy as np
import pandas as pd
from py_wake.rotor_avg_models import RotorCenter
import time
import matplotlib.pyplot as plt
import xarray as xr
import os
import utm
import pickle
from datetime import datetime
from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake import NOJ, BastankhahGaussian, Nygaard_2022
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasySGDDriver
from OptPlotBathy import XYPlotCompBathym
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint, InclusionZone
from py_wake.utils.gradients import fd, autograd
from topfarm.constraint_components.constraint_aggregation import DistanceConstraintAggregation
from py_wake.turbulence_models import CrespoHernandez
import windIO
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
# from ssms.CalculateMass import CalculateMass
from optiwindnet.api import WindFarmNetwork, ModelOptions, MILP, MetaHeuristic
from shapely.geometry import Point, Polygon
from topfarm.constraint_components.boundary import MultiWFBoundaryConstraint, BoundaryType
from RecordFunc import create_recorder, record_cable_metrics, record_main_metrics_multisub, record_main_metrics_singlesub, record_results_constraints
seed = 2
np.random.seed(seed)
#
#%% INPUTS
# General inputs
Mode = 'competitive'                    # 'cooperative' or 'competitive' or 'evaluate_recorder' or 'evaluate_multiter' or 'CompareCabling'
File = 'res_cop_8D_S2_processed'
Sequence = ['north','mid','south']*5    # define sequence of zones for sequential design
CableSolver = 'MetaHeuristic'           # 'Heuristic', 'MetaHeuristic', 'MILP_cplex' or 'MILP_ortools'
Continue = False                        # set to True if you give foregoing metrics_recorder to continue optimization
Model = 'turbopark'                     # 'jensen', 'gauss' or 'turbopark'
tur_nr = [33,33,34]                     # Desired turbine number in optimized farm, from north to south!
obj = 'lcoe'                            # 'lcoe' or 'aep'
plot_iter = False                       # True or False: plot and store layouts during optimization each plot_each iterations
plot_postpro = True                     # True or False: plot and store layouts during postprocessing (how often is linked to step)
plot_each = 1                           # define in which interval a plot should be made
d_RD = 8                                # min spacing distance in rotor diameters
step = 20                               # at each "step" iterations, the full wind rose is recalculated in postprocessing (when sampling is used during opt)

# SGD
maxiter = 5000
sgd_thresh = 0.02

# Monopile optimization
MP_ref = 1                              # reference turbine type for monopile mass scaling. 0 = 10MW, 1 = 15MW, 2 = 3.4MW

# lcoe parameters
d = 0.05                                # [-] discount rate
life = 25                               # years, lifetime
RP = 22                                 # MW
D = 284                                 # m
HH = 170                                # m
HTrans = 15                             # m
WaveHeight = 2.52                       # m
WavePeriod = 5.45                       # s
WindSpeed = 9.924                       # m/s ToDo: verfiy it is average wind speed
capex = 9.6712e7 * 0.924                # €2024 per turbine, excl. Monopile and cabling, from NREL COE Report 2024, converted to € with 2024 average exchange rate
OpexAnnual = 2.97e6 * 0.924             # €2024 per turbine, annual OPEX,from NREL COE Report 2024, converted to € with 2024 average exchange rate
LP = 0                                  # $2010 per turbine, liquidation proceeds, from DETECT for HKN scaled (22MW turbines)

# Cable data [cross section, capacity, price]
# Define cable properties using a list of dictionaries
cable_specs = [
    {"diameter_mm2": 185, "capacity_NrT": 3, "cost_€_m": 368.9},
    {"diameter_mm2": 400, "capacity_NrT": 5, "cost_€_m": 428.9},
    {"diameter_mm2": 1000, "capacity_NrT": 7, "cost_€_m": 737.1}
]

#%% Load data and setup pywake
# system_dat = sys.argv[1]
system_dat = windIO.load_yaml(Path(os.sep.join(['..', 'inputs', 'wind_energy_system.yaml'])))
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
elif Model == 'turbopark':
    wake_model = Nygaard_2022(site, windTurbines)
    
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

#%% Setup monopile calculus and optimizer
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
interpolator = RegularGridInterpolator((northing_values, easting_values), -Z, method='linear')
def depth_interp(x, y):
    return interpolator(np.array([y, x]).T)

# Calculate monopile mass for different water depth
depths = np.linspace(np.min(-Z),np.max(-Z),20)
# ph = 15     # Platform height [m]
# swh = 2.52  # Significant Wave Height [m]
# swp = 5.45  # Significiant Wave Period [s]
# # P_interpolator = interp1d(np.cumsum(sum(P)), ws, kind='linear')  # interpolator to get mean wind sped (@50% probability)
# # V_ave = P_interpolator(0.5).tolist()
# import scipy.special as sp
# def mean_wind_speed(A, k):
#     return A * sp.gamma(1 + 1/k)
# V_ave = []
# for i in range(len(wd)):
#     V_ave.append(mean_wind_speed(A['data'][i],k['data'][i]))
# V_ave = np.sum(np.array(V_ave) * np.array(freq['data']))
# masses = []
# for z in depths:
#    cur_mass = CalculateMass(RP=rp/1e6, D=rd, HTrans=ph, HHub_Ratio=hh/rd, WaterDepth=z, WaveHeight=swh, WavePeriod=swp, WindSpeed=V_ave, IP_item=MP_ref)
#    masses.append(cur_mass[0][0])
# # add transition piece (100t, from 22MW report)
# masses = [x + 100000 for x in masses]
masses = [985788.0033692981, 1006729.722727049, 1027991.492707832, 1049573.3133116479, 1071475.1845384957, 1093697.1063883766, 1116239.0788612892, 1139101.1019572348, 1162283.1756762124, 1185785.300018223, 1209607.4749832656, 1233749.7005713405, 1258211.9767824481, 1282994.3036165882, 1308096.6810737606, 1333519.1091539655, 1359261.587857203, 1385324.1171834725, 1411706.6971327746, 1438409.3277051093]

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

# Cable optimization
CableSolvers = ['Heuristic', 'MetaHeuristic', 'MILP_ortools', 'MILP_cplex']         # Available solvers
CableSolvers_order = [CableSolver] + [s for s in CableSolvers if s != CableSolver]  # Try primary solver first
Routers = {'Heuristic': None,
           'MetaHeuristic': MetaHeuristic(time_limit=0.3),
           'MILP_cplex': MILP(solver_name='cplex', time_limit=5, mip_gap=0.005, verbose=False),
           'MILP_ortools': MILP(solver_name='ortools', time_limit=5, mip_gap=0.005, verbose=False)}
cables = np.array([[c["capacity_NrT"], c["cost_€_m"]] for c in cable_specs])
cables_plot = np.array([[c["diameter_mm2"], c["capacity_NrT"], c["cost_€_m"]] for c in cable_specs])

# defaults
# neighbour wind farm with turbine coordinates and costs to consider
nf = False
xn = []
yn = []
x_if = [] # infeasible cabling layouts
y_if = []
cable_cost_n = [0,0]
mp_cost_n = [0,0]
SepCabling = False
opt_nr = 1
if Mode == 'cooperative':
    Sequence = list(dict.fromkeys(Sequence))

#%% Function to create the random sampling of wind speed and wind directions
def sampling():
    ind = np.random.choice(np.arange(dirs.size), samps, p=freqs)
    wd = dirs[ind]
    A = As[ind]
    k = ks[ind]
    ws = A * np.random.weibull(k)
    return wd, ws

#%% Cabling optimization
def opt_cabling(x=None,y=None,Sx=None,Sy=None,cables=None):
    for i, solver in enumerate(CableSolvers_order):
        if i > 0: print(f"Trying with solver: {solver}")
        router = Routers.get(solver)
        try:
            wfn = WindFarmNetwork(turbinesC=np.column_stack((x, y)),substationsC=np.column_stack((Sx, Sy)),cables=cables,router=router)
            wfn.optimize()
            break
        except Exception as e:
            print(f"Solver {solver} failed: {e}")
            x_if.append(x)
            y_if.append(y)
    else:
        print("All solvers failed.")
    return wfn

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
    dmasses = (np.array(dmasses) * get_depth_grads(x, y)[:, 0, :].T)    # gradients (for function later, to avoid double calculus)
    dmp_cost = 3 * dmasses          # from ORBIT 2025 tp_steel_cost default
    #
    # 3.) Cable costs
    if SepCabling:
        indices = [np.arange(sum(tur_nr[:idx]), sum(tur_nr[:idx + 1])) for idx in range(len(wf))]
        cable_costs = []
        dcable_cost = np.empty((0, 2))
        for z, zone in enumerate(Sequence):
            wfn = opt_cabling(x=x[indices[z]],y=y[indices[z]],Sx=Sx[z],Sy=Sy[z],cables=cables)
            cable_costs.append(wfn.cost())
            record_cable_metrics(metrics_recorder, wfn, zone, nnb, nb)
            cur_dcable_cost, _ = wfn.gradient(gradient_type='cost')
            dcable_cost = np.vstack((dcable_cost,cur_dcable_cost))
        cable_cost = sum(cable_costs)
    else:
        wfn = opt_cabling(x=x,y=y,Sx=Sx,Sy=Sy,cables=cables)
        cable_cost = wfn.cost()     # Costs in Euro
        record_cable_metrics(metrics_recorder, wfn, curzone, nnb, nb)   # recorder
        dcable_cost, _ = wfn.gradient(gradient_type='cost')             # gradients (for function later, to avoid double calculus)
    # !! ToDo: Scale costs to same currency and year of reference
    #
    # 4.) lcoe
    CRF = d / (1 - (1 + d) ** -life)
    npv = (capex*len(x) + np.sum(mp_cost) + cable_cost + LP*len(x)) * CRF + OpexAnnual*len(x)
    # lcoe = npv / np.sum(aep).item()
    lcoe = npv / aep.isel(wt=slice(0,len(x))).sum().item()
    #
    # 5. Record the missing metrics
    if Mode == 'cooperative' or curzone == 'all':
        record_main_metrics_multisub(metrics_recorder, opt_nr, aep, x, y, mp_cost, cable_costs, lcoe,
            curzone, nb, nnb, wf, tur_nr, cable_cost_n, mp_cost_n,
            npv, capex, LP, CRF, OpexAnnual, Sequence)
    elif Mode == 'competitive' or Mode == 'evaluate_multiter':
        record_main_metrics_singlesub(metrics_recorder, opt_nr, aep, x, y, mp_cost, cable_cost, lcoe,
            curzone, nb, nnb, wf, tur_nr, cable_cost_n, mp_cost_n,
            npv, capex, LP, CRF, OpexAnnual)
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
min_spacing_m = d_RD * windTurbines.diameter()  #minimum inter-turbine spacing in meters

#%% Recorder
metrics_recorder = create_recorder(Sequence)

#%% Convergence plotting script
def plot_convergence(mr=None,item=None,plotstr=None,obj=1,overall=0,optfat=0,feas=0):
    FS = 9
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_north']], label='North', linewidth = 1)
    ax.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_mid']], label='Mid', linewidth = 1)
    ax.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_south']], label='South', linewidth = 1)
    if obj:
        ax.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item]], label='Overall', linewidth = 1)
    if overall:
        ax.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_all']], label='Overall', linewidth = 1)
    if optfat:
        for i in range(0,len(mr['lcoe_all'])-1):
            if i == 0:
                leg = 'Current objective'
            else:
                leg = None
            if mr['opt_nr'][i] == mr['opt_nr'][i+1]:
                ax.plot(np.array(mr['iteration'][i:i+2]),
                    np.array(mr[item+'_'+mr['cur_zone'][i][0]][i:i+2]), linestyle="--", linewidth = 0.8, color="black", label=leg)
    if feas:
        ax2 = ax.twinx()
        if mr['cur_zone'][0][0] == 'all':
            ax2.plot(np.array(mr['iteration']), [(-x / len(mr['x_final'][0])) if x is not None else np.nan for x in mr['tur_dist_violation']], 'r:', linewidth = 1, label='Spacing constraint')
            ax2.plot(np.array(mr['iteration']), [(-x / len(mr['x_final'][0])) if x is not None else np.nan for x in mr['bound_violation']], 'b:', linewidth = 1, label='Boundary constraint')
        else:
            ax2.plot(np.array(mr['iteration']), [(-x / len(mr['y_' + cz[0]][i])) if x is not None else np.nan for i, (x, cz) in enumerate(zip(mr['tur_dist_violation'], mr['cur_zone']))], 'r:', linewidth = 1, label='Spacing constraint')
            ax2.plot(np.array(mr['iteration']), [(-x / len(mr['y_' + cz[0]][i])) if x is not None else np.nan for i, (x, cz) in enumerate(zip(mr['bound_violation'], mr['cur_zone']))], 'b:', linewidth = 1, label='Boundary constraint')
        ax2.set_ylabel('Constraint violation [m/turbine]')
        ax2.tick_params(axis='x', labelsize=FS - 1)
        ax2.tick_params(axis='y', labelsize=FS - 1)
        # ax2.set_yscale('log')
        # legend
        line1, label1 = ax.get_legend_handles_labels()
        line2, label2 = ax2.get_legend_handles_labels()
        ax.legend(line1 + line2, label1 + label2, fontsize=FS)
    else:
        ax.legend(fontsize=FS)
    # decorate
    ax.grid(True)
    ax.set_xlabel('Iteration', fontsize=FS)
    ax.set_ylabel(plotstr, fontsize=FS)
    ax.set_title(plotstr, fontsize=FS + 2)
    ax.tick_params(axis='x', labelsize=FS - 1)
    ax.tick_params(axis='y', labelsize=FS - 1)
    fig.tight_layout()
    ax.set_xlim([0,max(mr['iteration'])])
    
    # ax.set_ylim([87.7, 95])
    # ax2.set_ylim([-0.3,7])
    

#%% Postprocessing script for full wind rose
def postprocess_recorder(data):
    global metrics_recorder, nb, curzone, Sequence, SepCabling, Sx, Sy, nnb, sample, opt_nr, xn, yn, nf, cable_cost_n, mp_cost_n
    # new recorder
    metrics_recorder = create_recorder(Sequence)
    # get the index in metrics recorder of each last iteration step
    vec = pd.Series(data['opt_nr'])
    eva_indices = vec.groupby(vec).apply(lambda x: x.index[-1]).values
    eva_indices = np.column_stack((eva_indices, eva_indices + 1)).ravel()
    eva_indices = np.insert(eva_indices,0,0)
    eva_indices = np.delete(eva_indices,-1)
    
    # Add indices in between
    new_indices = []
    for i in range(len(eva_indices) - 1):
        start = eva_indices[i]
        end = eva_indices[i + 1]
        step_indices = list(range(start, end, step))
        new_indices.extend(step_indices)
    new_indices.append(eva_indices[-1])  # Add the final value
    eva_indices = new_indices
    opt_nr_check = 0 # helper to see when swapping sequence
    # go through each required index and add results to metrics_recorder.
    for i, idx in enumerate(eva_indices):
        print('evaluate case ' + str(i+1) + '/' + str(len(eva_indices)))
        nb = data['neighbours'][idx]
        curzone = data['cur_zone'][idx][0]
        xn = []
        yn = []
        cable_cost_n = [0,0]
        mp_cost_n = [0,0]
        if curzone == 'all':
            # cooperative
            x0 = np.array([])
            y0 = np.array([])
            Sequence = list(dict.fromkeys(metrics_recorder['sequence']))
            for zone in Sequence:
                x0 = np.concatenate([x0,data['x_' + zone][idx]])
                y0 = np.concatenate([y0,data['y_' + zone][idx]])
            SepCabling = True
            Sx = Subs_x
            Sy = Subs_y
            nnb = []
        else:
            # sequential
            x0 = np.array(data['x_' + curzone][idx])
            y0 = np.array(data['y_' + curzone][idx])
            SepCabling = False
            nnb = list(set(Sequence))
            nnb.remove(curzone)
            nnb = [x for x in nnb if x not in nb]
            Sx = [Subs_x[wf[curzone]]]
            Sy = [Subs_y[wf[curzone]]]
        sample = False
        if nb:
            nf = True
            xn = np.array([])
            yn = np.array([])
            for j, zone in enumerate(nb):
                xn = np.concatenate([xn, data['x_' + zone][idx]])
                yn = np.concatenate([yn, data['y_' + zone][idx]])
                cable_cost_n[j] = data['cable_cost_' + zone][idx]
                mp_cost_n[j]= data['mp_cost_' + zone][idx]
        else:
            nf = False

        boundplot = list(boundaries.values())
        plot_folder = "Figures_processed_" + File
        opt_nr = data['opt_nr'][idx]
        metrics_recorder['sgd_constraint_violation'].append(data['sgd_constraint_violation'][idx])
        metrics_recorder['tur_dist_violation'].append(data['tur_dist_violation'][idx])
        metrics_recorder['bound_violation'].append(data['bound_violation'][idx])
        if opt_nr_check != opt_nr:
            # update setting only for new zone
            metrics_recorder['settings'].append(data['settings'][opt_nr])
            opt_nr_check = opt_nr
        #
        # run
        lcoe_func(x0,y0)
        # plot
        if plot_postpro:
            plt.figure()
            plot = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sub_x, Sy=Sub_y, cables=cables_plot, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot, opt_nr=1, folder=plot_folder, sampling=sample, obj=obj, optimize=False, iter_nr = i)
            inputs = {}
            inputs['x'] = np.array(x0)
            inputs['y'] = np.array(y0)
            plot.compute(inputs,[])
            plt.close()
    metrics_recorder['iteration'] = eva_indices
    metrics_recorder['x_final'] = data['x_final']
    metrics_recorder['y_final'] = data['y_final']
    return metrics_recorder
 
#%% Cooperative design
if Mode == 'cooperative':
    Sequence = ['north','mid','south']
    plot_folder = "Figures_cooperative_2D_6000it"
    # Initital Layout
    x0 = np.concatenate([np.array(points[name])[:, 0] for name in wf])
    y0 = np.concatenate([np.array(points[name])[:, 1] for name in wf])
    
    # Constraint
    joint_boundaries = MultiWFBoundaryConstraint(
        geometry = [boundaries[name] for name in wf],  # Boundary mapping
        wt_groups=[np.arange(sum(tur_nr[:idx]), sum(tur_nr[:idx + 1])) for idx in range(len(wf))],  # Turbine groups
        boundtype = BoundaryType.POLYGON
    )

    # Options
    sample = True
    SepCabling = True       # Indicate if cabling is only allowed within each zone or cross-zonal
    boundplot = list(boundaries.values())
    plot_folder = "Figures_cooperative_2D_10000it"
    Sx = Subs_x
    Sy = Subs_y
    nb = []
    nnb = []
    curzone = 'all'
    learning_rate = windTurbines.diameter()*0.2
    
    # Plot or not
    if plot_iter:
        plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sub_x, Sy=Sub_y, cables=cables_plot, metrics_recorder=metrics_recorder, b=boundplot, folder=plot_folder, sampling=sample, obj=obj, ploteach=plot_each)
    else:
        plot_comp = None
    
    # Max or min
    if obj == 'lcoe':
        maximize = False
    elif obj == 'aep':
        maximize = True
    
    # record settings
    metrics_recorder['settings'].append({'Mode':Mode,'CableSolver':CableSolver,'Model':Model,'seed:':seed,'d_RD':d_RD,'tur_nr':tur_nr,'maxiter':maxiter,'learning_rate':learning_rate,'sgd_thresh':sgd_thresh,'curzone':curzone,'obj':obj,'Sequence':Sequence,'x0':x0,'y0':y0,'sample':sample,'samps':samps,'SepCabling':SepCabling,'Sx':Sx,'Sy':Sy,'depths':depths,'masses':masses,'Routers':Routers}) 
    [metrics_recorder[key].append(None) for key in ["sgd_constraint_violation", "tur_dist_violation", "bound_violation"]]   # first run: no optimization
    
    # Optimization setup
    tf = TopFarmProblem(
            design_vars = {'x':x0, 'y':y0},         
            cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=sum(tur_nr), cost_function=lcoe_func, objective=True, cost_gradient_function=lcoe_jac, maximize=maximize),
            constraints = DistanceConstraintAggregation(joint_boundaries, sum(tur_nr), min_spacing_m, windTurbines), 
            driver = EasySGDDriver(maxiter=maxiter, learning_rate=learning_rate, speedupSGD=True, sgd_thresh=sgd_thresh),
            plot_comp = plot_comp
            )
    
    # Run
    tic = time.time()
    cost, state, recorder = tf.optimize()
    toc = time.time()
    print('Optimization with SGD took: {:.0f}s'.format(toc-tic), ' with a total constraint violation of ', recorder['sgd_constraint'][-1])

    # Store
    record_results_constraints(metrics_recorder, recorder, state, cost, min_spacing_m)
    
    # Save to a file
    with open(File + ".pkl", "wb") as file:
        pickle.dump({"metrics_recorder": metrics_recorder, "state": state, "recorder": recorder.recorder2list()}, file)
        
    # Postprocess for full wind rose
    if sample:
        print('Optimization finished.')
        print('Starting postprocessing...')
        metrics_recorder = postprocess_recorder(metrics_recorder)
        # Save processed file
        with open(File + "_processed.pkl", "wb") as file:
            pickle.dump({"metrics_recorder": metrics_recorder}, file)
    
    # Plot LCOE iterations
    plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=1,optfat=0,feas=1)

#%% Competitive design
elif Mode == 'competitive':
    if Continue:
        print('---- !! Careful, starting from foregoing design !! ----')
        file = "metric_recorder_sequential_samelr_linint_nmsnmsnms"
        with open("C:\\Software\\IEA-Wind-2200-22-ROWP\\examples\\" + file + ".pkl", "rb") as file:
            data = pickle.load(file)
        metrics_recorder = data['metrics_recorder']
        # Truncate each list
        for key in metrics_recorder:
            metrics_recorder[key] = metrics_recorder[key][:12126 + 1]
        Sequence = metrics_recorder['sequence']
        loop_range = range(6,len(Sequence))
    else:
        loop_range = range(len(Sequence))
    # general options
    sample = True
    SepCabling = False
    boundplot = list(boundaries.values())
    plot_folder = "Figures_" + ''.join([entry[0] for entry in Sequence])
        
    # go through each zone as specified in Sequenc
    for i in loop_range:
        now = datetime.now()
        print('Evaluate zone ' + str(i+1) + '/' + str(len(Sequence)) + '. Time: ' + now.strftime("%H:%M:%S"))
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
            plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sub_x, Sy=Sub_y, cables=cables_plot, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot, opt_nr=opt_nr, folder=plot_folder, sampling=sample, obj=obj, ploteach=plot_each)
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
            learning_rate = windTurbines.diameter()*0.15
        elif i < 3*len(list(set(Sequence))):
            learning_rate = windTurbines.diameter()*0.1
        else:
            learning_rate = windTurbines.diameter()*0.05
        
        # Same for time parameter in case of MILP cabling optimization. Set down computational time for re-iterations
        if i > len(list(set(Sequence))):
            Routers = {'Heuristic': None,
                       'MetaHeuristic': MetaHeuristic(time_limit=0.3),
                       'MILP_cplex': MILP(solver_name='cplex', time_limit=3, mip_gap=0.005, verbose=False),
                       'MILP_ortools': MILP(solver_name='ortools', time_limit=3, mip_gap=0.005, verbose=False)}
        
        # record settings
        [metrics_recorder[key].append(None) for key in ["sgd_constraint_violation", "tur_dist_violation", "bound_violation"]]   # first run: no optimization
        metrics_recorder['settings'].append({'Mode':Mode,'CableSolver':CableSolver,'Model':Model,'seed:':seed,'d_RD':d_RD,'tur_nr':tur_nr,'maxiter':maxiter,'learning_rate':learning_rate,'sgd_thresh':sgd_thresh,'curzone':curzone,'obj':obj,'Sequence':Sequence,'x0':x0,'y0':y0,'sample':sample,'samps':samps,'SepCabling':SepCabling,'Sx':Sx,'Sy':Sy,'depths':depths,'masses':masses,'Routers':Routers}) 
        
        # Optimization Setup
        tf = TopFarmProblem(
                design_vars = {'x':x0, 'y':y0},         
                cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=tur_nr[wf[Sequence[i]]], cost_function=lcoe_func, objective=True, cost_gradient_function=lcoe_jac, maximize=maximize),
                constraints = DistanceConstraintAggregation(constraint_comp, tur_nr[wf[Sequence[i]]], min_spacing_m, windTurbines), 
                driver = EasySGDDriver(maxiter=maxiter, learning_rate=learning_rate, speedupSGD=True, sgd_thresh=sgd_thresh),
                plot_comp = plot_comp)
        
        # Run
        tic = time.time()
        cost, state, recorder = tf.optimize()
        toc = time.time()
        print('Optimization with SGD took: {:.0f}s'.format(toc-tic), ' with a total constraint violation of ', recorder['sgd_constraint'][-1])
        
        # Store
        record_results_constraints(metrics_recorder, recorder, state, cost, min_spacing_m)
        
        # Save recorder to file
        with open(File + ".pkl", "wb") as file:
            pickle.dump({"metrics_recorder": metrics_recorder, "state": state, "recorder": recorder.recorder2list()}, file)
   
    # Postprocess for full wind rose
    if sample:
        print('Optimization finished.')
        print('Starting postprocessing...')
        metrics_recorder = postprocess_recorder(metrics_recorder)
        # Save processed file
        with open(File + "_processed.pkl", "wb") as file:
            pickle.dump({"metrics_recorder": metrics_recorder}, file)
            
    # Plot LCOE iterations
    plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=1,optfat=1,feas=1)

#%% Manually start postprocessing
elif Mode == 'evaluate_multiter':
    with open("C:\\Software\\IEA-Wind-2200-22-ROWP\\examples\\" + File + ".pkl", "rb") as file:
        data = pickle.load(file)
    data = data['metrics_recorder']
    metrics_recorder = postprocess_recorder(data)
    # Save processed file
    with open(File + "_processed.pkl", "wb") as file:
        pickle.dump({"metrics_recorder": metrics_recorder}, file)
    plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=1,optfat=1)
    plot_convergence(mr=metrics_recorder,item='aep',plotstr='AEP (GWh)',obj=0,overall=0,optfat=1)
    plot_convergence(mr=metrics_recorder,item='cable_cost',plotstr='Cable Cost (€)',obj=0,overall=0,optfat=1)
    plot_convergence(mr=metrics_recorder,item='mp_cost',plotstr='Monopile Cost (€)',obj=0,overall=0,optfat=1)
    
#%% Load metric_recorder and plot
elif Mode == 'evaluate_recorder':
    with open("C:\\Software\\IEA-Wind-2200-22-ROWP\\examples\\" + File + ".pkl", "rb") as file:
        data = pickle.load(file)
    metrics_recorder = data['metrics_recorder']
    if plot_iter:
        plot_folder = "FinalResult"
        boundplot = list(boundaries.values())
        nb = metrics_recorder['neighbours'][-1]
        xn = []
        yn = []
        for zone in nb:
            xn = np.concatenate([xn, metrics_recorder['x_' + zone][-1]])
            yn = np.concatenate([yn, metrics_recorder['y_' + zone][-1]])
        plt.figure()
        
        import matplotlib.font_manager as font_manager
        font_dir = ["font\Serif"]
        for font in font_manager.findSystemFonts(font_dir):
            font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = "Serif"
            
        plot = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=True, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sub_x, Sy=Sub_y, cables=cables_plot, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot, opt_nr=opt_nr, folder=plot_folder, sampling=sample, obj=obj, optimize=False, paper=True)
        inputs = {}
        curzone = metrics_recorder['cur_zone'][-1][-1]
        if curzone == 'all':
            # cooperative
            x0 = np.array([])
            y0 = np.array([])
            Sequence = list(dict.fromkeys(metrics_recorder['sequence']))
            for zone in Sequence:
                x0 = np.concatenate([x0,metrics_recorder['x_' + zone][-1]])
                y0 = np.concatenate([y0,metrics_recorder['y_' + zone][-1]])
            inputs['x'] = x0
            inputs['y'] = y0
        else:
            inputs['x'] = np.array(np.array(metrics_recorder['x_' + curzone][-1]))
            inputs['y'] = np.array(np.array(metrics_recorder['y_' + curzone][-1]))
        plot.compute(inputs,[])
        
        plt.gcf().subplots_adjust(
            top=0.995,
            bottom=0.088,
            left=0.0,
            right=0.83,
            hspace=0.2,
            wspace=0.2
        )
        plt.gcf().savefig("Layout.pdf", dpi=500, pad_inches=0)
        # plt.text(555000, 5842000, 'N', color='white', fontsize=22, ha='center', va='center')
        # plt.text(548000, 5833000, 'M', color='white', fontsize=22, ha='center', va='center')
        # plt.text(543000, 5823000, 'S', color='white', fontsize=22, ha='center', va='center')
        # plt.gcf().savefig("Bathymetry.pdf", pad_inches=0, dpi=500)
        
    plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=0,optfat=1,feas=1)
    # plot_convergence(mr=metrics_recorder,item='aep',plotstr='AEP (GWh)',obj=0,overall=0,optfat=1)
    # plot_convergence(mr=metrics_recorder,item='cable_cost',plotstr='Cable Cost (€)',obj=0,overall=0,optfat=1)
    # plot_convergence(mr=metrics_recorder,item='mp_cost',plotstr='Monopile Cost (€)',obj=0,overall=0,optfat=1)
#%% Compare Cabling
elif Mode == 'CompareCabling':
    plt.figure(figsize=(5, 3))
    Files = ["metric_recorder_sequential_heuristic_nmsnmsnmsnmsnms_processed",
             "metric_recorder_sequential_metaheuristic_nmsnmsnmsnmsnms_processed",
             "metric_recorder_sequential_cplex_nmsnmsnmsnmsnms_processed",
             "metric_recorder_sequential_ortools_nmsnmsnmsnmsnms_processed"]
    CabOpt = ["heuristic","metaheuristic","cplex","ortools"]
    for f, File in enumerate(Files):
    # File = "metric_recorder_cooperative_2D_6000it_processed"
        # specify file you want to load
        with open("C:\\Software\\IEA-Wind-2200-22-ROWP\\examples\\" + File + ".pkl", "rb") as file:
            data = pickle.load(file)
        mr = data['metrics_recorder']
        item = 'lcoe'
        # item = 'cable_cost'
        plotstr = 'LCOE (€/MWh)'
        # plotstr = 'Cable Cost (€)'
        overall = 0
        optfat = 1
        obj = 0
        
        # plt.figure(figsize=(5, 3))
        # plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_north']], label='North_' + CabOpt[f], linewidth = 1)
        plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_mid']], label='Mid_' + CabOpt[f], linewidth = 1)
        # plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_south']], label='South_' + CabOpt[f], linewidth = 1)
        if obj:
            plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item]], label='Overall', linewidth = 1)
        if overall:
            plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_all']], label='Overall', linewidth = 1)
        if optfat:
            for i in range(0,len(mr['lcoe_all'])-1):
                if i == 0 and f == len(Files)-1:
                    leg = 'Current objective'
                else:
                    leg = None
                if mr['opt_nr'][i] == mr['opt_nr'][i+1] and mr['cur_zone'][i][0] == 'mid':
                    plt.plot(np.array(mr['iteration'][i:i+2])-1,
                        np.array(mr[item+'_'+mr['cur_zone'][i][0]][i:i+2]), linestyle="--", linewidth = 0.8, color="black", label=leg)
    FS = 9
    plt.legend(fontsize=FS)
    plt.grid()
    plt.xlabel('Iteration',fontsize=FS)
    plt.ylabel(plotstr,fontsize=FS)
    plt.title(plotstr,fontsize=FS+2)
    plt.xticks(fontsize=FS-1)
    plt.yticks(fontsize=FS-1)