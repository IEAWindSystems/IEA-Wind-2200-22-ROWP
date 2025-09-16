#%% The IEA Wind 2200-22-MW Reference Offshore Wind Plants 
#
# ***** Routine to optimize (or evaluate) wind farm clusters *****
# - for different objectives and with different design approaches
# - Based on TopFarm optimization platform, using the gradient-based SGD solver
# - Combination of AEP, monopile and cable costs assessment to determine location-dependent LCOE of a wind farm (cluster) considering neighbouring wind farm impacts
# 
# Developed through an international collaborative effort in IEA Wind Task 55 "REFWIND"
# 
# Developers:
# 1. Samuel Kainz (Technical University of Munich, TUM). Contact: samuel.kainz@tum.de
# 2. Julian Quick (Denmark Technical University, DTU)
# 3. Amir Arasteh (DTU)
# 4. Mauricio Souza de Alencar (DTU)
#
# Advisors:
# 5. Carlo L. Bottasso (TUM)
# 6. Rafael Valotta Rodrigues (University of Massachusetts Boston)
# 7. Pierre-Elouan Réthoré (DTU)
# 8. Abhinav Kapila (RWE)
# 9. Bruno Nguyen (RWE)
# 10. Pietro Bortolotti (NREL)
# 11. P.J. Stanley (Shell)
# 12. Sebastian Sanchez Perez Moreno (Shell)
#
# Project lead and advisor:
# 13. Christopher J. Bay (NREL)
#
#%% Preamble
# ---
# General
import numpy as np
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import utm
import pickle
from datetime import datetime
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from scipy.interpolate import RegularGridInterpolator
import os
os.environ["OPENMDAO_WORKDIR"] = os.path.join(os.path.dirname(__file__), ".openmdao_out")
#
# ---
# Topfarm and Pywake
from py_wake.utils.gradients import fd, autograd
from py_wake.turbulence_models import CrespoHernandez
from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake import NOJ, Nygaard_2022
from py_wake.literature import Bastankhah_PorteAgel_2014
from py_wake.rotor_avg_models import RotorCenter
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasySGDDriver
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint, InclusionZone
from topfarm.constraint_components.constraint_aggregation import DistanceConstraintAggregation
from topfarm.constraint_components.boundary import MultiWFBoundaryConstraint, BoundaryType
#
# ---
# Specific
import windIO
from optiwindnet.api import WindFarmNetwork, MILPRouter, HGSRouter, EWRouter
from optiwindnet.augmentation import poisson_disc_filler
from ssms.CalculateMass import CalculateMass
from ssms.curve_fit_monopile import trainQLS
from OptPlotBathy import XYPlotCompBathym
from RecordFunc import create_recorder, record_cable_metrics, record_main_metrics_multisub, record_main_metrics_singlesub, record_results_constraints
#
#%% Main script to run
# def run_script(seed=2):
#%% INPUTS
# General inputs
Mode = 'competitive'                    # 'cooperative' or 'competitive' or 'evaluate_recorder' or 'evaluate_multiter' or 'CompareCabling' or 'evaluate_layout'
seed = 2                                # random np seed for initial layout configuration
Continue = False                        # set to True if you give foregoing metrics_recorder to continue optimization
File = 'test_s' + str(seed)             # define name of files that is stored or loaded
Sequence = ['north','mid','south']*4    # define sequence of zones for sequential design
CableSolver = 'MetaHeuristic'           # 'Heuristic', 'MetaHeuristic', 'MILP_cplex', 'MILP_ortools' or 'MILP_gurobi'
Model = 'turbopark'                     # 'jensen', 'gauss' or 'turbopark'
tur_nr = [33,33,34]                     # Desired turbine number in optimized farm, from north to south!
obj = 'lcoe'                            # 'lcoe' or 'aep'
plot_iter = True                        # True or False: plot and store layouts during optimization each plot_each iterations
plot_postpro = True                     # True or False: plot and store layouts during postprocessing (how often is linked to step)
plot_each = 100                         # define in which interval a plot should be made
d_RD = 6                                # min spacing distance in rotor diameters
step = 10000                            # at each "step" iterations, the full wind rose is recalculated in postprocessing (when sampling is used during opt)

# plot lims
xlim = None                             # specify xlim for convergence plot or put None
ylim = None                             # specify ylim for axis1 (obj) for convergence plot or put None
ylim2 = None                            # specify ylim for axis2 (penalty) for convergence plot or put None
ax2_ystep = None                        # specify tick step for axis2 or put None

# SGD
maxiter = 3000                          # maximum nr of iterations for SGD opt
sgd_thresh = 0.02                       # SGD threshold

# Monopile optimization
MP_ref = 1                              # reference turbine type for monopile mass scaling. 0 = 10MW, 1 = 15MW, 2 = 3.4MW

# lcoe parameters
d = 0.0661                              # [-] discount rate, from NREL CoE 2024 report
life = 25                               # years, lifetime
capex = 9.6734e7 * 0.924                # €2024 per turbine, excl. Monopile and cabling, from NREL COE Report 2024, converted to € with 2024 average exchange rate
OpexAnnual = 2.97e6 * 0.924             # €2024 per turbine, annual OPEX,from NREL COE Report 2024, converted to € with 2024 average exchange rate
LP = 0                                  # $2010 per turbine, liquidation proceeds, from DETECT for HKN scaled (22MW turbines)

# mass surrogate
PlatformHeight = 15                     # Platform height [m]
WaveHeight = 1.4                        # Significant Wave Height [m], taken from HKW metocean report p. 183
WavePeriod = 6.75                       # Associated Wave Period [s]

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
# int_speeds = np.linspace(np.min(np.min([p_ws, ct_ws])), np.max(np.max([p_ws, ct_ws])), 10000)
# ps_int = np.interp(int_speeds, p_ws, p)
# cts_int = np.interp(int_speeds, ct_ws, ct)
windTurbines = WindTurbine(name=farm_dat['turbines']['name'], diameter=rd, hub_height=hh, 
                      powerCtFunction=PowerCtTabular(p_ws, p, power_unit='W', ct=ct))

# wake model
if Model == 'jensen':
    wake_model = NOJ(site, windTurbines, k=0.05, rotorAvgModel=RotorCenter())
elif Model == 'gauss':
    wake_model = Bastankhah_PorteAgel_2014(site, windTurbines, rotorAvgModel=RotorCenter(), turbulenceModel=CrespoHernandez(), k=0.0324555)
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
# Note: a mass surrogate model only with 20MW results is built. Approximation of the 22MW machine.
trainQLS()
depths = np.linspace(np.min(-Z),np.max(-Z),20)
import scipy.special as sp
def mean_wind_speed(A, k):
    return A * sp.gamma(1 + 1/k)
V_ave = []
for i in range(len(wd)):
    V_ave.append(mean_wind_speed(A['data'][i],k['data'][i]))
V_ave = np.sum(np.array(V_ave) * np.array(freq['data']))
masses = []
for z in depths:
   cur_mass = CalculateMass(D=rd, HTrans=PlatformHeight, HHub=hh, WaterDepth=z, WaveHeight=WaveHeight, WavePeriod=WavePeriod, WindSpeed=V_ave, IP_item=1)
   masses.append(cur_mass[0][0])
# add transition piece (100t, from 22MW report)
masses = [x + 100000 for x in masses]
# masses = [985788.0033692981, 1006729.722727049, 1027991.492707832, 1049573.3133116479, 1071475.1845384957, 1093697.1063883766, 1116239.0788612892, 1139101.1019572348, 1162283.1756762124, 1185785.300018223, 1209607.4749832656, 1233749.7005713405, 1258211.9767824481, 1282994.3036165882, 1308096.6810737606, 1333519.1091539655, 1359261.587857203, 1385324.1171834725, 1411706.6971327746, 1438409.3277051093]

# Fit a polynomial of degree 2
depthmass = np.column_stack((masses,depths))
coefficients = np.polyfit(depthmass[:, 1], depthmass[:, 0], 2)
polynomial = np.poly1d(coefficients)
polynomial_gradients = np.polyder(polynomial)

# SGD sampling
sample = False
samps = 100    #number of samples 
site.interp_method = 'linear'

# Cable optimization
CableSolvers = ['MetaHeuristic', 'Heuristic', 'MILP_cplex', 'MILP_gurobi', 'MILP_ortools']         # Available solvers in default order
CableSolvers_order = [CableSolver] + [s for s in CableSolvers if s != CableSolver]  # Try primary solver first
tl_metaheuristic = 0.3  # time limit
tl_milp = 3
mip_gap = 0.005
Routers = {'Heuristic': EWRouter(),
           'MetaHeuristic': HGSRouter(time_limit=tl_metaheuristic),
           'MILP_cplex': MILPRouter(solver_name='cplex', time_limit=tl_milp, mip_gap=mip_gap, verbose=False),
           'MILP_ortools': MILPRouter(solver_name='ortools', time_limit=tl_milp, mip_gap=mip_gap, verbose=False),
           'MILP_gurobi': MILPRouter(solver_name='gurobi', time_limit=tl_milp, mip_gap=mip_gap, verbose=False)}
cables = np.array([[c["capacity_NrT"], c["cost_€_m"]] for c in cable_specs])
cables_plot = np.array([[c["diameter_mm2"], c["capacity_NrT"], c["cost_€_m"]] for c in cable_specs])

# results and figure folders
os.makedirs("Results", exist_ok=True)
os.makedirs("Figures", exist_ok=True)
# create empty figure if plot_iter is on and optimization should run
if plot_iter and (Mode == 'cooperative' or Mode == 'competitive'):
    plt.figure()
    
# defaults
# neighbour wind farm with turbine coordinates and costs to consider
nf = False
xn = []
yn = []
cable_cost_n = [0,0]
mp_cost_n = [0,0]
SepCabling = False
opt_nr = 1
if Mode == 'cooperative':
    Sequence = list(dict.fromkeys(Sequence))
    
# update font to latex
import matplotlib.font_manager as font_manager
font_dir = ["font\Serif"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
plt.rcParams["font.family"] = "Serif"
plt.rcParams['svg.fonttype'] = 'path'
#
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
    if CableSolver in ['MILP_cplex', 'MILP_ortools', 'MILP_gurobi']:
        # warm start
        wfn = WindFarmNetwork(turbinesC=np.column_stack((x, y)),substationsC=np.column_stack((Sx, Sy)),cables=cables,router=Routers.get('MetaHeuristic'))
        wfn.optimize()
        # chosen router
        wfn.optimize(router=Routers.get(CableSolver))
    else:
        wfn = WindFarmNetwork(turbinesC=np.column_stack((x, y)),substationsC=np.column_stack((Sx, Sy)),cables=cables,router=Routers.get(CableSolver))
        wfn.optimize()
    return wfn

#%% Objective function
def lcoe_func(x, y, **kwargs):
    # 0.) Extract relevant parameters from kwargs
    metrics_recorder = kwargs["metrics_recorder"]
    cable_cost_n = kwargs["cable_cost_n"]
    mp_cost_n = kwargs["mp_cost_n"]
    #
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
    #
    # 5.) Store results into kwargs that need to be handed over to derivative function  
    kwargs["aep"]["value"] = aep.isel(wt=slice(0,len(x))).sum().item()
    kwargs["cable_cost"]["value"] = cable_cost
    kwargs["dcable_cost"]["value"] = dcable_cost
    kwargs["mp_cost"]["value"] = mp_cost
    kwargs["dmp_cost"]["value"] = dmp_cost
    kwargs["wd_current"]["value"] = wd_current
    kwargs["ws_current"]["value"] = ws_current
    kwargs["Time"]["value"] = Time

    if obj == 'lcoe':
        return lcoe # $/MWh
    elif obj == 'aep':
        return aep.isel(wt=slice(0,len(x))).sum().item()

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
    # 0.) Extract relevant parameters from kwargs
    aep = kwargs["aep"]["value"]
    cable_cost = kwargs["cable_cost"]["value"]
    dcable_cost = kwargs["dcable_cost"]["value"]
    mp_cost = kwargs["mp_cost"]["value"]
    dmp_cost = kwargs["dmp_cost"]["value"]
    wd_current = kwargs["wd_current"]["value"]
    ws_current = kwargs["ws_current"]["value"]
    Time = kwargs["Time"]["value"]
    #
    # 1.) aep
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
# metrics_recorder = create_recorder(Sequence)
# nb = []
# nnb = []
# curzone = 'north'
# x0 = [550507.3,551004.3,552353.3,551501.3,552850.3,550010.3,552353.3,553915.3,553986.3,553986.3,559453.3,553986.3,551501.3,554057.3,555619.3,555619.3,555619.3,555619.3,556684.3,555690.3,555619.3,555619.3,555619.3,555690.3,557181.3,557749.3,558033.3,558104.3,559311.3,557465.3,557394.3,558033.3,558885.3]
# y0 = [5841236.0,5842585.0,5836621.0,5837686.0,5841591.0,5839603.0,5844218.0,5846064.0,5836692.0,5839745.0,5849898.0,5844076.0,5839319.0,5834491.0,5833639.0,5835485.0,5836834.0,5839035.0,5834704.0,5841804.0,5843934.0,5846135.0,5848123.0,5832716.0,5849969.0,5840242.0,5842088.0,5843934.0,5848265.0,5848123.0,5838325.0,5850963.0,5846135.0]
# Sx = [Subs_x[wf[curzone]]]
# Sy = [Subs_y[wf[curzone]]]
# lcoe = lcoe_func(x0, y0)
# lcoe_grad = lcoe_jac(x0, y0)
# def wrap_lcoe(s): return lcoe_func(*np.split(s, 2))
# grad = fd(wrap_lcoe, 0.000001)(np.append(x0, y0))
# print('difference between fd and analytic grads: ')
# print(grad - np.array(lcoe_grad).flatten())

#%% General constraints
# Min spacing
min_spacing_m = d_RD * windTurbines.diameter()  #minimum inter-turbine spacing in meters

# Generate initial layouts for each wind farm
coords = {name: poisson_disc_filler(tur_nr[wf[name]], min_dist=0.8*min_spacing_m, BorderC=boundaries[name], seed=seed)
          for name in wf}

#%% Recorder + kwargs
metrics_recorder = create_recorder(Sequence)

# Prepare (mostly mutable) kwargs that need to be passed throughout optimization and from lcoe to lcoe gradient function
extra_vars = dict(metrics_recorder=metrics_recorder, aep={"value": None}, cable_cost={"value": None}, dcable_cost={"value": None},
    mp_cost={"value": None}, dmp_cost={"value": None}, cable_cost_n=cable_cost_n, mp_cost_n=mp_cost_n, wd_current={"value": None},
    ws_current={"value": None}, Time={"value": None})

#%% Convergence plotting script
def plot_convergence(mr=None,item=None,plotstr=None,obj=1,overall=0,optfat=0,feas=0,reg=1):
    FS = 9
    LW = 1.3
    LW2 = 0.5
    fig, ax = plt.subplots(figsize=(6.5, 4))
    zones = list(dict.fromkeys(Sequence))
    for zone in zones:
        ax.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_'+str(zone)]], label=str(zone), linewidth = LW, zorder=2)
    if obj:
        ax.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item]], label='Overall', linewidth = LW)
    if overall:
        ax.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_all']], label='Overall', linewidth = LW, zorder=1)
    if optfat:
        for i in range(0,len(mr['lcoe_all'])-1):
            if mr['opt_nr'][i] == mr['opt_nr'][i+1]:
                ax.plot(np.array(mr['iteration'][i:i+2]),
                    np.array(mr[item+'_'+mr['cur_zone'][i][0]][i:i+2]), linestyle=":", linewidth = LW-0.2, color="black", label='Current objective' if i == 0 else "")
    if reg:
        vertical_lines = [mr['iteration'][i] for i in range(1, len(mr['opt_nr'])) if mr['opt_nr'][i] > mr['opt_nr'][i-1]]  # Find indices where opt_nr increases
        for x in vertical_lines:
            plt.axvline(x=x, linestyle=(0, (1, 3)), color='purple', label='New zone' if x == vertical_lines[0] else "", linewidth = 0.75)
    if feas:
        ax2 = ax.twinx()
        if mr['cur_zone'][0][0] == 'all':
            ax2.plot(np.array(mr['iteration']), [(-x / len(mr['x_final'][0])) if x is not None else np.nan for x in mr['tur_dist_violation']], 'r:', linewidth = LW2, label='Spacing constraint')
            ax2.plot(np.array(mr['iteration']), [(-x / len(mr['x_final'][0])) if x is not None else np.nan for x in mr['bound_violation']], 'b:', linewidth = LW2, label='Boundary constraint')
        else:
            ax2.plot(np.array(mr['iteration']), [(-x / len(mr['y_' + cz[0]][i])) if x is not None else np.nan for i, (x, cz) in enumerate(zip(mr['tur_dist_violation'], mr['cur_zone']))], 'r--', linewidth = LW2, label='Spacing constraint')
            ax2.plot(np.array(mr['iteration']), [(-x / len(mr['y_' + cz[0]][i])) if x is not None else np.nan for i, (x, cz) in enumerate(zip(mr['bound_violation'], mr['cur_zone']))], 'b--', linewidth = LW2, label='Boundary constraint')
        ax2.set_ylabel('Constraint violation [m/turbine]')
        ax2.tick_params(axis='x', labelsize=FS - 1)
        ax2.tick_params(axis='y', labelsize=FS - 1)
        if ax2_ystep: ax2.set_yticks(range(0, 41, ax2_ystep))
        # ax2.set_yscale('log')
        # legend
        line1, label1 = ax.get_legend_handles_labels()
        line2, label2 = ax2.get_legend_handles_labels()
        ax.legend(line1 + line2, label1 + label2, fontsize=FS,ncol=2)
        if ylim2: ax2.set_ylim(ylim2)
    else:
        ax.legend(fontsize=FS)
    # decorate
    ax.grid(True)
    ax.set_xlabel('Iteration', fontsize=FS)
    ax.set_ylabel(plotstr, fontsize=FS)
    # ax.set_title(plotstr, fontsize=FS + 2)
    ax.tick_params(axis='x', labelsize=FS - 1)
    ax.tick_params(axis='y', labelsize=FS - 1)
    fig.tight_layout()
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([0,max(mr['iteration'])])
    if ylim: ax.set_ylim(ylim)
    
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
            
        # update kwargs
        extra_vars['cable_cost_n'] = cable_cost_n
        extra_vars['mp_cost_n'] = mp_cost_n
        
        # other
        boundplot = list(boundaries.values())
        plot_folder = "Figures//Figures_processed_" + File
        opt_nr = data['opt_nr'][idx]
        metrics_recorder['sgd_constraint_violation'].append(data['sgd_constraint_violation'][idx])
        metrics_recorder['tur_dist_violation'].append(data['tur_dist_violation'][idx])
        metrics_recorder['bound_violation'].append(data['bound_violation'][idx])
        if opt_nr_check != opt_nr:
            # update setting only for new zone
            metrics_recorder['settings'].append(data['settings'][opt_nr_check])
            opt_nr_check = opt_nr
        #
        # run
        lcoe_func(x0,y0,**extra_vars)
        # plot
        if plot_postpro:
            plt.figure()
            plot = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sub_x, Sy=Sub_y, cables=cables_plot, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot, opt_nr=1, folder=plot_folder, sampling=sample, obj=obj, optimize=False, iter_nr=i, paper=False)
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
    plot_folder = "Figures//" + File
    # Initial Layout
    x0, y0 = np.concatenate(tuple(coords[name] for name in wf)).T
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
    Sx = Subs_x
    Sy = Subs_y
    nb = []
    nnb = []
    curzone = 'all'
    learning_rate = windTurbines.diameter()*0.2
    
    # Plot or not
    if plot_iter:
        plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=True, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sub_x, Sy=Sub_y, cables=cables_plot, metrics_recorder=metrics_recorder, b=boundplot, folder=plot_folder, sampling=sample, obj=obj, ploteach=plot_each)
    else:
        plot_comp = None
    
    # Max or min
    if obj == 'lcoe':
        maximize = False
    elif obj == 'aep':
        maximize = True
    
    # record settings
    metrics_recorder['settings'].append({'Mode':Mode,'Model':Model,'seed:':seed,'d_RD':d_RD,'tur_nr':tur_nr,'maxiter':maxiter,'learning_rate':learning_rate,'sgd_thresh':sgd_thresh,
                                         'curzone':curzone,'obj':obj,'Sequence':Sequence,'x0':x0,'y0':y0,'sample':sample,'samps':samps,'SepCabling':SepCabling,'Sx':Sx,'Sy':Sy,'depths':depths,'masses':masses,
                                         'CableSolver':CableSolver,'tl_metaheuristic':tl_metaheuristic,'tl_milp':tl_milp,'mip_gap':mip_gap}) 
    [metrics_recorder[key].append(None) for key in ["sgd_constraint_violation", "tur_dist_violation", "bound_violation"]]   # first run: no optimization
    
    # define cost and gradient function with handed over extra_vars
    cost_func = partial(lcoe_func, **extra_vars)
    cost_grad_func = partial(lcoe_jac, **extra_vars)
        
    # Optimization setup
    tf = TopFarmProblem(
            design_vars = {'x':x0, 'y':y0},         
            cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=sum(tur_nr), cost_function=cost_func, objective=True, cost_gradient_function=cost_grad_func, maximize=maximize),
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
    with open("Results//" + File + ".pkl", "wb") as file:
        pickle.dump({"metrics_recorder": metrics_recorder, "state": state, "recorder": recorder.recorder2list()}, file)
        
    # Postprocess for full wind rose
    if sample:
        print('Optimization finished.')
        print('Starting postprocessing...')
        metrics_recorder = postprocess_recorder(metrics_recorder)
        # Save processed file
        with open("Results//" + File + "_processed.pkl", "wb") as file:
            pickle.dump({"metrics_recorder": metrics_recorder}, file)
    
    # Plot LCOE iterations
    plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=1,optfat=0,feas=1)

#%% Competitive design
elif Mode == 'competitive':
    if Continue:
        print('---- !! Careful, starting from foregoing design !! ----')
        file = "metric_recorder_sequential_samelr_linint_nmsnmsnms"
        with open("C:\\Software\\IEA-Wind-2200-22-ROWP\\examples\\Results\\" + file + ".pkl", "rb") as file:
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
    plot_folder = "Figures//" + File
        
    # go through each zone as specified in Sequenc
    for i in loop_range:
        now = datetime.now()
        print('Evaluate zone ' + str(i+1) + '/' + str(len(Sequence)) + ' for seed ' + str(seed) + '. Time: ' + now.strftime("%H:%M:%S"))
        opt_nr = i+1
        curzone = Sequence[i]
        # Initial Layout
        if any(metrics_recorder['x_'+curzone]):
            x0 = np.array(metrics_recorder['x_'+curzone][-1])
            y0 = np.array(metrics_recorder['y_'+curzone][-1])
        else:
            x0, y0 = coords[curzone].T
            # x0 = [550720.266374936,550720.266374936,552353.266374936,552353.266374936,552353.266374936,552353.266374936,552353.266374936,553986.266374936,553986.266374936,553986.266374936,553986.266374936,553986.266374936,553986.266374936,553986.266374936,555619.266374936,555619.266374936,555619.266374936,555619.266374936,555619.266374936,555619.266374936,555619.266374936,555619.266374936,555619.266374936,555619.266374936,557252.266374936,557252.266374936,557252.266374936,557252.266374936,557252.266374936,557252.266374936,557252.266374936,557252.266374936,558885.266374936]
            # y0 = [5839886.96637494,5841519.96637494,5836620.96637494,5838253.96637494,5839886.96637494,5841519.96637494,5843152.96637494,5834987.96637494,5836620.96637494,5838253.96637494,5839886.96637494,5841519.96637494,5843152.96637494,5844785.96637494,5833354.96637494,5834987.96637494,5836620.96637494,5838253.96637494,5839886.96637494,5841519.96637494,5843152.96637494,5844785.96637494,5846418.96637494,5848051.96637494,5838253.96637494,5839886.96637494,5841519.96637494,5843152.96637494,5844785.96637494,5846418.96637494,5848051.96637494,5849684.96637494,5846418.96637494]
        
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
            plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=True, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sub_x, Sy=Sub_y, cables=cables_plot, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot, opt_nr=opt_nr, folder=plot_folder, sampling=sample, obj=obj, ploteach=plot_each)
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
                
        # record settings
        [metrics_recorder[key].append(None) for key in ["sgd_constraint_violation", "tur_dist_violation", "bound_violation"]]   # first run: no optimization
        metrics_recorder['settings'].append({'Mode':Mode,'Model':Model,'seed:':seed,'d_RD':d_RD,'tur_nr':tur_nr,'maxiter':maxiter,'learning_rate':learning_rate,'sgd_thresh':sgd_thresh,
                                             'curzone':curzone,'obj':obj,'Sequence':Sequence,'x0':x0,'y0':y0,'sample':sample,'samps':samps,'SepCabling':SepCabling,'Sx':Sx,'Sy':Sy,'depths':depths,'masses':masses,
                                             'CableSolver':CableSolver,'tl_metaheuristic':tl_metaheuristic,'tl_milp':tl_milp,'mip_gap':mip_gap}) 
        
        # update kwargs
        extra_vars['cable_cost_n'] = cable_cost_n
        extra_vars['mp_cost_n'] = mp_cost_n
        
        # define cost and gradient function with handed over extra_vars
        cost_func = partial(lcoe_func, **extra_vars)
        cost_grad_func = partial(lcoe_jac, **extra_vars)
        
        # Optimization Setup
        tf = TopFarmProblem(
                design_vars = {'x':x0, 'y':y0},         
                cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=tur_nr[wf[Sequence[i]]], cost_function=cost_func, objective=True, cost_gradient_function=cost_grad_func, maximize=maximize),
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
        with open("Results//" + File + ".pkl", "wb") as file:
            pickle.dump({"metrics_recorder": metrics_recorder, "state": state, "recorder": recorder.recorder2list()}, file)
   
    # Postprocess for full wind rose
    if sample:
        print('Optimization finished.')
        print('Starting postprocessing...')
        metrics_recorder = postprocess_recorder(metrics_recorder)
        # Save processed file
        with open("Results//" + File + "_processed.pkl", "wb") as file:
            pickle.dump({"metrics_recorder": metrics_recorder}, file)
            
    # Plot LCOE iterations
    plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=1,optfat=1,feas=1)

#%% Manually start postprocessing
elif Mode == 'evaluate_multiter':
    with open("Results\\" + File + ".pkl", "rb") as file:
        data = pickle.load(file)
    data = data['metrics_recorder']
    metrics_recorder = postprocess_recorder(data)
    # Save processed file
    with open("Results\\" + File + "_processed.pkl", "wb") as file:
        pickle.dump({"metrics_recorder": metrics_recorder}, file)
    plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=1,optfat=1)
    plot_convergence(mr=metrics_recorder,item='aep',plotstr='AEP (GWh)',obj=0,overall=0,optfat=1)
    plot_convergence(mr=metrics_recorder,item='cable_cost',plotstr='Cable Cost (€)',obj=0,overall=0,optfat=1)
    plot_convergence(mr=metrics_recorder,item='mp_cost',plotstr='Monopile Cost (€)',obj=0,overall=0,optfat=1)
    
#%% Load metric_recorder and plot
elif Mode == 'evaluate_recorder':
    with open("Results\\" + File + ".pkl", "rb") as file:
        data = pickle.load(file)
    metrics_recorder = data['metrics_recorder']
    if plot_iter:
        plot_folder = "Figures//FinalResult"
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
        plt.gcf().savefig("Figures//FinalLayout.pdf", dpi=500, pad_inches=0)
        
        # plt.text(555000, 5842000, 'N', color='white', fontsize=22, ha='center', va='center')
        # plt.text(548000, 5833000, 'M', color='white', fontsize=22, ha='center', va='center')
        # plt.text(543000, 5823000, 'S', color='white', fontsize=22, ha='center', va='center')
        # plt.gcf().savefig("Bathymetry.pdf", pad_inches=0, dpi=500)
        
    plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=1,optfat=0,feas=0)
    # plot_convergence(mr=metrics_recorder,item='aep',plotstr='AEP (GWh)',obj=0,overall=0,optfat=1)
    # plot_convergence(mr=metrics_recorder,item='cable_cost',plotstr='Cable Cost (€)',obj=0,overall=0,optfat=1)
    # plot_convergence(mr=metrics_recorder,item='mp_cost',plotstr='Monopile Cost (€)',obj=0,overall=0,optfat=1)
#%% Compare Cabling
elif Mode == 'CompareCabling':
    Files = ["test3_heuristic_processed",
             "test3_metaheuristic_processed",
             "test3_ortools_processed",
             "test3_gurobi_processed",
             "test3_cplex_cont_processed"]
    CabOpt = ["heuristic","metaheuristic","ortools","gurobi","cplex"]
    
    fig, axs = plt.subplots(1, len(Sequence), figsize=(15, 4))
    for i, zone in enumerate(Sequence):
        ax = axs[i]
        for f, File in enumerate(Files):
        # File = "metric_recorder_cooperative_2D_6000it_processed"
            # specify file you want to load
            with open("C:\\Software\\IEA-Wind-2200-22-ROWP\\examples\\Results\\" + File + ".pkl", "rb") as file:
                data = pickle.load(file)
            mr = data['metrics_recorder']
            item = 'lcoe'
            # item = 'cable_cost'
            plotstr = 'LCOE (€/MWh)'
            # plotstr = 'Cable Cost (€)'
            overall = 0
            optfat = 0
            obj = 0
            
            # plt.figure(figsize=(5, 3))
            ax.plot(
                np.array(mr['iteration'])[np.array([z[0] for z in mr['cur_zone']]) == zone],
                np.array(mr[item + '_' + zone])[np.array([z[0] for z in mr['cur_zone']]) == zone],
                label=zone + '_' + CabOpt[f],
                linewidth=1
            )
            # plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_'+zone]], label=zone+'_' + CabOpt[f], linewidth = 1)
            # plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_mid']], label='Mid_' + CabOpt[f], linewidth = 1)
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
        ax.legend(fontsize=FS)
        ax.grid()
        ax.set_xlabel('Iteration',fontsize=FS)
        ax.set_ylabel(plotstr,fontsize=FS)
        ax.set_title(zone,fontsize=FS+2)
        ax.tick_params(axis='both', labelsize=FS-1)
    
    # plt.figure(figsize=(5, 3))
    # for zone in Sequence:
    #     for f, File in enumerate(Files):
    #     # File = "metric_recorder_cooperative_2D_6000it_processed"
    #         # specify file you want to load
    #         with open("C:\\Software\\IEA-Wind-2200-22-ROWP\\examples\\Results\\" + File + ".pkl", "rb") as file:
    #             data = pickle.load(file)
    #         mr = data['metrics_recorder']
    #         item = 'lcoe'
    #         # item = 'cable_cost'
    #         plotstr = 'LCOE (€/MWh)'
    #         # plotstr = 'Cable Cost (€)'
    #         overall = 0
    #         optfat = 0
    #         obj = 0
            
    #         # plt.figure(figsize=(5, 3))
    #         plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_'+zone]], label=zone+'_' + CabOpt[f], linewidth = 1)
    #         # plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_mid']], label='Mid_' + CabOpt[f], linewidth = 1)
    #         # plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_south']], label='South_' + CabOpt[f], linewidth = 1)
    #         if obj:
    #             plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item]], label='Overall', linewidth = 1)
    #         if overall:
    #             plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_all']], label='Overall', linewidth = 1)
    #         if optfat:
    #             for i in range(0,len(mr['lcoe_all'])-1):
    #                 if i == 0 and f == len(Files)-1:
    #                     leg = 'Current objective'
    #                 else:
    #                     leg = None
    #                 if mr['opt_nr'][i] == mr['opt_nr'][i+1] and mr['cur_zone'][i][0] == 'mid':
    #                     plt.plot(np.array(mr['iteration'][i:i+2])-1,
    #                         np.array(mr[item+'_'+mr['cur_zone'][i][0]][i:i+2]), linestyle="--", linewidth = 0.8, color="black", label=leg)
    # FS = 9
    # plt.legend(fontsize=FS)
    # plt.grid()
    # plt.xlabel('Iteration',fontsize=FS)
    # plt.ylabel(plotstr,fontsize=FS)
    # plt.title(plotstr,fontsize=FS+2)
    # plt.xticks(fontsize=FS-1)
    # plt.yticks(fontsize=FS-1)
#%% Evaluate layout
elif Mode == 'evaluate_layout':
    # x_eva = [550507.3,551004.3,552353.3,551501.3,552850.3,550010.3,552353.3,553915.3,553986.3,553986.3,559453.3,553986.3,551501.3,554057.3,555619.3,555619.3,555619.3,555619.3,556684.3,555690.3,555619.3,555619.3,555619.3,555690.3,557181.3,557749.3,558033.3,558104.3,559311.3,557465.3,557394.3,558033.3,558885.3]
    # y_eva = [5841236.0,5842585.0,5836621.0,5837686.0,5841591.0,5839603.0,5844218.0,5846064.0,5836692.0,5839745.0,5849898.0,5844076.0,5839319.0,5834491.0,5833639.0,5835485.0,5836834.0,5839035.0,5834704.0,5841804.0,5843934.0,5846135.0,5848123.0,5832716.0,5849969.0,5840242.0,5842088.0,5843934.0,5848265.0,5848123.0,5838325.0,5850963.0,5846135.0]
    # from matlab (best layout):
    # x_eva = [555889.0664,555661.8664,553730.6664,554525.8664,557593.0664,552253.8664,556229.8664,554185.0664,550322.6664,551345.0664,559524.2664,556002.6664,556797.8664,552594.6664,549300.2664,557820.2664,556911.4664,554412.2664,553276.2664,551004.2664,558047.4664,558161.0664,555207.4664,552481.0664,558388.2664,552935.4664,555548.2664,558956.2664,556570.6664,554753.0664,559297.0664,556911.4664,558956.2664]
    # y_eva = [5833653.166,5837969.966,5834902.766,5833880.366,5839333.166,5836833.966,5844104.366,5839333.166,5839219.566,5837969.966,5849670.766,5841037.166,5835243.566,5840923.566,5840469.166,5840923.566,5836833.966,5842513.966,5845240.366,5842513.966,5842741.166,5851147.566,5845694.766,5844104.366,5844558.766,5835925.166,5832630.766,5846489.966,5848080.366,5847057.966,5848193.966,5849670.766,5850579.566]
    # benchmark from this optimization code:
    # x_eva=[550138.93628142,549259.97124703,551947.1687341,551019.50470179,553801.57761174,550508.20477946,551773.18524297,554633.28768032,553717.07111651,552856.50282707,554997.27299282,552460.96597628,553006.20209499,554242.14168257,555573.09986941,556563.82476249,556932.45031753,557538.99869909,555410.58799388,554852.39596903,557683.80991463,556552.40023336,555594.99499565,556896.35505898,557230.10925774,557844.72557715,558155.13816112,558470.59021471,558975.42699074,559585.90789837,558953.69877364,558353.63201273,559276.98154229]
    # y_eva=[5839398.84857337,5840513.67240592,5837104.40069435,5838282.78628338,5839364.62296964,5842008.46254412,5843523.87890845,5833698.25831585,5834861.06881341,5835951.85926047,5837961.09093571,5841081.87001151,5845001.13167038,5846482.04930752,5832507.51262233,5834245.45600763,5835810.9687916,5838975.57066336,5841687.83415406,5843776.80017039,5844991.23114793,5846610.60971491,5848103.08146724,5849660.05563233,5837356.21944075,5840574.79147581,5842202.57087209,5843855.86226534,5846500.73447366,5849700.21857155,5850583.80185068,5851407.91239366,5848083.88552206]
    with open("Results\wesc_res_comp_6D_S2_12it_processed.pkl","rb") as file:
        data = pickle.load(file)
    metrics_recorder=data['metrics_recorder']
    extra_vars['metrics_recorder'] = metrics_recorder
    curzone = 'north'
    x_eva = metrics_recorder['x_north'][-1]
    y_eva = metrics_recorder['y_north'][-1]
    xn = metrics_recorder['x_mid'][-1] + metrics_recorder['x_south'][-1]
    yn = metrics_recorder['y_mid'][-1] + metrics_recorder['y_south'][-1]
    # xn = []
    # yn = []
    nb = ['mid','south']
    nnb = []
    Sx = [Subs_x[wf[curzone]]]
    Sy = [Subs_y[wf[curzone]]]
    res = lcoe_func(x_eva,y_eva,**extra_vars)
    
    plot_folder = "Figures//FinalResult"
    boundplot = list(boundaries.values())
    plot = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=True, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sub_x, Sy=Sub_y, cables=cables_plot, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot, opt_nr=opt_nr, folder=plot_folder, sampling=sample, obj=None, optimize=False, paper=True)
    inputs = {}
    inputs['x'] = x_eva
    inputs['y'] = y_eva
    plot.compute(inputs,[])
        
#%% Run layout optimization for different initial layouts  
# if __name__ == "__main__":
#     run_script(seed=3)
#     # seeds = np.arange(1,200)     # random np seed for initial layout configuration
#     # num_workers = 4         # number of workers for parallelization
#     # with Pool(processes=num_workers) as pool:
#     #     pool.map(run_script, seeds)