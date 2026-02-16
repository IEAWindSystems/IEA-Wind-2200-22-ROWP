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
import matplotlib.pyplot as plt
import xarray as xr
import utm
import pickle
from datetime import datetime
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from scipy.interpolate import RegularGridInterpolator
from types import SimpleNamespace
import os
import copy
import gc

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
from py_wake.literature import Bastankhah_PorteAgel_2014, Niayifar_PorteAgel_2016
from py_wake.rotor_avg_models import RotorCenter
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasySGDDriver
from topfarm.constraint_components.boundary import InclusionZone
from topfarm.constraint_components.boundary import MultiWFBoundaryConstraint, BoundaryType
from topfarm.constraint_components.constraint_aggregation import DistanceConstraintAggregation
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
from TopfarmAdvancedConstraints import CorrectedXYBoundaryConstraint, SpacingConstraintWithAdditionalTurbines
from TopfarmAdvancedConstraints import DistanceConstraintAggregation as DistanceConstraintAggregationAdvanced
#
#%% INPUTS
# Parallelization
# seeds = [3]
seeds = np.arange(1,250)        # random np seed for initial layout configuration. If more than 1, parallel execution.
num_workers = 48                # number of workers for parallel execution if len(seeds)>1 

# General inputs
#
# For Mode, pick from the following scenarios:
# 'cooperative'         ...optimize the developed zones together in order to minimize/maximize the overall cluster objective
# 'competitive'         ...optimize the developed zones sequentially in the indicated sequence. The farm respective KPI is the objective. Iterative sequential optimization is possible.
# 'evaluate_recorder'   ...evaluate a certain file for convergence, incl. a final layout plot if desired
# 'evaluate_multiter'   ...postprocess a file by evaluating the recorded layouts over the whole windrose (required when sampling was on during optimization and postprocessing was not automatically executed). Specify in which recorder intervals the layouts should be evaluated.
# 'evaluate_layout'     ...evaluate a certain layout (incl. neighbours if needed) for the desired KPI
# 'evaluate_seeds'      ...evaluate the recorders of multiple seeds to identify which seed performed the best and plot the distributions
# 'refine_opt_results'  ...reoptimize unfeasible layouts (=constraint violations) by only enforcing the constraints and ignoring the objective
#
Mode = 'competitive'                            # 'cooperative' or 'competitive' or 'evaluate_recorder' or 'evaluate_multiter' or 'evaluate_layout' or 'evaluate_seeds' or 'refine_opt_results'
Continue = False                                # set to True if you give foregoing metrics_recorder to continue optimization
File = 'comp_TI'                                   # define name of files that is stored or loaded. the seed will be added as e.g. "_s3"
Sequence = ['north','mid','south']*5            # define sequence of zones for sequential design
CableSolver_opt = 'MetaHeuristic'               # Cable solver using during nested optimization: 'Heuristic', 'MetaHeuristic', 'cplex', 'ortools' or 'gurobi'
CableSolver_final = 'ortools'                   # Cable solver used for final cabling plan optimization.
Model = 'turbopark'                             # 'jensen', 'gauss' or 'turbopark'
tur_nr = {"north": 33, "mid": 33, "south": 34}  # turbine number assigned to each zone
obj = 'lcoe'                                    # 'lcoe' or 'aep'
plot_iter = True                                # True or False: plot and store layouts during optimization each plot_each iterations
plot_postpro = True                             # True or False: plot and store layouts during postprocessing (how often is linked to step)
plot_each = 500                                  # define in which interval a plot should be made
d_RD = 6                                        # min spacing distance in rotor diameters
step = 500                                      # at each "step" iterations, the full wind rose is recalculated in postprocessing (when sampling is used during opt)

# plot lims
xlim = None                             # specify xlim for convergence plot or put None
ylim = None                             # specify ylim for axis1 (obj) for convergence plot or put None
ylim2 = None                            # specify ylim for axis2 (penalty) for convergence plot or put None
ax2_ystep = None                        # specify tick step for axis2 or put None

# SGD
maxiter = 3000                          # maximum nr of iterations for SGD opt
sgd_thresh = 0.02                       # SGD threshold

# Monopile optimization
MP_data = 'LUT'                         # 'LUT' for mass-vs-depth look-up tables, 'Surrogate' for mass surrogate model
MP_LUT_file = 'ssms//Mass_Monopile_22MW_extrapolated.csv'   # file name of the LUT if option chosen
MP_ref = 1                              # reference turbine type for monopile mass scaling if surrogate chosen. 0 = 10MW, 1 = 15MW, 2 = 3.4MW

# lcoe parameters
d = 0.0661                              # [-] discount rate, from NREL CoE 2024 report
life = 25                               # years, lifetime
capex = 9.6734e7 * 0.924                # €2024 per turbine, excl. Monopile and cabling, from NREL COE Report 2024, converted to € with 2024 average exchange rate
OpexAnnual = 2.97e6 * 0.924             # €2024 per turbine, annual OPEX,from NREL COE Report 2024, converted to € with 2024 average exchange rate
LP = 0                                  # $2010 per turbine, liquidation proceeds, from DETECT for HKN scaled (22MW turbines)

# mass surrogate
PlatformHeight = 15                     # Platform height [m]
WaveHeight = 1.5                        # Significant Wave Height [m], taken from HKW metocean report p. 183 (There: 1.40 --> approximated with 1.5 as it is the lower bound of the mass surrogate model)
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
wd_wb = resource_dat['wind_resource']['wind_direction']
TI_org =  resource_dat['wind_resource']['turbulence_intensity']['data']
ws_TI = resource_dat['wind_resource']['wind_speed']
site = XRSite(
       ds=xr.Dataset(data_vars=
                        {'Sector_frequency': ('wd', freq['data']), 
                         'Weibull_A': ('wd', A['data']), 
                         'Weibull_k': ('wd', k['data']),
                         'TI': ('ws', [0.05]*2 + TI_org + [0.05]*2)   # add dummies to avoid interpolation errors during sampling
                         },
                      coords={'wd': wd_wb, 'ws': [0,ws_TI[0]-0.01] + ws_TI + [ws_TI[-1]+0.01,100]}))

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
elif Model == 'gauss2014':
    wake_model = Bastankhah_PorteAgel_2014(site, windTurbines, rotorAvgModel=RotorCenter(), turbulenceModel=CrespoHernandez(), k=0.0324555)
elif Model == 'gauss2016':
    wake_model = Niayifar_PorteAgel_2016(site, windTurbines, rotorAvgModel=RotorCenter(), turbulenceModel=CrespoHernandez(rotorAvgModel=RotorCenter()))
elif Model == 'turbopark':
    wake_model = Nygaard_2022(site, windTurbines)
    
# wind resource
dirs = np.arange(0, 360, 1) #wind directions
ws = np.arange(cut_in, cut_out+1, 1)
freqs = site.local_wind([0], [0], wd=dirs, ws=ws).P_ilk[0, :, :].sum(1)     # all frequencies
freqs = freqs / freqs.sum()
TI = np.interp(ws, ws_TI, TI_org)

# for sampling:
As = site.local_wind([0], [0], wd=dirs, ws=ws).Weibull_A_ilk[0, :, 0]               #weibull A
ks = site.local_wind([0], [0], wd=dirs, ws=ws).Weibull_k_ilk[0, :, 0]               #weibull k

# Boundaries and substations
b = system_dat['site']
Subs_x_raw = system_dat['wind_farm']['electrical_substations']['electrical_substation']['coordinates']['x']
Subs_y_raw = system_dat['wind_farm']['electrical_substations']['electrical_substation']['coordinates']['y']
# Mapping for wind farms to indices
wf = {"north": 0,"mid": 1,"south": 2}
boundaries = {
    name: np.array([
        b['boundaries']['polygons'][index]['x'],
        b['boundaries']['polygons'][index]['y']
    ]).T
    for name, index in wf.items()
}
Subs_x = {name: Subs_x_raw[index] for name, index in wf.items()}
Subs_y = {name: Subs_y_raw[index] for name, index in wf.items()}

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
if MP_data == 'Surrogate':
    # A mass surrogate model only with 20MW results is built. Approximation of the 22MW machine.
    trainQLS()
    depths = np.linspace(np.min(-Z),np.max(-Z),20)
    import scipy.special as sp
    def mean_wind_speed(A, k):
        return A * sp.gamma(1 + 1/k)
    V_ave = []
    for i in range(len(wd_wb)):
        V_ave.append(mean_wind_speed(A['data'][i],k['data'][i]))
    V_ave = np.sum(np.array(V_ave) * np.array(freq['data']))
    masses = []
    for z in depths:
       cur_mass = CalculateMass(D=rd, HTrans=PlatformHeight, HHub=hh, WaterDepth=z, WaveHeight=WaveHeight, WavePeriod=WavePeriod, WindSpeed=V_ave, IP_item=1)
       masses.append(cur_mass[0][0])
elif MP_data == 'LUT':
    # Load external file with mass-vs-depth data
    df = pd.read_csv(MP_LUT_file)
    depths = df['Depth_m'].tolist()
    masses = df['Mass_kg'].tolist()
    
# add transition piece (100t, from 22MW report)
masses = [x + 100000 for x in masses]

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
# CableSolvers = ['MetaHeuristic', 'Heuristic', 'cplex', 'gurobi', 'ortools']         # Available solvers in default order
# CableSolvers_order = [CableSolver] + [s for s in CableSolvers if s != CableSolver]  # Try primary solver first
tl_opt = 0.3      # time limit during optimization
tl_final = 3600   # time limit for final run
mip_gap_opt = 0.005
mip_gap_final = 0.0005
cables = np.array(
    [(int(c["capacity_NrT"]), float(c["cost_€_m"])) for c in cable_specs],
    dtype=[("capacity", int), ("cost", float)])
cables_plot = np.array([[c["diameter_mm2"], c["capacity_NrT"], c["cost_€_m"]] for c in cable_specs])

# results and figure folders
os.makedirs("Results", exist_ok=True)
os.makedirs("Figures", exist_ok=True)
    
# defaults
# neighbour wind farm with turbine coordinates and costs to consider
nf = False
cable_cost_n = [0,0]
mp_cost_n = [0,0]
if Mode == 'cooperative':
    Sequence = list(dict.fromkeys(Sequence))
    
# update font to latex
import matplotlib.font_manager as font_manager
font_dir = [r"font\Serif"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
plt.rcParams["font.family"] = "CMU Serif"
plt.rcParams['svg.fonttype'] = 'path'
#
#%% Function to create the random sampling of wind speed and wind directions
def sampling():
    # todo: use inputs only, no global variables!
    ind = np.random.choice(np.arange(dirs.size), samps, p=freqs)
    wd = dirs[ind]
    A = As[ind]
    k = ks[ind]
    ws = A * np.random.weibull(k)
    ti = np.interp(ws, ws_TI, TI_org, left=0.05, right=0.05)    # dummy values outside operating range to avoid site interpolation errors
    return wd, ws, ti

#%% Cabling optimization
def opt_cabling(x=None,y=None,Sx=None,Sy=None,cables=None,CableSolver='MetaHeuristic',time_limit=0.3, mip_gap=0.005):
    if CableSolver in ['cplex', 'ortools', 'gurobi']:
        # warm start
        wfn = WindFarmNetwork(turbinesC=np.column_stack((x,y)),substationsC=np.column_stack((Sx, Sy)),cables=cables,router=HGSRouter(time_limit=0.3))
        wfn.optimize()
        # chosen router
        wfn.optimize(router=MILPRouter(solver_name=CableSolver, time_limit=time_limit, mip_gap=mip_gap, verbose=False))
    elif CableSolver == 'MetaHeuristic':
        wfn = WindFarmNetwork(turbinesC=np.column_stack((x, y)),substationsC=np.column_stack((Sx, Sy)),cables=cables,router=HGSRouter(time_limit=time_limit))
        wfn.optimize()
    elif CableSolver == 'Heuristic':
        wfn = WindFarmNetwork(turbinesC=np.column_stack((x, y)),substationsC=np.column_stack((Sx, Sy)),cables=cables,router=EWRouter())
        wfn.optimize()
    return wfn

#%% Objective function
def lcoe_func(x, y, **kwargs):
    # 0.) make kwargs available
    args = SimpleNamespace(**kwargs)
    metrics_recorder = kwargs["metrics_recorder"]
    #
    # 1.) aep
    if args.sample:
        wd_current, ws_current, TI_current = sampling()
        Time = True
    else:
        wd_current = args.dirs
        ws_current = args.ws
        TI_current = args.TI
        Time = False
    if args.nf:
        aep = args.wake_model(x=np.concatenate((x,args.xn)), y=np.concatenate((y,args.yn)), wd=wd_current, ws=ws_current, TI=TI_current, time=Time).aep() * 1e3
    else:
        aep = args.wake_model(x=x, y=y, wd=wd_current, ws=ws_current, TI=TI_current, time=Time).aep() * 1e3
    #
    # 2.) monopile costs
    depths = args.depth_interp(x, y)
    masses = []
    dmasses = []
    for water_depth in depths:
        dmasses.append(args.polynomial_gradients(water_depth))
        masses.append(args.polynomial(water_depth))
    mp_cost = 3.636 * 0.924 * np.array(masses)  # from ORBIT 2025 monopile_steel_cost default
    dmasses = (np.array(dmasses) * args.get_depth_grads(x, y)[:, 0, :].T)    # gradients (for function later, to avoid double calculus)
    dmp_cost = 3.636 * 0.924 * dmasses          # from ORBIT 2025 tp_steel_cost default
    #
    # 3.) Cable costs
    if args.CableOpt == 'multi_sub':
        # optimize each subzone with respective substation sequentially
        indices = [np.arange(sum(args.tur_nr[zone] for zone in args.Sequence[:i]),sum(args.tur_nr[zone] for zone in args.Sequence[:i+1])) for i in range(len(args.Sequence))]
        cable_costs = []
        dcable_cost = np.empty((0, 2))
        for z, zone in enumerate(args.Sequence):
            wfn = opt_cabling(x=x[indices[z]],y=y[indices[z]],Sx=args.Sx[z],Sy=args.Sy[z],cables=args.cables,CableSolver=args.CableSolver,time_limit=args.time_limit,mip_gap=args.mip_gap)
            cable_costs.append(wfn.cost())
            record_cable_metrics(metrics_recorder, wfn, zone, args.nnb, args.nb)
            cur_dcable_cost, _ = wfn.gradient(gradient_type='cost')
            dcable_cost = np.vstack((dcable_cost,cur_dcable_cost))
        cable_cost = sum(cable_costs)
    elif args.CableOpt == 'single_sub':
        # all turbines connected to one substation
        wfn = opt_cabling(x=x,y=y,Sx=args.Sx,Sy=args.Sy,cables=args.cables,CableSolver=args.CableSolver,time_limit=args.time_limit,mip_gap=args.mip_gap)
        cable_cost = wfn.cost()     # Costs in Euro
        record_cable_metrics(metrics_recorder, wfn, args.curzone, args.nnb, args.nb)   # recorder
        dcable_cost, _ = wfn.gradient(gradient_type='cost')             # gradients (for function later, to avoid double calculus)
    elif args.CableOpt == 'off':
        # read cable costs from kwargs
        cable_cost = args.cable_cost_external
        cable_costs = args.cable_costs_external
        dcable_cost = None
    # !! ToDo: Scale costs to same currency and year of reference
    #
    # 4.) lcoe
    CRF = args.d / (1 - (1 + args.d) ** -args.life)
    npv = (args.capex*len(x) + np.sum(mp_cost) + cable_cost + args.LP*len(x)) * CRF + args.OpexAnnual*len(x)
    lcoe = npv / aep.isel(wt=slice(0,len(x))).sum().item()
    #
    # 5. Record the missing metrics
    if args.Mode == 'cooperative' or args.curzone == 'all':
        record_main_metrics_multisub(metrics_recorder, args.opt_nr, aep, x, y, mp_cost, cable_costs, lcoe,
            args.curzone, args.nb, args.nnb, args.tur_nr, args.cable_cost_n, args.mp_cost_n,
            npv, args.capex, args.LP, CRF, args.OpexAnnual, args.Sequence)
    elif args.Mode == 'competitive' or args.Mode == 'evaluate_multiter':
        record_main_metrics_singlesub(metrics_recorder, args.opt_nr, aep, x, y, mp_cost, cable_cost, lcoe,
            args.curzone, args.nb, args.nnb, args.tur_nr, args.cable_cost_n, args.mp_cost_n,
            npv, args.capex, args.LP, CRF, args.OpexAnnual)
    #
    # 5.) Store results into kwargs that need to be handed over to derivative function  
    kwargs["aep"]["value"] = aep.isel(wt=slice(0,len(x))).sum().item()
    kwargs["cable_cost"]["value"] = cable_cost
    kwargs["dcable_cost"]["value"] = dcable_cost
    kwargs["mp_cost"]["value"] = mp_cost
    kwargs["dmp_cost"]["value"] = dmp_cost
    kwargs["wd_current"]["value"] = wd_current
    kwargs["ws_current"]["value"] = ws_current
    kwargs["TI_current"]["value"] = TI_current
    kwargs["Time"]["value"] = Time
    #
    if args.obj == 'lcoe':
        return lcoe # $/MWh
    elif args.obj == 'aep':
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
    # 0.) Extract kwargs
    args = SimpleNamespace(**kwargs)
    #
    # 1.) aep
    if args.nf:
        daep = args.wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=np.concatenate((x,args.xn)), y=np.concatenate((y,args.yn)), ws=args.ws_current["value"], TI=args.TI_current["value"], wd=args.wd_current["value"], time=args.Time["value"])[:args.tur_nr[args.curzone],:args.tur_nr[args.curzone]] * 1e3
    else:
        daep = args.wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=x, y=y, ws=args.ws_current["value"], TI=args.TI_current["value"], wd=args.wd_current["value"], time=args.Time["value"]) * 1e3
    #
    # 2.) monopile costs
    # have been calculated earlier (global variable)
    #
    # 3.) Cable costs
    # have been calculated earlier (global variable)
    #
    # 4.) lcoe
    CRF = args.d / (1 - (1 + args.d) ** -args.life)
    dlcoe = (CRF*(args.dmp_cost["value"]+args.dcable_cost["value"].T)*args.aep["value"] - ((args.capex*len(x)+args.LP*len(x)+np.sum(args.mp_cost["value"])+args.cable_cost["value"])*CRF+args.OpexAnnual*len(x))*daep) / (args.aep["value"] ** 2)
    #
    if args.obj == 'lcoe':
        return dlcoe # $/MWh
    elif args.obj == 'aep':
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

#%% Recorder + kwargs
metrics_recorder = create_recorder(Sequence)

# Store setting
metrics_recorder['general_settings'].append({'Mode':Mode,'Model':Model,'d_RD':d_RD,'tur_nr':tur_nr,'maxiter':maxiter,'sgd_thresh':sgd_thresh,'obj':obj,'Sequence':Sequence,'samps':samps,'boundaries':boundaries,'Subs_x':Subs_x,'Subs_y':Subs_y,'wf':wf, 
                                     'depths':depths,'masses':masses,'CableSolver_opt':CableSolver_opt,'CableSolver_final':CableSolver_final,'tl_opt':tl_opt,'tl_final':tl_final,'mip_gap_opt':mip_gap_opt,'mip_gap_final':mip_gap_final}) 

# Prepare kwargs (mutable for objective function + other required variables passed through functions) explicitly
extra_vars = dict(
    # mutable trackers
    metrics_recorder=metrics_recorder, aep={"value": None}, cable_cost={"value": None},dcable_cost={"value": None}, mp_cost={"value": None},
    dmp_cost={"value": None},wd_current={"value": None}, ws_current={"value": None}, TI_current={"value": None}, Time={"value": None},
    # other required variables
    Sequence=Sequence, boundaries=boundaries, File=File,Subs_x=Subs_x, Subs_y=Subs_y, X_utm=X_utm, Y_utm=Y_utm, Z=Z, cables=cables, cables_plot=cables_plot,
    obj=obj, plot_each=plot_each, windTurbines=windTurbines, Mode=Mode, tur_nr=tur_nr, maxiter=maxiter, sgd_thresh=sgd_thresh, step=step,
    samps=samps, CableSolver_opt=CableSolver_opt, CableSolver_final=CableSolver_final, tl_opt=tl_opt, tl_final=tl_final, mip_gap_opt=mip_gap_opt, mip_gap_final=mip_gap_final, wake_model=wake_model,
    min_spacing_m=min_spacing_m, cable_cost_n=cable_cost_n, mp_cost_n=mp_cost_n, ws=ws, dirs=dirs, TI =TI, d=d, life=life, LP=LP, capex=capex, OpexAnnual=OpexAnnual,
    polynomial_gradients=polynomial_gradients, polynomial=polynomial, get_depth_grads=get_depth_grads, depth_interp=depth_interp
)
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
        ax.legend(line1 + line2, label1 + label2, fontsize=FS,ncol=2,loc='upper right')
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
def postprocess_recorder(data, Sequence, step, Subs_x, Subs_y, boundaries, File, X_utm, Y_utm, Z, cables_plot, obj, **extra_vars):
    # new recorder
    metrics_recorder = create_recorder(Sequence)
    metrics_recorder['general_settings'].append(data['general_settings'][-1])
    
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
            Sx = [Subs_x[zone] for zone in Sequence]
            Sy = [Subs_y[zone] for zone in Sequence]
            nnb = []
        else:
            # sequential
            x0 = np.array(data['x_' + curzone][idx])
            y0 = np.array(data['y_' + curzone][idx])
            nnb = list(set(Sequence))
            nnb.remove(curzone)
            nnb = [x for x in nnb if x not in nb]
            Sx = [Subs_x[curzone]]
            Sy = [Subs_y[curzone]]
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
            
        # other
        boundplot = list(boundaries.values())
        plot_folder = "Figures//Figures_processed_" + File
        opt_nr = data['opt_nr'][idx]
        metrics_recorder['sgd_constraint_violation'].append(data['sgd_constraint_violation'][idx])
        metrics_recorder['tur_dist_violation'].append(data['tur_dist_violation'][idx])
        metrics_recorder['bound_violation'].append(data['bound_violation'][idx])
        if opt_nr_check != opt_nr:
            # update setting only for new zone
            metrics_recorder['current_settings'].append(data['current_settings'][opt_nr_check])
            opt_nr_check = opt_nr
        
        # copy some cable entries (cable plan already calculated)
        for k, v in data.items():
            if "cable" in k and "cable_cost" not in k and v:
                metrics_recorder[k].append(v[idx])
        cable_costs_external = []
        if curzone == 'all':
            for zone in Sequence:
                cable_costs_external.append(data['cable_cost_'+zone][idx])
        
        # update kwargs
        extra_vars.update(xn=xn, yn=yn, Sx=Sx, Sy=Sy, curzone=curzone, nb=nb, nnb=nnb, cable_cost_n=cable_cost_n, mp_cost_n=mp_cost_n, opt_nr=opt_nr,
                          nf=nf, sample=sample, File=File, metrics_recorder=metrics_recorder, Sequence=Sequence, obj=obj,
                          CableOpt='off', cable_cost_external=data["cable_cost"][idx], cable_costs_external=cable_costs_external)     
        
        # run
        lcoe_func(x0,y0,**extra_vars)
        metrics_recorder = extra_vars['metrics_recorder']
        
        # plot
        if plot_postpro:
            plt.figure()
            plot = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Subs_x, Sy=Subs_y, cables=cables_plot, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot, opt_nr=1, folder=plot_folder, sampling=sample, obj=obj, optimize=False, iter_nr=i, paper=False)
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
def opt_cooperative(seed,*,Sequence,boundaries,File,metrics_recorder,Subs_x,Subs_y,X_utm,Y_utm,Z,cables_plot,obj,
                    plot_each,windTurbines,tur_nr,maxiter,sgd_thresh,min_spacing_m,**extra_vars):
    
    # general options
    File += '_s' + str(seed)
    sample = True
    CableOpt = 'multi_sub'
    boundplot = list(boundaries.values())
    plot_folder = "Figures//" + File
    opt_nr = 1  # dummy
    curzone = 'all'
    Sx = [Subs_x[zone] for zone in Sequence]
    Sy = [Subs_y[zone] for zone in Sequence]
    nf = False
    nb = []
    nnb = []
    cable_cost_n = [0,0]
    mp_cost_n = [0,0]
    learning_rate = windTurbines.diameter()*0.2
    
    # Initial layout with current seed
    coords = {name: poisson_disc_filler(tur_nr[name], min_dist=0.8*min_spacing_m, BorderC=boundaries[name], seed=seed)
              for name in Sequence}
    x0, y0 = np.concatenate(tuple(coords[name] for name in Sequence)).T

    # Constraints
    boundary_constraint = MultiWFBoundaryConstraint(
        geometry = [boundaries[name] for name in Sequence],  # Boundary mapping
        wt_groups=[np.arange(sum(tur_nr[zone] for zone in Sequence[:i]),sum(tur_nr[zone] for zone in Sequence[:i+1])) for i in range(len(Sequence))],  # Turbine groups
        boundtype = BoundaryType.POLYGON
    )
    aggregated_constraints = DistanceConstraintAggregation(boundary_constraint, sum([tur_nr[zone] for zone in Sequence]), min_spacing_m, windTurbines)

    # Plot or not
    if plot_iter:
        plt.figure()
        plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=True, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Subs_x, Sy=Subs_y, cables=cables_plot, metrics_recorder=metrics_recorder, b=boundplot, folder=plot_folder, sampling=sample, obj=obj, ploteach=plot_each)
    else:
        plot_comp = None
    
    # Max or min
    if obj == 'lcoe':
        maximize = False
    elif obj == 'aep':
        maximize = True

    # record settings
    [metrics_recorder[key].append(None) for key in ["sgd_constraint_violation", "tur_dist_violation", "bound_violation"]]   # first run: no optimization
    metrics_recorder['current_settings'].append({'seed':seed,'learning_rate':learning_rate,'curzone':curzone,'x0':x0,'y0':y0,'sample':sample,'CableOpt':CableOpt,'Sx':Sx,'Sy':Sy})
    
    # update kwargs
    extra_vars.update(Sx=Sx, Sy=Sy, curzone=curzone, nb=nb, nnb=nnb, cable_cost_n=cable_cost_n, mp_cost_n=mp_cost_n, opt_nr=opt_nr, nf=nf, sample=sample, time_limit=tl_opt, mip_gap=mip_gap_opt,
                      CableOpt=CableOpt, File=File, metrics_recorder=metrics_recorder, Sequence=Sequence, obj=obj, tur_nr=tur_nr, boundaries=boundaries, X_utm=X_utm, Y_utm=Y_utm, Z=Z, cables_plot=cables_plot, CableSolver=CableSolver_opt)
    
    # define cost and gradient function with handed over extra_vars
    cost_func = partial(lcoe_func, **extra_vars)
    cost_grad_func = partial(lcoe_jac, **extra_vars)
    
    # Optimization setup
    tf = TopFarmProblem(
            design_vars = {'x':x0, 'y':y0},         
            cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=sum([tur_nr[zone] for zone in Sequence]), cost_function=cost_func, objective=True, cost_gradient_function=cost_grad_func, maximize=maximize),
            constraints = aggregated_constraints, 
            driver = EasySGDDriver(maxiter=maxiter, learning_rate=learning_rate, speedupSGD=True, sgd_thresh=sgd_thresh),
            plot_comp = plot_comp
            )
    
    # Run
    tic = time.time()
    cost, state, recorder = tf.optimize()
    toc = time.time()
    print('Optimization with SGD took: {:.0f}s'.format(toc-tic), ' with a total constraint violation of ', recorder['sgd_constraint'][-1])
    
    # postprocess recorder
    recorder = recorder.recorder2list()[1]['driver_iteration_dict']
    recorder['sgd_constraint'] = np.array(recorder['sgd_constraint']).flatten()
    recorder['wtSeparationSquared'] = np.array(recorder['wtSeparationSquared'])
    recorder['boundaryDistances'] = np.array(recorder['boundaryDistances'])
    
    # Final cabling optimization
    print('Final cabling optimization...')
    tic = time.time()
    extra_vars.update(time_limit=tl_final, mip_gap=mip_gap_final, CableSolver=CableSolver_final)
    lcoe_func(state['x'], state['y'], **extra_vars)
    toc = time.time()
    print('Final cabling optimization took: {:.0f}s'.format(toc-tic))
    # copy sgd recorder values
    recorder['sgd_constraint'] = np.append(recorder['sgd_constraint'],recorder['sgd_constraint'][-1])
    recorder['wtSeparationSquared'] = np.vstack([recorder['wtSeparationSquared'],recorder['wtSeparationSquared'][-1]])
    recorder['boundaryDistances'] = np.vstack([recorder['boundaryDistances'],recorder['boundaryDistances'][-1]])
    
    # Store
    metrics_recorder = extra_vars['metrics_recorder']
    record_results_constraints(metrics_recorder, recorder, state, cost, min_spacing_m)
    
    # Save to a file
    with open("Results//" + File + ".pkl", "wb") as file:
        pickle.dump({"metrics_recorder": metrics_recorder, "state": state, "recorder": recorder}, file)
        
    # Postprocess for full wind rose
    if sample:
        print('Optimization finished.')
        print('Starting postprocessing...')
        extra_vars.update(Subs_x=Subs_x,Subs_y=Subs_y)
        metrics_recorder = postprocess_recorder(metrics_recorder,**extra_vars)
        # Save processed file
        with open("Results//" + File + "_processed.pkl", "wb") as file:
            pickle.dump({"metrics_recorder": metrics_recorder}, file)
    
    # Plot LCOE iterations
    plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=1,optfat=0,feas=1)
    
#%% Competitive design
def opt_competitive(seed,*,Sequence,boundaries,File,metrics_recorder,Subs_x,Subs_y,X_utm,Y_utm,Z,cables_plot,obj,
                    plot_each,windTurbines,tur_nr,maxiter,sgd_thresh,min_spacing_m,**extra_vars):
    # make sure that extra_vars is only updated for the current run
    extra_vars = copy.deepcopy(extra_vars)
    # metrics_recorder, File
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
    
    # Generate initial layouts for each wind farm with the current seed
    coords = {name: poisson_disc_filler(tur_nr[name], min_dist=0.8*min_spacing_m, BorderC=boundaries[name], seed=seed)
              for name in list(dict.fromkeys(Sequence))}
    
    # Find the last index of each zone (for final cabling optimization)
    last_indices = {item: idx for idx, item in enumerate(Sequence)}
    
    # general options
    File += '_s' + str(seed)
    sample = True
    CableOpt = 'single_sub'
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
        
        # Options
        Sx = [Subs_x[curzone]]
        Sy = [Subs_y[curzone]]
        
        nb = Sequence[:i]
        nb = list(dict.fromkeys(nb[::-1]))[::-1]    # only keep latest ones in case of multiple entries
        nb = [x for x in nb if x not in curzone]    # kick out if current zone countained
        nnb = list(dict.fromkeys([x for x in Sequence if x not in nb and x not in curzone]))
        
        xn = []
        yn = []
        cable_cost_n = [0,0]
        mp_cost_n = [0,0]
        if nb:
            nf = True
            xn = np.array([])
            yn = np.array([])
            for j, zone in enumerate(nb):
                xn = np.concatenate([xn, metrics_recorder['x_' + zone][-1]])
                yn = np.concatenate([yn, metrics_recorder['y_' + zone][-1]])
                cable_cost_n[j] = metrics_recorder['cable_cost_' + zone][-1]
                mp_cost_n[j]= metrics_recorder['mp_cost_' + zone][-1]
        else:
            nf = False
              
        # Constraint
        boundary_constraint = CorrectedXYBoundaryConstraint([InclusionZone(boundaries[Sequence[i]])], boundary_type='multi_polygon')
        additional_turbines = [xn, yn]
        spacing_constraint = SpacingConstraintWithAdditionalTurbines(min_spacing_m, additional_turbines)
        aggregated_constraints = DistanceConstraintAggregationAdvanced(boundary_constraint=boundary_constraint, spacing_constraint=spacing_constraint, n_wt=tur_nr[Sequence[i]])
        
        # Plot or not
        if plot_iter:
            plt.figure()
            plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=True, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Subs_x, Sy=Subs_y, cables=cables_plot, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot, opt_nr=opt_nr, folder=plot_folder, sampling=sample, obj=obj, ploteach=plot_each)
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
        metrics_recorder['current_settings'].append({'seed':seed,'learning_rate':learning_rate,'curzone':curzone,'x0':x0,'y0':y0,'sample':sample,'CableOpt':CableOpt,'Sx':Sx,'Sy':Sy})
        
        # update kwargs
        extra_vars.update(xn=xn, yn=yn, Sx=Sx, Sy=Sy, curzone=curzone, nb=nb, nnb=nnb, cable_cost_n=cable_cost_n, mp_cost_n=mp_cost_n, opt_nr=opt_nr, nf=nf, sample=sample, time_limit=tl_opt, mip_gap=mip_gap_opt,
                          CableOpt=CableOpt, File=File, metrics_recorder=metrics_recorder, Sequence=Sequence, obj=obj, tur_nr=tur_nr, boundaries=boundaries, X_utm=X_utm, Y_utm=Y_utm, Z=Z, cables_plot=cables_plot, CableSolver=CableSolver_opt)

        # define cost and gradient function with handed over extra_vars
        cost_func = partial(lcoe_func, **extra_vars)
        cost_grad_func = partial(lcoe_jac, **extra_vars)
        
        # Optimization Setup
        tf = TopFarmProblem(
                design_vars = {'x':x0, 'y':y0},         
                cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=tur_nr[Sequence[i]], cost_function=cost_func, objective=True, cost_gradient_function=cost_grad_func, maximize=maximize),
                constraints = aggregated_constraints, 
                driver = EasySGDDriver(maxiter=maxiter, learning_rate=learning_rate, speedupSGD=True, sgd_thresh=sgd_thresh),
                plot_comp = plot_comp)
        
        # Run
        tic = time.time()
        cost, state, recorder = tf.optimize()
        toc = time.time()
        print('Optimization with SGD took: {:.0f}s'.format(toc-tic), ' with a total constraint violation of ', recorder['sgd_constraint'][-1])
        
        # postprocess recorder
        recorder = recorder.recorder2list()[1]['driver_iteration_dict']
        recorder['sgd_constraint'] = np.array(recorder['sgd_constraint']).flatten()
        recorder['wtSeparationSquared'] = np.array(recorder['wtSeparationSquared'])
        recorder['boundaryDistances'] = np.array(recorder['boundaryDistances'])
        
        # Final cabling optimization (last time zone is optimized...)
        if i == last_indices[curzone]:
            print('Final cabling optimization...')
            tic = time.time()
            extra_vars.update(time_limit=tl_final, mip_gap=mip_gap_final, CableSolver=CableSolver_final)
            lcoe_func(np.array(metrics_recorder['x_'+curzone][-1]), np.array(metrics_recorder['y_'+curzone][-1]),**extra_vars)
            toc = time.time()
            print('Final cabling optimization took: {:.0f}s'.format(toc-tic))
            # copy sgd recorder values
            recorder['sgd_constraint'] = np.append(recorder['sgd_constraint'],recorder['sgd_constraint'][-1])
            recorder['wtSeparationSquared'] = np.vstack([recorder['wtSeparationSquared'],recorder['wtSeparationSquared'][-1]])
            recorder['boundaryDistances'] = np.vstack([recorder['boundaryDistances'],recorder['boundaryDistances'][-1]])
        
        # Store
        metrics_recorder = extra_vars['metrics_recorder']
        record_results_constraints(metrics_recorder, recorder, state, cost, min_spacing_m)

        # Save recorder to file
        with open("Results//" + File + ".pkl", "wb") as file:
            pickle.dump({"metrics_recorder": metrics_recorder, "state": state, "recorder": recorder}, file)
            
    # Postprocess for full wind rose
    if sample:
        print('Optimization finished.')
        print('Starting postprocessing...')
        extra_vars.update(Subs_x=Subs_x,Subs_y=Subs_y)
        metrics_recorder = postprocess_recorder(metrics_recorder,**extra_vars)
        # Save processed file
        with open("Results//" + File + "_processed.pkl", "wb") as file:
            pickle.dump({"metrics_recorder": metrics_recorder}, file)
            
    # Plot LCOE iterations
    plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=1,optfat=1,feas=1)
    
    # delete variables (to avoid memory accumulation problems)
    del seed, Sequence, boundaries, File, metrics_recorder, Subs_x, Subs_y, X_utm, Y_utm, Z, cables_plot, obj, plot_each, windTurbines, tur_nr, maxiter, sgd_thresh, min_spacing_m, extra_vars
    gc.collect()
    
#%% Manually start postprocessing
def evaluate_multiter():
    with open("Results\\" + File + ".pkl", "rb") as file:
        data = pickle.load(file)
    data = data['metrics_recorder']
    metrics_recorder = postprocess_recorder(data,**extra_vars)

    # Save processed file
    with open("Results\\" + File + "_processed.pkl", "wb") as file:
        pickle.dump({"metrics_recorder": metrics_recorder}, file)
    plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=1,optfat=1)
    plot_convergence(mr=metrics_recorder,item='aep',plotstr='AEP (GWh)',obj=0,overall=0,optfat=1)
    plot_convergence(mr=metrics_recorder,item='cable_cost',plotstr='Cable Cost (€)',obj=0,overall=0,optfat=1)
    plot_convergence(mr=metrics_recorder,item='mp_cost',plotstr='Monopile Cost (€)',obj=0,overall=0,optfat=1)
    
#%% Load metric_recorder and plot
def evaluate_recorder():
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
        font_dir = [r"font\Serif"]
        for font in font_manager.findSystemFonts(font_dir):
            font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = "CMU Serif"
            
        plot = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=True, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Subs_x, Sy=Subs_y, cables=cables_plot, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot, folder=plot_folder, sampling=sample, obj=obj, optimize=False, paper=True)
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
            top=0.999,
            bottom=0.083,
            left=0.0,
            right=0.925,
            hspace=0.2,
            wspace=0.2
        )
        fig = plt.gcf()
        fig.set_size_inches(16 / 2.54, fig.get_size_inches()[1])
        
        plt.gcf().savefig("Figures//FinalLayout.pdf", dpi=500, pad_inches=0)
        
        # plt.text(555000, 5842000, 'N', color='white', fontsize=22, ha='center', va='center')
        # plt.text(548000, 5833000, 'M', color='white', fontsize=22, ha='center', va='center')
        # plt.text(543000, 5823000, 'S', color='white', fontsize=22, ha='center', va='center')
        # plt.gcf().savefig("Bathymetry.pdf", pad_inches=0, dpi=500)
        
    plot_convergence(mr=metrics_recorder,item='lcoe',plotstr='LCOE (€/MWh)',obj=0,overall=1,optfat=1,feas=1)
    # plot_convergence(mr=metrics_recorder,item='aep',plotstr='AEP (GWh)',obj=0,overall=0,optfat=1)
    # plot_convergence(mr=metrics_recorder,item='cable_cost',plotstr='Cable Cost (€)',obj=0,overall=0,optfat=1)
    # plot_convergence(mr=metrics_recorder,item='mp_cost',plotstr='Monopile Cost (€)',obj=0,overall=0,optfat=1)
#%% Compare Cabling
# def CompareCabling():
#     Files = ["test3_heuristic_processed",
#              "test3_metaheuristic_processed",
#              "test3_ortools_processed",
#              "test3_gurobi_processed",
#              "test3_cplex_cont_processed"]
#     CabOpt = ["heuristic","metaheuristic","ortools","gurobi","cplex"]
    
#     fig, axs = plt.subplots(1, len(Sequence), figsize=(15, 4))
#     for i, zone in enumerate(Sequence):
#         ax = axs[i]
#         for f, File in enumerate(Files):
#         # File = "metric_recorder_cooperative_2D_6000it_processed"
#             # specify file you want to load
#             with open("C:\\Software\\IEA-Wind-2200-22-ROWP\\examples\\Results\\" + File + ".pkl", "rb") as file:
#                 data = pickle.load(file)
#             mr = data['metrics_recorder']
#             item = 'lcoe'
#             # item = 'cable_cost'
#             plotstr = 'LCOE (€/MWh)'
#             # plotstr = 'Cable Cost (€)'
#             overall = 0
#             optfat = 0
#             obj = 0
            
#             # plt.figure(figsize=(5, 3))
#             ax.plot(
#                 np.array(mr['iteration'])[np.array([z[0] for z in mr['cur_zone']]) == zone],
#                 np.array(mr[item + '_' + zone])[np.array([z[0] for z in mr['cur_zone']]) == zone],
#                 label=zone + '_' + CabOpt[f],
#                 linewidth=1
#             )
#             # plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_'+zone]], label=zone+'_' + CabOpt[f], linewidth = 1)
#             # plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_mid']], label='Mid_' + CabOpt[f], linewidth = 1)
#             # plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_south']], label='South_' + CabOpt[f], linewidth = 1)
#             if obj:
#                 plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item]], label='Overall', linewidth = 1)
#             if overall:
#                 plt.plot(np.array(mr['iteration']),[x if x!=0 else np.nan for x in mr[item+'_all']], label='Overall', linewidth = 1)
#             if optfat:
#                 for i in range(0,len(mr['lcoe_all'])-1):
#                     if i == 0 and f == len(Files)-1:
#                         leg = 'Current objective'
#                     else:
#                         leg = None
#                     if mr['opt_nr'][i] == mr['opt_nr'][i+1] and mr['cur_zone'][i][0] == 'mid':
#                         plt.plot(np.array(mr['iteration'][i:i+2])-1,
#                             np.array(mr[item+'_'+mr['cur_zone'][i][0]][i:i+2]), linestyle="--", linewidth = 0.8, color="black", label=leg)
#         FS = 9
#         ax.legend(fontsize=FS)
#         ax.grid()
#         ax.set_xlabel('Iteration',fontsize=FS)
#         ax.set_ylabel(plotstr,fontsize=FS)
#         ax.set_title(zone,fontsize=FS+2)
#         ax.tick_params(axis='both', labelsize=FS-1)
    
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
def evaluate_layout():
    # define
    with open(r"Results\comp_s75_processed.pkl","rb") as file:
        data = pickle.load(file)
    metrics_recorder=data['metrics_recorder']
    extra_vars['metrics_recorder'] = metrics_recorder
    curzone = 'north'
    x_eva = metrics_recorder['x_north'][-1]
    y_eva = metrics_recorder['y_north'][-1]
    xn = metrics_recorder['x_mid'][-1] + metrics_recorder['x_south'][-1]
    yn = metrics_recorder['y_mid'][-1] + metrics_recorder['y_south'][-1]
    nf = True
    # xn = []
    # yn = []
    nb = ['mid','south']
    nnb = []
    Sx = [Subs_x[curzone]]
    Sy = [Subs_y[curzone]]
    
    # run
    extra_vars.update(xn=xn, yn=yn, nf=nf, curzone=curzone, nb=nb, nnb=nnb, Sx=Sx, Sy=Sy, sample=False, opt_nr=1, CableOpt="single_sub", CableSolver=CableSolver_opt, time_limit=tl_opt, mip_gap=mip_gap_opt)
    res = lcoe_func(x_eva,y_eva,**extra_vars)
    
    # plot
    plot_folder = "Figures//FinalResult"
    plt.figure()
    boundplot = list(boundaries.values())
    plot = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=True, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Subs_x, Sy=Subs_y, cables=cables_plot, metrics_recorder=metrics_recorder, Xn=xn, Yn=yn, b=boundplot, folder=plot_folder, sampling=sample, obj=None, optimize=False, paper=True)
    inputs = {}
    inputs['x'] = x_eva
    inputs['y'] = y_eva
    plot.compute(inputs,[])
#%% enforce constraints after optimization and conduct proper cable optimization
def refine_opt_results(seed,*,Sequence,boundaries,File,metrics_recorder,Subs_x,Subs_y,X_utm,Y_utm,Z,cables_plot,obj,
                    plot_each,windTurbines,tur_nr,maxiter,sgd_thresh,min_spacing_m,**extra_vars):
    #
    # Load file
    try:
        File += '_s' + str(seed) + '_processed'
        with open("Results/" + File + ".pkl", "rb") as file:
            data = pickle.load(file)
        #
        # First, check if constraints are violated.
        # find last index of the respetively optimized zones
        mr = data['metrics_recorder']
        constr_violation = []
        cur_zone = [n[0] for n in mr["cur_zone"]]
        for zone in ['north','mid','south']:
            last_index = len(cur_zone) - 1 - cur_zone[::-1].index(zone)
            constr_violation.append(abs(mr["tur_dist_violation"][last_index]))
            constr_violation.append(abs(mr["bound_violation"][last_index]))
        
        if sum(constr_violation) > 0:
            # load data
            Sequence =  list(dict.fromkeys(mr['sequence']))
            
            # create new metrics_recorder
            metrics_recorder = create_recorder(Sequence)
            metrics_recorder['general_settings'].append(data['metrics_recorder']['general_settings'][-1])
            
            # general options
            maxiter = 3000
            sgd_thresh = 0.9999
            sample = True
            CableOpt = 'multi_sub'
            boundplot = list(boundaries.values())
            plot_folder = "Figures//" + File + "_refined"
            opt_nr = data['metrics_recorder']['opt_nr'][-1] + 1
            curzone = 'all'
            Sx = [Subs_x[zone] for zone in Sequence]
            Sy = [Subs_y[zone] for zone in Sequence]
            nf = False
            nb = []
            nnb = []
            cable_cost_n = [0,0]
            mp_cost_n = [0,0]
            learning_rate = windTurbines.diameter()*0.1
            seed = data['metrics_recorder']['current_settings'][-1]['seed']
            
            # Layout from foregoing optimization
            x0 = np.array([])
            y0 = np.array([])
            for zone in list(dict.fromkeys(data['metrics_recorder']['sequence'])):
                x0 = np.concatenate([x0, data['metrics_recorder']['x_' + zone][-1]])
                y0 = np.concatenate([y0, data['metrics_recorder']['y_' + zone][-1]])
        
            # Constraints
            boundary_constraint = MultiWFBoundaryConstraint(
                geometry = [boundaries[name] for name in Sequence],  # Boundary mapping
                wt_groups=[np.arange(sum(tur_nr[zone] for zone in Sequence[:i]),sum(tur_nr[zone] for zone in Sequence[:i+1])) for i in range(len(Sequence))],  # Turbine groups
                boundtype = BoundaryType.POLYGON
            )
            aggregated_constraints = DistanceConstraintAggregation(boundary_constraint, sum([tur_nr[zone] for zone in Sequence]), min_spacing_m, windTurbines)
        
            # Plot or not
            if plot_iter:
                plt.figure()
                plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=True, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Subs_x, Sy=Subs_y, cables=cables_plot, metrics_recorder=metrics_recorder, b=boundplot, folder=plot_folder, sampling=sample, obj=obj, ploteach=plot_each)
            else:
                plot_comp = None
            
            # Max or min
            if obj == 'lcoe':
                maximize = False
            elif obj == 'aep':
                maximize = True
        
            # record settings
            [metrics_recorder[key].append(None) for key in ["sgd_constraint_violation", "tur_dist_violation", "bound_violation"]]   # first run: no optimization
            metrics_recorder['current_settings'].append({'seed':seed,'learning_rate':learning_rate,'curzone':curzone,'x0':x0,'y0':y0,'sample':sample,'CableOpt':CableOpt,'Sx':Sx,'Sy':Sy})
            
            # update kwargs
            extra_vars.update(Sx=Sx, Sy=Sy, curzone=curzone, nb=nb, nnb=nnb, cable_cost_n=cable_cost_n, mp_cost_n=mp_cost_n, opt_nr=opt_nr, nf=nf, sample=sample, time_limit=tl_opt, mip_gap=mip_gap_opt,
                              CableOpt=CableOpt, File=File, metrics_recorder=metrics_recorder, Sequence=Sequence, obj=obj, tur_nr=tur_nr, boundaries=boundaries, X_utm=X_utm, Y_utm=Y_utm, Z=Z, cables_plot=cables_plot, CableSolver=CableSolver_opt)
            
            # define cost and gradient function with handed over extra_vars
            cost_func = partial(lcoe_func, **extra_vars)
            cost_grad_func = partial(lcoe_jac, **extra_vars)
            
            # Optimization setup
            tf = TopFarmProblem(
                    design_vars = {'x':x0, 'y':y0},         
                    cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=sum([tur_nr[zone] for zone in Sequence]), cost_function=cost_func, objective=True, cost_gradient_function=cost_grad_func, maximize=maximize),
                    constraints = aggregated_constraints, 
                    driver = EasySGDDriver(maxiter=maxiter, learning_rate=float(learning_rate), speedupSGD=True, sgd_thresh=sgd_thresh),
                    plot_comp = plot_comp
                    )
            
            # Run
            tic = time.time()
            cost, state, recorder = tf.optimize()
            toc = time.time()
            print('Optimization with SGD took: {:.0f}s'.format(toc-tic), ' with a total constraint violation of ', recorder['sgd_constraint'][-1])
            
            # postprocess recorder
            recorder = recorder.recorder2list()[1]['driver_iteration_dict']
            recorder['sgd_constraint'] = np.array(recorder['sgd_constraint']).flatten()
            recorder['wtSeparationSquared'] = np.array(recorder['wtSeparationSquared'])
            recorder['boundaryDistances'] = np.array(recorder['boundaryDistances'])
            
            # Final cabling optimization
            print('Final cabling optimization...')
            tic = time.time()
            extra_vars.update(time_limit=tl_final, mip_gap=mip_gap_final, CableSolver=CableSolver_final)
            lcoe_func(state['x'], state['y'], **extra_vars)
            toc = time.time()
            print('Final cabling optimization took: {:.0f}s'.format(toc-tic))
            # copy sgd recorder values
            recorder['sgd_constraint'] = np.append(recorder['sgd_constraint'],recorder['sgd_constraint'][-1])
            recorder['wtSeparationSquared'] = np.vstack([recorder['wtSeparationSquared'],recorder['wtSeparationSquared'][-1]])
            recorder['boundaryDistances'] = np.vstack([recorder['boundaryDistances'],recorder['boundaryDistances'][-1]])
            
            # Store
            metrics_recorder = extra_vars['metrics_recorder']
            record_results_constraints(metrics_recorder, recorder, state, cost, min_spacing_m)
            
            # Save to a file
            with open("Results//" + File + "_refined_raw.pkl", "wb") as file:
                pickle.dump({"metrics_recorder": metrics_recorder, "state": state, "recorder": recorder}, file)
                
            # Postprocess for full wind rose
            if sample:
                print('Optimization finished.')
                print('Starting postprocessing...')
                extra_vars.update(Subs_x=Subs_x,Subs_y=Subs_y)
                metrics_recorder = postprocess_recorder(metrics_recorder,**extra_vars)
                # Save processed file
                with open("Results//" + File + "_refined.pkl", "wb") as file:
                    pickle.dump({"metrics_recorder": metrics_recorder}, file)
        else:
            print('Constraints for seed ' + str(seed) + ' not violated.')
    except Exception:
        print('Results for seed ' + str(seed) + ' do not exist.')
        

#%% final seed evaluation
def evaluate_seeds():
    from collections import defaultdict
    results = defaultdict(list)
    obj = "lcoe"
    objplot = "LCOE"
    scaler = 1
    for s in seeds:
        # first check if results for the current seed exist
        try:
            try:
                # load the finetuned recorder (with enforced constraints) if it exists
                with open(f"Results/comp_s{s}_processed_refined.pkl", "rb") as file:
                    data = pickle.load(file)
            except:
                # otherwise, load the original file
                # (constraints have not been violated for the final design or not yet been finetuned)
                with open(f"Results/comp_s{s}_processed.pkl", "rb") as file:
                    data = pickle.load(file)
            mr = data["metrics_recorder"]
            # store main metrics
            results["seed"].append(int(s))
            results[obj+"_north"].append(float(mr[obj+"_north"][-1])*scaler)
            results[obj+"_mid"].append(float(mr[obj+"_mid"][-1])*scaler)
            results[obj+"_south"].append(float(mr[obj+"_south"][-1])*scaler)
            results[obj+"_all"].append(float(mr[obj+"_all"][-1])*scaler)
    
            # find last index of the respetively optimized zones
            cur_zone = [n[0] for n in mr["cur_zone"]]
            for zone in ['north','mid','south','all']:
                if zone in cur_zone:
                    last_index = len(cur_zone) - 1 - cur_zone[::-1].index(zone)
                    results["dc_" + zone].append(mr["tur_dist_violation"][last_index])
                    results["bc" + zone].append(mr["bound_violation"][last_index])
        except:
            results["failed_seed"].append(int(s))
    
    ConstraintFailed = 0
    for idx in range(len(results['bcmid'])-1,-1,-1):
        cur_constraints = []
        for zone in ['north','mid','south']:
            cur_constraints.append(abs(results["dc_" + zone][idx]))
            cur_constraints.append(abs(results["bc" + zone][idx]))
        if sum(cur_constraints) > 0:
            results["violated_seeds"].append(int(results['seed'][idx]))
            for key, lst in results.items():
                if key != "failed_seed" and key != "violated_seeds":
                    lst.pop(idx)
            ConstraintFailed += 1

    #%% Plot
    from matplotlib.ticker import ScalarFormatter
    
    # Extract lists from results
    lcoe_north = results[obj+"_north"]
    lcoe_mid   = results[obj+"_mid"]
    lcoe_south = results[obj+"_south"]
    lcoe_all   = results[obj+"_all"]
    
    all_lcoes = [lcoe_north, lcoe_mid, lcoe_south, lcoe_all]
    labels = [objplot+" North", objplot+" Mid", objplot+" South", objplot+" All"]
    colors = ["skyblue", "lightgreen", "orange", "lightgray"]
    
    # Find indices of minima for each metric
    min_indices = [[i for i, v in enumerate(arr) if v == min(arr)] for arr in all_lcoes]
    
    # Collect marker values
    marker_values = []
    for idx_list in min_indices:
        scenario_values = [[arr[i] for i in idx_list] for arr in all_lcoes]
        marker_values.append(scenario_values)
    
    print("Seeds corresponding to minima:\n")
    
    for source_label, indices in zip(labels, min_indices):
        print(f"Minima source: {source_label}")
        for idx in indices:
            seed_num = results["seed"][idx]  # actual seed number
            seed_values = [arr[idx] for arr in all_lcoes]  # LCOE values for all metrics at this seed
            print(f"  Seed {seed_num}: LCOE North={seed_values[0]:.2f}, "
                  f"Mid={seed_values[1]:.2f}, South={seed_values[2]:.2f}, All={seed_values[3]:.2f}")

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.flatten()
    
    for ax, values, label, color, metric_markers in zip(axes, all_lcoes, labels, colors, zip(*marker_values)):
        counts, bins, patches = ax.hist(values, bins=20, alpha=0.7, color=color, edgecolor='black')
        ax.set_title(label)
        if obj == 'lcoe':
            ax.set_xlabel("LCOE [$/MWh]")
        elif obj == 'aep':
            ax.set_xlabel("AEP [GWh]")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle='--', alpha=0.6)
    
        # Force plain numbers on x-axis
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
    
        # Add markers
        y_marker = max(counts) * 0.035
        for mvals, mcolor in zip(metric_markers, colors):
            ax.scatter(
                mvals,
                [y_marker]*len(mvals),
                color=mcolor,
                edgecolor='black',
                linewidth=0.8,
                marker='v',
                s=90,
                zorder=5
            )
    
    plt.tight_layout()
    plt.show()

    return results
            
#%% correct recorder
def correct_recorder():
    from collections import defaultdict
    results = defaultdict(list)
    for s in seeds:
        try:
            with open(f"Results/comp_s{s}.pkl", "rb") as file:
                data = pickle.load(file)
            mr = data["metrics_recorder"]
    
            # target keys to modify
            keys_to_clean = ['bound_violation', 'sgd_constraint_violation', 'tur_dist_violation']
            
            # find all indices to remove (last occurrence of 1–9)
            indices_to_remove = []
            for val in range(1, 10):  # only 1 to 9
                if val in mr['opt_nr']:
                    last_idx = len(mr['opt_nr']) - 1 - mr['opt_nr'][::-1].index(val) + 1
                    indices_to_remove.append(last_idx)
            
            # sort indices in descending order so deletion doesn’t shift earlier indices
            indices_to_remove.sort(reverse=True)
            
            # delete entries from all lists
            for idx in indices_to_remove:
                for key in keys_to_clean:
                    del mr[key][idx]
           
            # save corrected file
            with open(f"Results/comp_s{s}.pkl", "wb") as file:
                pickle.dump(data,file)
            
            # open processed file
            with open(f"Results/comp_s{s}_processed.pkl", "rb") as file:
                data2 = pickle.load(file)
                
            idxs = data2['metrics_recorder']['iteration']
            data2['metrics_recorder']['sgd_constraint_violation'] = [data['metrics_recorder']['sgd_constraint_violation'][i] for i in idxs]
            data2['metrics_recorder']['bound_violation'] = [data['metrics_recorder']['bound_violation'][i] for i in idxs]
            data2['metrics_recorder']['tur_dist_violation'] = [data['metrics_recorder']['tur_dist_violation'][i] for i in idxs]

            # save corrected file
            with open(f"Results/comp_s{s}_processed.pkl", "wb") as file:
                pickle.dump(data2,file)

        except:
            results["failed_seed"].append(int(s))
      
#%% Run
if __name__ == "__main__":
    if Mode == "evaluate_recorder":
        evaluate_recorder()
    elif Mode == "evaluate_seeds":
        results = evaluate_seeds()
    elif Mode == "evaluate_multiter":
        evaluate_multiter()
    elif Mode == "correct_recorder":
        correct_recorder()
    elif Mode == "evaluate_layout":
        evaluate_layout()
    elif Mode == "refine_opt_results":
        if len(seeds) == 1:
            refine_opt_results(seeds[0],**extra_vars)
        else:
            worker = partial(refine_opt_results, **extra_vars)
            with Pool(num_workers) as pool:
                pool.map(worker,seeds)
    elif Mode == "cooperative":
        if len(seeds) == 1:
            opt_cooperative(seeds[0],**extra_vars)
        else:
            worker = partial(opt_cooperative, **extra_vars)
            with Pool(num_workers) as pool:
                pool.map(worker,seeds)
    elif Mode == "competitive":
        if len(seeds) == 1:
            opt_competitive(seeds[0],**extra_vars)
        else:
            worker = partial(opt_competitive, **extra_vars)
            with Pool(num_workers) as pool:
                pool.map(worker,seeds)
    else:
        raise Exception("Mode not implemented yet.")