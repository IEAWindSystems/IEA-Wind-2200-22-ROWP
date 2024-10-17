import numpy as np
from py_wake.rotor_avg_models import RotorCenter
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os
import yaml
import utm
from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.utils.gradients import autograd
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake import NOJ, BastankhahGaussian
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasySGDDriver, EasyScipyOptimizeDriver
from topfarm.plotting import XYPlotComp, NoPlot
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint, InclusionZone, ExclusionZone
from topfarm.recorders import TopFarmListRecorder
from topfarm.constraint_components.constraint_aggregation import ConstraintAggregation
from topfarm.constraint_components.constraint_aggregation import DistanceConstraintAggregation
from py_wake.turbulence_models import CrespoHernandez
from windIO.utils.yml_utils import load_yaml
from ssms.CalculateMass import CalculateMass
from scipy.interpolate import RegularGridInterpolator
np.random.seed(2)
#
# -------------
# INPUTS
Zone = 'Mid'            # 'North', 'Mid', 'South', 'North+Mid'
NeighbourFarm = 'North' # Give respective Zone or set to 'None'
tur_nr = 34             # Desired turbine number in optimized farm
Model = 'gauss'
#
# -------------
#
# Process neighbour farm (coordinates from foregoing optimization)
if NeighbourFarm == 'Mid':
    if Model == 'jensen':
        x2 = np.array([541341,550836,544095,549679,552827,548301,547732,553573,544962,544294,551219,542278,547977,540371,546517,552340,549699,539471,543190,541690,553190,554045,551485,540404,550614,545651,546825,543045,546950,545879,554922,554306,549436,548470])
        y2 = np.array([5831010,5830930,5834300,5838020,5827720,5828080,5838680,5829020,5835360,5828400,5827850,5832120,5834340,5828710,5831600,5834640,5827970,5828780,5833220,5828610,5833580,5832500,5835730,5829880,5836840,5828290,5837590,5828510,5828190,5836440,5831380,5830300,5832470,5839560])
    elif Model == 'gauss':
        x2 = np.array([546333.63474559,554920.10151423,549422.85587285,546136.74470996,551339.58169912,552852.58216252,542217.91962822,552563.93138148,548460.45635016,553679.48424965,551240.56269371,543914.55456359,548850.91117952,549236.3384757,551178.87826587,553422.54905415,548133.27423075,547090.46991575,547943.92712465,547786.48566117,553740.83040674,544228.99665126,545129.8804181,540455.63131987,546049.81628749,550456.72983694,548053.10952293,542196.78698654,550261.41313528,539482.53936678,543624.48270107,551115.36789015,541530.89116684,544675.83124295])
        y2 = np.array([5830219.623481509,5831376.711578628,5828114.7643633755,5836695.036029441,5829932.731539264,5827771.895479895,5830249.439991704,5834368.310071459,5839538.126536711,5832958.970977972,5836058.456399757,5828427.682958245,5830686.145856599,5838598.578527443,5827873.383495137,5831044.530321654,5832643.9577727085,5835511.398954034,5837464.337685456,5828187.188584721,5829304.02112439,5830364.057144121,5832401.619418706,5829946.1247157445,5828254.626678088,5833963.653209071,5834407.431636803,5832041.984982793,5837285.254158805,5828782.918614267,5833555.560298287,5831972.252517432,5828629.793272737,5835006.545328568])
    nf = True
elif NeighbourFarm == 'North':
    if Model == 'jensen':
        x2 = [558697,551288,557376,557913,552741,558575,556733,555572,558155,551800,550335,554356,550826,558771,559155,554525,559579,558969,558366,556623,553612,549270,553257,553986,554154,557081,555019,555498,559122,559359,552287,558355,557670,549940]
        y2 = [5850940,5837960,5838180,5840990,5844660,5844430,5834790,5832510,5842240,5843530,5839170,5838690,5842380,5845480,5847490,5846810,5849710,5846480,5843360,5849310,5845710,5840510,5835470,5841400,5834320,5836630,5833220,5847970,5850340,5848560,5836690,5851410,5839720,5841310]
    elif Model == 'gauss':
        x2 = [555462.7815047 , 558192.75454615, 557516.95709109, 551829.25075218, 550906.82672596, 555600.47948374, 558294.64529255, 558932.63025182, 557077.83007341, 556702.56967423, 554081.4810287 , 553136.88029812, 550784.88483612, 559575.0531913 , 559265.67213045, 555157.10127519, 557837.84987992, 552320.39653102, 552444.73170613, 559052.32906201, 553313.24132169, 552566.69671044, 549261.01000065, 558779.94156623, 555127.47952396, 554903.43873529, 553800.68239313, 558477.01382161, 556843.9695099 , 555800.98347535, 556386.59751983, 555546.99726727, 550023.50032033, 554326.91460964]
        y2 = [5841417.144829278, 5842449.350646424, 5838857.727073366, 5843575.869298234, 5838572.66438408, 5832557.501989138, 5851325.744269247, 5850591.631111388, 5836630.666193126, 5834622.1146755405, 5834421.453206649, 5844866.258713254, 5842051.520705445, 5849682.703235188, 5848296.551732236, 5839556.5244619595, 5840686.231288356, 5837875.876761292, 5836553.9026119085, 5846923.457904408, 5835390.807303021, 5840743.531747452, 5840515.480496754, 5845500.254311497, 5837099.082503276, 5833366.378956759, 5843199.386602501, 5843950.981754125, 5849353.568481362, 5843556.2583834445, 5846208.534290497, 5848030.195665719, 5839581.2183083175, 5846580.890253602]
    nf = True
elif NeighbourFarm == 'None':
    nf = False
else:
    raise ValueError('NeighbourFarm not defined correctly.')
#
# ----------------------------------------------------------------------
#
# system_dat = sys.argv[1]
import os
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

# get initial x and y positions
x0 = farm_dat['layouts']['initial_layout']['coordinates']['x']
y0 = farm_dat['layouts']['initial_layout']['coordinates']['y']

# define turbine
hh = farm_dat['turbines']['hub_height']
rd = farm_dat['turbines']['rotor_diameter']
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
freqs = site.local_wind(x0, y0, wd=dirs, ws=speeds).P_ilk[0, :, :]     # all frequencies

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
interpolator = RegularGridInterpolator((northing_values, easting_values), Z)
def depth_interp(x, y):
    return interpolator(np.array([y, x]).T)




# objective function and gradient function
samps = 50    #number of samples 
site.interp_method = 'linear'

# reduce to desired turbine nr
x0 = x0[0:tur_nr]
y0 = y0[0:tur_nr]

#function to create the random sampling of wind speed and wind directions
def sampling():
    idx = np.random.choice(np.arange(dirs.size), samps, p=np.sum(freqs,axis=1)/np.sum(freqs))
    wd = dirs[idx]
    ws = np.array([np.random.choice(speeds, p=freqs[i] / sum(freqs[i])) for i in idx])
    return wd, ws

#aep function - SGD
def aep_func(x, y, full=False, **kwargs):
    wd, ws = sampling()
    ti = np.array([TI]*len(ws))
    if nf:
        aep_sgd = wake_model(np.concatenate((x,x2)), np.concatenate((y,y2)), wd=wd, ws=ws, time=True, TI=ti).aep().isel(wt=slice(0, tur_nr)).sum().values * 1e6
    else:
        aep_sgd = wake_model(x, y, wd=wd, ws=ws, time=True, TI=ti).aep().sum().values * 1e6
    return aep_sgd

#lcoe function - SGD
#aep function - SLSQP
def lcoe_func(x, y, **kwargs):
    wd = np.arange(0, 360, 1)
    #ws = np.arange(3, 25, 1)
    aep = wake_model(x, y, wd=wd, ws=ws, TI=TI).aep().sum().values * 1e6
    # Inputs
    RP = 10              # MW
    D = 198              # m
    HH = 145             # m
    HTrans = 10          # m
    WaveHeight = 2.52    # m
    WavePeriod = 5.45    # s
    WindSpeed = 9.924    # m/s ToDo: verfiy it is average wind speed
    # Calculate Water Depth for current x/y coordinates
    



    
    
    WaterDepth = 33.77   # m
    # Call surrogate
    depths = depth_interp(x, y)
    masses = []
    for water_depth in depths:
       mass = CalculateMass(RP=RP, D=D, HTrans=HTrans, HHub_Ratio=HH/D, WaterDepth=water_depth, WaveHeight=WaveHeight, WavePeriod=WavePeriod, WindSpeed=WindSpeed)
       masses.append(np.sum(mass))
    # todo: interpoalte depth for turbine-specific masses
    
    return aep

lcoe_func(x0, y0)
#gradient function - SGD
def aep_jac(x, y, **kwargs):
    wd, ws = sampling()
    ti = np.array([TI]*len(ws))
    if nf:
        jx, jy = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=np.concatenate((x,x2)), y=np.concatenate((y,y2)), ws=ws, TI=ti, wd=wd, time=True)
        daep_sgd = np.array([np.atleast_2d(jx), np.atleast_2d(jy)])[:,:,:tur_nr] * 1e6
    else:
        jx, jy = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=x, y=y, ws=ws, TI=ti, wd=wd, time=True)
        daep_sgd = np.array([np.atleast_2d(jx), np.atleast_2d(jy)]) * 1e6
    return daep_sgd

#aep function - SLSQP
def aep_func2(x, y, **kwargs):
    wd = np.arange(0, 360, 1)
    #ws = np.arange(3, 25, 1)
    aep_slsqp = wake_model(x, y, wd=wd, ws=ws, TI=TI).aep().sum().values * 1e6
    return aep_slsqp

#gradient function - SLSQP
def aep_jac2(x, y, **kwargs):
    wd = np.arange(0, 360, 1)
   # ws = np.arange(3, 25, 1)
    jx, jy = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=x, y=y, TI=TI, wd=wd, time=False)
    daep_slsqp = np.array([np.atleast_2d(jx), np.atleast_2d(jy)]) * 1e6
    return daep_slsqp

n_wt = len(x0)
b = system_dat['site']
if Zone == 'North':
    boundary = np.array([b['boundaries']['polygons'][0]['x'], b['boundaries']['polygons'][0]['y']]).T
elif Zone == 'Mid':
    boundary = np.array([b['boundaries']['polygons'][1]['x'], b['boundaries']['polygons'][1]['y']]).T
elif Zone == 'South':
    boundary = np.array([b['boundaries']['polygons'][2]['x'], b['boundaries']['polygons'][2]['y']]).T
elif Zone == 'North+Mid':
    boundary1 = np.array([b['boundaries']['polygons'][0]['x'], b['boundaries']['polygons'][0]['y']]).T
    boundary2 = np.array([b['boundaries']['polygons'][1]['x'], b['boundaries']['polygons'][1]['y']]).T
    boundary = np.vstack((boundary1,boundary2[1],boundary2[0],boundary2[-1],boundary2[2]))
    exzone = [np.vstack((boundary1[0],boundary1[-1],boundary2[1:3]))]

#aep component - SGD
aep_comp = CostModelComponent(input_keys=['x','y'], n_wt=n_wt, cost_function=aep_func, objective=True, cost_gradient_function=aep_jac, maximize=True)

#aep component - SLSQP
aep_comp2 = CostModelComponent(input_keys=['x','y'], n_wt=n_wt, cost_function=aep_func2, objective=True, cost_gradient_function=aep_jac2, maximize=True)

cost_comps = [aep_comp2, aep_comp]

min_spacing_m = 2 * windTurbines.diameter()  #minimum inter-turbine spacing in meters
if 'exzone' in locals():
    constraint_comp = XYBoundaryConstraint([InclusionZone(boundary), ExclusionZone(exzone[0])], 'multi_polygon')
else:
    constraint_comp = XYBoundaryConstraint([InclusionZone(boundary)], 'multi_polygon')
    

#constraints
constraints = [[SpacingConstraint(min_spacing_m), constraint_comp],
               DistanceConstraintAggregation([SpacingConstraint(min_spacing_m), constraint_comp],n_wt, min_spacing_m, windTurbines)]

#driver specs
driver_names = ['SLSQP', 'SGD_again']
drivers = [EasyScipyOptimizeDriver(maxiter=200, tol=1e-3),
           EasySGDDriver(maxiter=10000, learning_rate=windTurbines.diameter(), max_time=1008000, gamma_min_factor=0.1, speedupSGD=True, sgd_thresh=0.05)]

driver_no = 1    #SGD driver
ec = [10,1]      #expected cost for SLSQP (10) and SGD (1) drivers

tf = TopFarmProblem(
        design_vars = {'x':x0, 'y':y0},         
        cost_comp = cost_comps[driver_no],    
        constraints = constraints[driver_no], 
        driver = drivers[driver_no],
        plot_comp = NoPlot(),
        expected_cost = ec[driver_no]
        )

if 1:
    tic = time.time()
    cost, state, recorder = tf.optimize()
    toc = time.time()
    print('Optimization with SGD took: {:.0f}s'.format(toc-tic), ' with a total constraint violation of ', recorder['sgd_constraint'][-1])
    recorder.save(f'{driver_names[driver_no]}')

