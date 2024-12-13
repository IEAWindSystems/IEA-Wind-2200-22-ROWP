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
from OptPlotBathy import XYPlotCompBathym
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint, InclusionZone, ExclusionZone
from py_wake.utils.gradients import fd, cs, autograd
from topfarm.recorders import TopFarmListRecorder
from topfarm.constraint_components.constraint_aggregation import ConstraintAggregation
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
Zone = 'All'            # 'North', 'Mid', 'South', 'All'
SepCabling = True       # Indicate if cabling is only allowed within each zone or cross-zonal
NeighbourFarm = 'None'  # Give respective Zone or set to 'None'
tur_nr = 34             # Desired turbine number in optimized farm
Model = 'gauss'
plot_convergence = True

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
#%% Process neighbour farm (coordinates from foregoing optimization)
if NeighbourFarm == 'Mid':
    if Model == 'jensen':
        x2 = np.array([541341,550836,544095,549679,552827,548301,547732,553573,544962,544294,551219,542278,547977,540371,546517,552340,549699,539471,543190,541690,553190,554045,551485,540404,550614,545651,546825,543045,546950,545879,554922,554306,549436,548470])
        y2 = np.array([5831010,5830930,5834300,5838020,5827720,5828080,5838680,5829020,5835360,5828400,5827850,5832120,5834340,5828710,5831600,5834640,5827970,5828780,5833220,5828610,5833580,5832500,5835730,5829880,5836840,5828290,5837590,5828510,5828190,5836440,5831380,5830300,5832470,5839560])
    elif Model == 'gauss':
        x2 = np.array([546233.40801446, 554913.3546076 , 549258.84180142, 547675.59711394,553003.20065231, 552822.05226814, 541881.46933462, 552389.84300364,548469.46646853, 553221.84909349, 551397.34606148, 543109.95792625,549736.30634869, 549087.45312486, 551241.13767395, 553973.94672575,547741.78653302, 549833.43855587, 547157.10892055, 547107.32391826,553688.63300953, 545306.4826107 , 544270.84935661, 541259.9214019 ,545304.89107516, 550218.89150858, 543151.52583301, 545699.42696366,550709.38240519, 539483.75008356, 544650.18513836, 550072.68331866,540964.7365428 , 548767.40372445])
        y2 = np.array([5833124.56209649, 5831383.91057012, 5828016.41798088, 5836248.15223335, 5830556.47456355, 5827724.81771515, 5831604.65742586, 5834561.48431823, 5839554.4955233 , 5833511.15981051, 5835619.0583126 , 5828505.90018199, 5829436.12249334, 5838762.31942038, 5827944.28601616, 5832433.79291792, 5831321.91185011, 5837729.29220668, 5837938.94766808, 5828184.69888873, 5829267.34471564, 5830029.16650908, 5831772.65262087, 5830143.88800132, 5828431.27179463, 5832911.42095095, 5833161.66122081, 5836209.39084711, 5836696.60054887, 5828800.72653631, 5834741.40168167, 5831215.09146241, 5828679.88819848, 5834457.20341153])
    nf = True
elif NeighbourFarm == 'North':
    if Model == 'jensen':
        x2 = [558697,551288,557376,557913,552741,558575,556733,555572,558155,551800,550335,554356,550826,558771,559155,554525,559579,558969,558366,556623,553612,549270,553257,553986,554154,557081,555019,555498,559122,559359,552287,558355,557670,549940]
        y2 = [5850940,5837960,5838180,5840990,5844660,5844430,5834790,5832510,5842240,5843530,5839170,5838690,5842380,5845480,5847490,5846810,5849710,5846480,5843360,5849310,5845710,5840510,5835470,5841400,5834320,5836630,5833220,5847970,5850340,5848560,5836690,5851410,5839720,5841310]
    elif Model == 'gauss':
        x2 = np.array([556137.81567704, 557909.1025308 , 557123.27886796, 552498.96511937,551934.43269332, 555562.08650912, 558361.48440039, 558976.03069405,556929.95907233, 556591.07708086, 554047.66779233, 551841.50243604,550711.57754798, 559526.76655738, 559193.07206787, 555800.15032453,557546.36966399, 553010.57570251, 554836.50886617, 558905.51560983,554888.26252476, 551097.95660925, 549288.42109679, 558576.87132594,555254.03460263, 554961.64789883, 553514.38055605, 558177.82935027,557192.42911719, 556824.9672117 , 556199.41501778, 554615.78353315,550074.47706368, 553162.17213733])
        y2 = np.array([5844973.8713822 , 5841076.98130545, 5837366.41937725,5840558.02613111, 5837173.41125418, 5832526.42645761,5851377.82162195, 5850496.41191597, 5835893.789062,5834339.20979233, 5834481.01493262, 5843552.38764934,5842000.29355378, 5849563.38637446, 5847850.52271652,5842999.84171197, 5839201.70500668, 5835823.29836516,5836984.88558822, 5846294.60518425, 5838701.72657067,5838326.4636966 , 5840486.43086775, 5844506.04522385,5840943.63473863, 5833354.67682881, 5843231.36099884,5842864.29059435, 5849966.48777579, 5847268.85374947,5848668.82834503, 5846876.35457277, 5839517.29192004,5845158.20709307])
    nf = True
elif NeighbourFarm == 'None':
    nf = False
else:
    raise ValueError('NeighbourFarm not defined correctly.')
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

# get initial x and y positions
#x0 = farm_dat['layouts']['initial_layout']['coordinates']['x']
#y0 = farm_dat['layouts']['initial_layout']['coordinates']['y']

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

# Substation coordinate
# if Zone == 'Mid':
# #     x0 = [546333.63474559,554920.10151423,549422.85587285,546136.74470996,551339.58169912,552852.58216252,542217.91962822,552563.93138148,548460.45635016,553679.48424965,551240.56269371,543914.55456359,548850.91117952,549236.3384757,551178.87826587,553422.54905415,548133.27423075,547090.46991575,547943.92712465,547786.48566117,553740.83040674,544228.99665126,545129.8804181,540455.63131987,546049.81628749,550456.72983694,548053.10952293,542196.78698654,550261.41313528,539482.53936678,543624.48270107,551115.36789015,541530.89116684,544675.83124295]
# #     y0 = [5830219.623481509,5831376.711578628,5828114.7643633755,5836695.036029441,5829932.731539264,5827771.895479895,5830249.439991704,5834368.310071459,5839538.126536711,5832958.970977972,5836058.456399757,5828427.682958245,5830686.145856599,5838598.578527443,5827873.383495137,5831044.530321654,5832643.9577727085,5835511.398954034,5837464.337685456,5828187.188584721,5829304.02112439,5830364.057144121,5832401.619418706,5829946.1247157445,5828254.626678088,5833963.653209071,5834407.431636803,5832041.984982793,5837285.254158805,5828782.918614267,5833555.560298287,5831972.252517432,5828629.793272737,5835006.545328568]
#     Sx = Subs_x[1]
#     Sy = Subs_y[1]
# if Zone == 'North':
# #     x0 = [555462.7815047 , 558192.75454615, 557516.95709109, 551829.25075218, 550906.82672596, 555600.47948374, 558294.64529255, 558932.63025182, 557077.83007341, 556702.56967423, 554081.4810287 , 553136.88029812, 550784.88483612, 559575.0531913 , 559265.67213045, 555157.10127519, 557837.84987992, 552320.39653102, 552444.73170613, 559052.32906201, 553313.24132169, 552566.69671044, 549261.01000065, 558779.94156623, 555127.47952396, 554903.43873529, 553800.68239313, 558477.01382161, 556843.9695099 , 555800.98347535, 556386.59751983, 555546.99726727, 550023.50032033, 554326.91460964]
# #     y0 = [5841417.144829278, 5842449.350646424, 5838857.727073366, 5843575.869298234, 5838572.66438408, 5832557.501989138, 5851325.744269247, 5850591.631111388, 5836630.666193126, 5834622.1146755405, 5834421.453206649, 5844866.258713254, 5842051.520705445, 5849682.703235188, 5848296.551732236, 5839556.5244619595, 5840686.231288356, 5837875.876761292, 5836553.9026119085, 5846923.457904408, 5835390.807303021, 5840743.531747452, 5840515.480496754, 5845500.254311497, 5837099.082503276, 5833366.378956759, 5843199.386602501, 5843950.981754125, 5849353.568481362, 5843556.2583834445, 5846208.534290497, 5848030.195665719, 5839581.2183083175, 5846580.890253602]
#     Sx = Subs_x[0]
#     Sy = Subs_y[0]
Sx = Subs_x
Sy = Subs_y
#%% Objective function
aep = None
dcable_cost = None
def lcoe_func(x, y, **kwargs):
    global metrics_recorder, aep, cable_cost, dcable_cost, mp_cost, dmp_cost
    #
    # 1.) aep
    wd = np.arange(0, 360, 1)
    if nf:
        aep = wake_model(x=np.concatenate((x,x2)), y=np.concatenate((y,y2)), wd=wd, ws=ws, TI=TI).aep().isel(wt=slice(0, 3 * tur_nr)).sum().values * 1e3
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
    lcoe = npv / np.sum(aep).item()
    #
    # 5. Record the missing metrics
    metrics_recorder["iteration"].append(kwargs.get("iteration", len(metrics_recorder["iteration"]) + 1))
    metrics_recorder["aep"].append(np.sum(aep).item())
    metrics_recorder["mp_cost"].append(sum(mp_cost))
    metrics_recorder["cable_cost"].append(cable_cost)
    metrics_recorder["lcoe"].append(lcoe)
    # performance of individual zones
    if SepCabling:
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
        metrics_recorder["cable_cost1"].append(cable_cost1)
        metrics_recorder["cable_cost2"].append(cable_cost2)
        metrics_recorder["cable_cost3"].append(cable_cost3)
        metrics_recorder["mp_cost1"].append(mp_cost1)
        metrics_recorder["mp_cost2"].append(mp_cost2)
        metrics_recorder["mp_cost3"].append(mp_cost3)
        metrics_recorder["lcoe1"].append(lcoe1)
        metrics_recorder["lcoe2"].append(lcoe2)
        metrics_recorder["lcoe3"].append(lcoe3)
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
    #if nf:
    #    daep = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=np.concatenate((x,x2)), y=np.concatenate((y,y2)), ws=ws, TI=TI, wd=wd)[:tur_nr,:tur_nr] * 1e3
    #    aep = wake_model(x=np.concatenate((x,x2)), y=np.concatenate((y,y2)), wd=wd, ws=ws, TI=TI).aep().isel(wt=slice(0, 3 * tur_nr)).sum().values * 1e3    # sum(axis=(1, 2))
    #else:
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

#%% Constraints
# Boundaries and exclusion zones
b = system_dat['site']
north_boundary = np.array([b['boundaries']['polygons'][0]['x'], b['boundaries']['polygons'][0]['y']]).T
mid_boundary = np.array([b['boundaries']['polygons'][1]['x'], b['boundaries']['polygons'][1]['y']]).T
south_boundary = np.array([b['boundaries']['polygons'][2]['x'], b['boundaries']['polygons'][2]['y']]).T

# Min spacing
min_spacing_m = 2 * windTurbines.diameter()  #minimum inter-turbine spacing in meters

#%% Initial layout
# random rectangular
# x0 = np.random.uniform(np.min(boundary[:,0]),np.max(boundary[:,0]),(tur_nr,1)).flatten().tolist()
# y0 = np.random.uniform(np.min(boundary[:,1]),np.max(boundary[:,1]),(tur_nr,1)).flatten().tolist()
#
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
x0 = np.append(np.array(points_n).T[0], np.append(np.array(points_m).T[0], np.array(points_s).T[0]))
y0 = np.append(np.array(points_n).T[1], np.append(np.array(points_m).T[1], np.array(points_s).T[1]))

joint_boundaries = MultiWFPolygonBoundaryConstraint({0: north_boundary, 1: mid_boundary, 2: south_boundary}, turbine_groups={0: np.arange(tur_nr), 1: np.arange(tur_nr, 2 * tur_nr), 2: np.arange(2 * tur_nr, 3 * tur_nr)})

#x_coords, y_coords = zip(*random_points)
#x0 = list(x_coords)
#y0 = list(y_coords)

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
    "mp_cost1": [],
    "mp_cost2": [],
    "mp_cost3": [],
    "cable_cost1": [],
    "cable_cost2": [],
    "cable_cost3": [],
    "lcoe1": [],
    "lcoe2": [],
    "lcoe3": []
}

#%% Optimization setup
tf = TopFarmProblem(
        design_vars = {'x':x0, 'y':y0},         
        cost_comp = CostModelComponent(input_keys=['x','y'], n_wt=3 * tur_nr, cost_function=lcoe_func, objective=True, cost_gradient_function=lcoe_jac, maximize=False),
        constraints = DistanceConstraintAggregation([SpacingConstraint(min_spacing_m), joint_boundaries], 3 * tur_nr, min_spacing_m, windTurbines), 
        #constraints = DistanceConstraintAggregation([SpacingConstraint(min_spacing_m), constraint_comp],tur_nr, min_spacing_m, windTurbines), 
        driver = EasySGDDriver(maxiter=3000, learning_rate=windTurbines.diameter(), max_time=1008000, gamma_min_factor=0.1, speedupSGD=True, sgd_thresh=0.12),
        plot_comp = XYPlotCompBathym(save_plot_per_iteration=True, plot_initial=False, memory=0, X=X_utm, Y=Y_utm, Z=Z, Sx=Sx, Sy=Sy, cables=cables, metrics_recorder=metrics_recorder),
       # expected_cost = 1
        )

# SmartStart (work in progress)
# x = np.linspace(np.min(boundary[0:]),np.max(boundary[0:]),20)
# y = np.linspace(np.min(boundary[1:]),np.max(boundary[1:]),20)
# YY, XX = np.meshgrid(y, x)
# tf.smart_start(XX, YY, tf.cost_comp.get_aep4smart_start())

#%% Run
tic = time.time()
cost, state, recorder = tf.optimize()
toc = time.time()
print('Optimization with SGD took: {:.0f}s'.format(toc-tic), ' with a total constraint violation of ', recorder['sgd_constraint'][-1])

#%% Plot history
if plot_convergence:
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
    # Save to a file
    import pickle
    with open("data.pkl", "wb") as file:
        pickle.dump(metrics_recorder, file)
