# -*- coding: utf-8 -*-
"""
Description: Simple exemplary script to load windIO data and run Floris wind farm flow analysis
Author: Samuel Kainz
Date: 23/04/2026
"""

#%% Preamble
import numpy as np
import windIO
from pathlib import Path
import os
from floris import FlorisModel, WindRose
from floris.turbine_library import build_cosine_loss_turbine_dict

#%% Inputs
Model = 'jensen'        # Wake model, pick from: 'jensen','gauss2016', 'turbopark'
Farm = 'north'          # Farm to evaluate, pick from: 'all', 'north', 'mid', 'south'. Note: neighbors not considered for individual plants!
wd_step = 1             # wind direction discretization in [deg]
ws_step = 1             # wind speed discretization in [m/s]

#%% Load windio data
system_dat = windIO.load_yaml(Path(os.sep.join(['..', 'data', 'wind_energy_system.yaml'])))

# site data
A = system_dat['site']['energy_resource']['wind_resource']['weibull_a']['data']
k = system_dat['site']['energy_resource']['wind_resource']['weibull_k']['data']
freq = system_dat['site']['energy_resource']['wind_resource']['sector_probability']['data']
wd_wb = system_dat['site']['energy_resource']['wind_resource']['wind_direction']
TI_org =  system_dat['site']['energy_resource']['wind_resource']['turbulence_intensity']['data']
ws_TI = system_dat['site']['energy_resource']['wind_resource']['wind_speed']

# turbines
hh = system_dat['wind_farm'][0]['turbines']['hub_height']
rd = system_dat['wind_farm'][0]['turbines']['rotor_diameter']
rp = system_dat['wind_farm'][0]['turbines']['performance']['rated_power']
cut_in = system_dat['wind_farm'][0]['turbines']['performance']['cutin_wind_speed']
cut_out = system_dat['wind_farm'][0]['turbines']['performance']['cutout_wind_speed']
p = system_dat['wind_farm'][0]['turbines']['performance']['power_curve']['power_values']
p_ws = system_dat['wind_farm'][0]['turbines']['performance']['power_curve']['power_wind_speeds']
ct = system_dat['wind_farm'][0]['turbines']['performance']['Ct_curve']['Ct_values']
ct_ws = system_dat['wind_farm'][0]['turbines']['performance']['Ct_curve']['Ct_wind_speeds']

# layouts
xs = {'north':  system_dat['wind_farm'][0]['layouts'][0]['coordinates']['x'],
      'mid':    system_dat['wind_farm'][1]['layouts'][0]['coordinates']['x'],
      'south':  system_dat['wind_farm'][2]['layouts'][0]['coordinates']['x']
      }
ys = {'north':  system_dat['wind_farm'][0]['layouts'][0]['coordinates']['y'],
      'mid':    system_dat['wind_farm'][1]['layouts'][0]['coordinates']['y'],
      'south':  system_dat['wind_farm'][2]['layouts'][0]['coordinates']['y']
      }

#%% Prepare wind farm simulation
# layout selection
if Farm == 'all':
    x = [item for sublist in xs.values() for item in sublist]
    y = [item for sublist in ys.values() for item in sublist]
else:
    x = xs[Farm]
    y = ys[Farm]

# create new turbine
turbine_data_dict = {
    "wind_speed": [0,p_ws[0]-0.01] + p_ws + [p_ws[-1]+0.01,50],
    "power": [0]*2 + [x/1e3 for x in p] + [0]*2,
    "thrust_coefficient": [0]*2 + ct + [0]*2,
}
turbine_dict = build_cosine_loss_turbine_dict(
    turbine_data_dict,
    "iea22MW",
    file_name=None,
    generator_efficiency=1,
    hub_height=hh,
    cosine_loss_exponent_yaw=1.88,
    cosine_loss_exponent_tilt=1.88,
    rotor_diameter=rd,
    TSR=9.15,
    ref_air_density=1.225,
    ref_tilt=6,
)

# Discretize Weibull for WindRose floris object
# a) Format and reshape
wb_scale = np.reshape(np.array(A),(-1,1))
wb_shape = np.reshape(np.array(k),(-1,1))
wb_wd_freq = np.reshape(np.array(freq),(-1,1))
# b) Wind speed discretization
wb_ws = np.arange(0, 51, ws_step)     # make sure full frequency spectrum is covered
# c) Upper and lower boundaries of wind speed bins
ws_low = np.arange(np.min(wb_ws)-ws_step/2,np.max(wb_ws)+ws_step/2,ws_step)
ws_high = ws_low + ws_step
ws_low[ws_low<0] = 0
ws_high[ws_high<0] = 0
# d) Discretize distribution for each wind direction and store in list (Weibull CDF)
freq_grid_raw = wb_wd_freq * ((1 - np.exp(-(1 / wb_scale * ws_high) ** wb_shape)) -
              (1 - np.exp(-(1 / wb_scale * ws_low) ** wb_shape)))

TI_wb_ws = np.array(TI_org)[np.clip(np.searchsorted(ws_TI, wb_ws), 0, len(TI_org)-1)]
TI_grid = np.tile(TI_wb_ws,wb_scale.shape)

# create windrose object
wind_rose = WindRose(
    wind_directions=np.array(wd_wb),
    wind_speeds=wb_ws,
    ti_table=TI_grid,
    freq_table=freq_grid_raw,
)
wind_rose = wind_rose.upsample(wd_step=wd_step, ws_step=ws_step)

# load input file and update settings
if Model == 'turbopark':
    fmodel = FlorisModel(r'subscripts\Floris4_turboparkgauss.yaml')
elif Model == 'jensen':
    fmodel = FlorisModel(r'subscripts\Floris4_jensen.yaml')
elif Model == 'gauss2016':
    fmodel = FlorisModel(r'subscripts\Floris4_gch.yaml')

# update settings
fmodel.set(
    layout_x=x,
    layout_y=y,
    wind_data = wind_rose,
    turbine_type=[turbine_dict],
    reference_wind_height=hh
)

#%% Run
fmodel.run()
aep = fmodel.get_farm_AEP()/1e9
fmodel.run_no_wake()
aep_nowake = fmodel.get_farm_AEP()/1e9
print(f"AEP = {aep:.2f} GWh")
print(f"Wake losses = {(1 - aep / aep_nowake) * 100:.2f}%")
print('Capcacity factor = %.3f' % ( aep / (len(x) * 22e6 * 8760 / 1e9)))