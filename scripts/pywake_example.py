# -*- coding: utf-8 -*-
"""
Description: Simple exemplary script to load windIO data and run pywake wind farm flow analysis
Author: Samuel Kainz
Date: 22/04/2026
"""

#%% Preamble
import numpy as np
import windIO
from pathlib import Path
import os
import xarray as xr

from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake import NOJ, Nygaard_2022
from py_wake.literature import Bastankhah_PorteAgel_2014, Niayifar_PorteAgel_2016
from py_wake.rotor_avg_models import RotorCenter
from py_wake.turbulence_models import CrespoHernandez

#%% Inputs
Model = 'jensen'        # Wake model, pick from: 'jensen', 'gauss2014', 'gauss2016', 'turbopark'
Farm = 'north'          # Farm to evaluate, pick from: 'all', 'north', 'mid', 'south'. Note: neighbors not considered for individual plants!
wd_step = 1             # wind direction discretization in [deg]
ws_step = 1             # wind speed discretization in [m/s]

#%% Load windio data
system_dat = windIO.load_yaml(Path(os.sep.join(['..', 'data', 'wind_energy_system.yaml'])))

# site data
A = system_dat['site']['energy_resource']['wind_resource']['weibull_a']
k = system_dat['site']['energy_resource']['wind_resource']['weibull_k']
freq = system_dat['site']['energy_resource']['wind_resource']['sector_probability']
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

# set up pywake turbines and site
windTurbines = WindTurbine(name=system_dat['wind_farm'][0]['turbines']['name'], diameter=rd, hub_height=hh, 
                      powerCtFunction=PowerCtTabular(p_ws, p, power_unit='W', ct=ct))
site = XRSite(
       ds=xr.Dataset(data_vars=
                        {'Sector_frequency': ('wd', freq['data']), 
                         'Weibull_A': ('wd', A['data']), 
                         'Weibull_k': ('wd', k['data']),
                         'TI': ('ws', [0.05]*2 + TI_org + [0.05]*2)   # add dummies to avoid interpolation errors during sampling
                         },
                      coords={'wd': wd_wb, 'ws': [0,ws_TI[0]-0.01] + ws_TI + [ws_TI[-1]+0.01,100]}))

# wind resource
wd = np.arange(0, 360, wd_step)             # wind directions
ws = np.arange(cut_in, cut_out+1, ws_step)  # wind speeds
TI = np.interp(ws, ws_TI, TI_org)           # turbulence intensities

# set up wake model
if Model == 'jensen':
    wake_model = NOJ(site, windTurbines, k=0.05, rotorAvgModel=RotorCenter())
elif Model == 'gauss2014':
    wake_model = Bastankhah_PorteAgel_2014(site, windTurbines, rotorAvgModel=RotorCenter(), turbulenceModel=CrespoHernandez(), k=0.0324555)
elif Model == 'gauss2016':
    wake_model = Niayifar_PorteAgel_2016(site, windTurbines, rotorAvgModel=RotorCenter(), turbulenceModel=CrespoHernandez(rotorAvgModel=RotorCenter()))
elif Model == 'turbopark':
    wake_model = Nygaard_2022(site, windTurbines)
    
#%% Run
res = wake_model(x=x, y=y, wd=wd, ws=ws, TI=TI)
aep = res.aep().sum().item()
aep_nowake = res.aep(with_wake_loss=False).sum().item()
print(f"AEP = {aep:.2f} GWh")
print(f"Wake losses = {(1 - aep / aep_nowake) * 100:.2f}%")
print('Capcacity factor = %.3f' % ( aep / (len(x) * 22e6 * 8760 / 1e9)))