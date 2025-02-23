# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:49:20 2023

@author: Samuel Kainz
"""
import os 
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class QLSModel(object):
    def __init__(self, model, input_scaler, output_scaler):
        self.model, self.input_scaler, self.output_scaler = model.getMetaModel(), input_scaler, output_scaler
        
    def predict(self, RP, D, HTrans, HHub_Ratio, WaterDepth, WaveHeight, WavePeriod, WindSpeed):
        inps = np.asarray([RP, D, HTrans, HHub_Ratio, WaterDepth, WaveHeight, WavePeriod, WindSpeed]).T
        inps_scaled = self.input_scaler.transform(np.atleast_2d(inps))
        scaled_output = self.model(inps_scaled)
        output = self.output_scaler.inverse_transform(scaled_output).ravel()
        return output

def CalculateMass(RP, D, HTrans, HHub_Ratio, WaterDepth, WaveHeight, WavePeriod, WindSpeed, IP_item):
    # load the surrogates
    model_path = 'ssms/models/QLS'
    model_indicator = '_QLS_surrogate_model.pickle'
    
    files = []
    IPs = []
    for file in os.listdir(model_path):
        if model_indicator in file:
            IP = float(file.split(model_indicator)[0])
            files.append(file)
            IPs.append(IP)
    IP = IPs[IP_item]
    path = os.path.join(model_path, files[IP_item])
    with open(path, 'rb') as f:
        dic = pickle.load(f)
    # define the outputs
    #input_channel_names = dic['input_channel_names']
    output_channel_names = dic['output_channel_names']
    res = []
    for i in range(2):
        out_item = i
        output_channel = output_channel_names[out_item]
        #print(output_channel)
        qlsm = QLSModel(dic['models'][out_item], dic['input_scaler'], dic['output_scalers'][output_channel])
        mass = qlsm.predict(RP, D, HTrans, HHub_Ratio, WaterDepth, WaveHeight, WavePeriod, WindSpeed)
        res.append(np.ndarray.tolist(mass))
    return res

# #%% Sample run
# # Inputs
# turstring = ['IEA-10MW','IEA-15MW','IEA-22MW']
# RP = [10, 15, 22]             # MW
# D = [198, 240, 284]                 # m
# HH = [119, 140, 170]                # m
# PlatformHeight = [10, 15, 15]     # m
# WaterDepth = [34]*3      # m
# SignificantWaveHeight = [2.52]*3    # m
# SignificantWavePeriod = [5.45]*3    # s
# V_ave = [9.924]*3           # m/s
# # Call surrogate
# mass = CalculateMass(RP=RP, D=D, HTrans=PlatformHeight, HHub_Ratio=[hh/d for hh, d in zip(HH,D)], WaterDepth=WaterDepth, WaveHeight=SignificantWaveHeight, WavePeriod=SignificantWavePeriod, WindSpeed=V_ave)
# print(f'Monopile mass: {mass[0][2]:.1f} kg')
# print(f'Tower mass: {mass[1][2]:.1f} kg')

# #%% Create water depth dependent vector
# masses = []
# depths = np.linspace(20,40,num=21)
# for z in depths:
#    #masses.append((water_depth))
#    cur_mass = CalculateMass(RP=RP, D=D, HTrans=PlatformHeight, HHub_Ratio=[hh/d for hh, d in zip(HH,D)], WaterDepth=[z]*len(D), WaveHeight=SignificantWaveHeight, WavePeriod=SignificantWavePeriod, WindSpeed=V_ave)
#    masses.append(cur_mass[0])
# masses_norm = np.array(masses) / np.array(masses)[np.array(depths)==34]

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.gca()
# for i in range(masses_norm.shape[1]):
#     ax.plot(depths, masses_norm[:,i], label=turstring[i])  # Set the label for each curve
# ax.legend()
# ax.grid(alpha=0.6)
# ax.set_ylabel('Rel. monopile mass wrt z=34m')
# ax.set_xlabel('Water depth z [m]')