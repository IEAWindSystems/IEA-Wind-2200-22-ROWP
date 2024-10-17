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

def CalculateMass(RP, D, HTrans, HHub_Ratio, WaterDepth, WaveHeight, WavePeriod, WindSpeed):
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
    IP_item = 0
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

#%% Sample run
# Inputs
# rating = 10             # MW
# D = 198                 # m
# HH = 145                # m
# PlatformHeight = 10     # m
# WaterDepth = 33.77      # m
# SignificantWaveHeight = 2.52    # m
# SignificantWavePeriod = 5.45    # s
# V_ave = 9.924           # m/s
# # Call surrogate
# mass = CalculateMass(rating,D,PlatformHeight,HH/D,WaterDepth,SignificantWaveHeight,SignificantWavePeriod, V_ave)
# print(f'Monopile mass: {mass[0][0]:.1f} kg')
# print(f'Tower mass: {mass[1][0]:.1f} kg')