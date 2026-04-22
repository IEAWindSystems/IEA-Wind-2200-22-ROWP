# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 08:38:29 2023

@author: mikf
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from ssms.models import model_path


class QLSModel(object):
    def __init__(self, model, input_scaler, output_scaler):
        self.model, self.input_scaler, self.output_scaler = model.getMetaModel(), input_scaler, output_scaler

    def predict(self, RP, D, HTrans, HHub_Ratio, WaterDepth, WaveHeight, WavePeriod, WindSpeed):
        inps = np.asarray([RP, D, HTrans, HHub_Ratio, WaterDepth, WaveHeight, WavePeriod, WindSpeed]).T
        inps_scaled = self.input_scaler.transform(np.atleast_2d(inps))
        scaled_output = self.model(inps_scaled)
        output = self.output_scaler.inverse_transform(scaled_output).ravel()
        return output


def get_r2(data, prediction):
    residuals = data - prediction
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data - np.mean(data))**2)
    r2_power = 1 - (ss_res / ss_tot)
    return r2_power


def main():
    if __name__ == '__main__':
        plt.close('all')

        # load the surrogates
        surrogate_path = os.path.join(model_path, 'QLS')
        model_indicator = '_QLS_surrogate_model.pickle'

        files = []
        IPs = []
        for file in os.listdir(surrogate_path):
            if model_indicator in file:
                IP = float(file.split(model_indicator)[0])
                files.append(file)
                IPs.append(IP)

        IP_item = 0
        IP = IPs[IP_item]
        print(f'IP: {IP}MW')
        path = os.path.join(surrogate_path, files[IP_item])
        with open(path, 'rb') as f:
            dic = pickle.load(f)

        input_channel_names = dic['input_channel_names']
        output_channel_names = dic['output_channel_names']
        out_item = 1
        output_channel = output_channel_names[out_item]
        print(f'output channel: {output_channel}')
        df = dic['df']

        # plot simulated and predicted data
        plt.figure()
        r2 = get_r2(df[output_channel], df[output_channel + '_fit'])
        plt.plot(df[output_channel], df[output_channel + '_fit'], '.')
        plt.title(f'{output_channel} simulation vs QLS IP={IP} r2={r2:.3f}')

        # predict
        qlsm = QLSModel(dic['models'][out_item], dic['input_scaler'], dic['output_scalers'][output_channel])
        # float input
        res = qlsm.predict(RP=11, D=214, HTrans=13, HHub_Ratio=0.7, WaterDepth=32, WaveHeight=3.5, WavePeriod=6, WindSpeed=9)
        print(res)

        # array-like input
        res = qlsm.predict(RP=[11, 12], D=[214, 220], HTrans=[13, 14], HHub_Ratio=2 * [0.7], WaterDepth=2 * [32], WaveHeight=2 * [3.5], WavePeriod=2 * [6], WindSpeed=2 * [9])
        print(res)


main()
