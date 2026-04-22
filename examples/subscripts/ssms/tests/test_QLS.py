# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:43:04 2022

@author: mikf
"""
import numpy as np
import os
import pickle
import pytest

from ssms.models import model_path
from ssms.load_QLS import QLSModel, get_r2


def test_QLS():
    surrogate_path = os.path.join(model_path, 'QLS')
    model_indicator = '_QLS_surrogate_model.pickle'

    files = []
    IPs = []
    for file in os.listdir(surrogate_path):
        if model_indicator in file:
            IP = float(file.split(model_indicator)[0])
            files.append(file)
            IPs.append(IP)

    IP_item = 1
    IP = IPs[IP_item]
    path = os.path.join(surrogate_path, files[IP_item])
    with open(path, 'rb') as f:
        dic = pickle.load(f)

    input_channel_names = dic['input_channel_names']
    output_channel_names = dic['output_channel_names']
    out_item = 1
    output_channel = output_channel_names[out_item]

    # predict
    qlsm = QLSModel(dic['models'][out_item], dic['input_scaler'], dic['output_scalers'][output_channel])
    # float input
    res = qlsm.predict(RP=11, D=214, HTrans=13, HHub_Ratio=0.7, WaterDepth=32, WaveHeight=3.5, WavePeriod=6, WindSpeed=9)
    np.testing.assert_allclose(res, 642458.07408856)

    # array-like input
    res = qlsm.predict(RP=[11, 12], D=[214, 220], HTrans=[13, 14], HHub_Ratio=2 * [0.7], WaterDepth=2 * [32], WaveHeight=2 * [3.5], WavePeriod=2 * [6], WindSpeed=2 * [9])
    np.testing.assert_allclose(res, [642458.07408856, 684316.37584743])
