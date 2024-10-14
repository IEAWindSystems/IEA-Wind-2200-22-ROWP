# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:00:22 2022

@author: mikf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openturns as ot
from sklearn.preprocessing import MinMaxScaler
import os
import pickle


def trainQLS():
    plt.close('all')

    data = pd.read_csv('data/tower_mass_results.dat', sep=' ', )
    data_extended = pd.read_csv('data/tower_mass_results_extended_depth_results.dat', sep=' ', )
    df = data[data.columns[:-1]]
    df_extended = data_extended[data_extended.columns[:-1]]
    df.columns = data.columns[1:]
    df_extended.columns = data_extended.columns[1:]
    df = pd.concat([df, df_extended])

    in_cols = ['RP', 'D', 'HTrans', 'HHub_Ratio',
               'WaterDepth', 'WaveHeight', 'WavePeriod', 'WindSpeed']
    short_in = ['RP', 'D', 'HT', 'HHR',
                'WD', 'WH', 'WP', 'WS']
    out_cols = ['monopile_mass', 'tower_mass', 'total_mass']
    df.reset_index(drop=True, inplace=True)
    inp = df[in_cols]
    out = df[out_cols]
    count = out.shape[0]

    model_path = 'models/QLS'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    name_map = {x: x for x in list(df)}

    def train_model(df):

        input_db = df[in_cols]
        output_db = df[out_cols]

        # # Input and output names.
        input_channel_names = input_db.columns.to_list()
        output_channel_names = output_db.columns.to_list()

        # Numpy versions of the input and output dataset.
        input_dataset = input_db.to_numpy()
        output_dataset = output_db.to_numpy()
        n_output = output_dataset.shape[1]

        # # %% Center and scale the input and output dataset.

        # Center and scale the input dataset.
        input_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        input_dataset_scaled = input_scaler.fit_transform(input_dataset)

        # Center and scale the output dataset.
        output_scalers = {}
        output_dataset_scaled = np.empty_like(output_dataset)
        for i in range(len(output_channel_names)):
            output_channel_name = output_channel_names[i]
            output_scalers[output_channel_name] = MinMaxScaler(
                feature_range=(-0.7, 0.7))
            output_dataset_scaled[:, i] = \
                output_scalers[output_channel_name].fit_transform(
                    output_dataset[:, [i]]).ravel()

        # Fit the model
        # Compose the names for the linear and quadratic dependencies.
        predicted_output = pd.DataFrame(columns=output_channel_names)
        names = []
        for i in range(len(input_channel_names)):
            for j in range(0, i + 1):
                names.append(name_map[input_channel_names[i]] + ' * ' + name_map[input_channel_names[j]])
        dependencies = pd.DataFrame(
            index=[name_map[input_channel_name] for input_channel_name in input_channel_names] + names,
            columns=[name_map[output_channel_name] for output_channel_name in output_channel_names])

        models = []
        coefficients = {}
        for i_output_channel in range(n_output):
            model = ot.QuadraticLeastSquares(input_dataset_scaled, output_dataset_scaled[:, [i_output_channel]])
            model.run()
            models.append(model)

            # Get linear and quadratic dependencies of output from input variables.
            constant = np.squeeze(np.array(model.getConstant()))
            linear = np.squeeze(np.array(model.getLinear()))
            quadratic_full = np.squeeze(np.array(model.getQuadratic()))
            coefficients[output_channel_names[i_output_channel]] = {'constant': constant,
                                                                    'linear': linear,
                                                                    'quadratic': quadratic_full,
                                                                    }

            quadratic = quadratic_full - np.diag(np.diag(quadratic_full) * 0.5)
            quadratic = quadratic[np.tril_indices_from(quadratic)]
            dependencies.iloc[:, i_output_channel] = np.concatenate((linear.ravel(), quadratic))
            output_channel_name = output_channel_names[i_output_channel]
            responseSurface = model.getMetaModel()
            scaled_output = responseSurface(input_dataset_scaled)
            out = output_scalers[output_channel_name].inverse_transform(scaled_output).ravel()
            predicted_output[output_channel_name] = out
            df[output_channel_name + '_fit'] = out
            df[output_channel_name + '_scaled'] = scaled_output
        return input_scaler, output_scalers, df, dependencies, models, coefficients, predicted_output, input_channel_names, output_channel_names

    for IP in df.IP.unique():
        input_scaler, output_scalers, df_res, dependencies, models, coefficients, predicted_output, input_channel_names, output_channel_names = train_model(df[df.IP == IP])
        path = os.path.join(model_path, f'{IP}_QLS_surrogate_model.pickle')
        dic = dict(input_scaler=input_scaler,
                   output_scalers=output_scalers,
                   df=df_res,
                   dependencies=dependencies,
                   models=models,
                   coefficients=coefficients,
                   predicted_output=predicted_output,
                   input_channel_names=input_channel_names,
                   output_channel_names=output_channel_names,
                   )
        with open(path, 'wb') as f:
            pickle.dump(dic, f)


def main():
    if __name__ == '__main__':
        pass


main()
