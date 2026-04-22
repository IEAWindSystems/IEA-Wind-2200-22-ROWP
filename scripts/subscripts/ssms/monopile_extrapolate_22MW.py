#%% Monopile Mass Extrapolation
# - Routine to extrapolate monopile mass with varying water depth to the 22MW machine
# - Necessary since original mass surrogate only allows rated power up to 20MW
# - Creates look-up table (csv file) with mass vs. depth
#
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openturns as ot
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

def trainQLS(rp):
    print('Building pickel files for mass surrogate model...')
    data = pd.read_csv('data/tower_mass_results.dat', sep=' ', )
    data_extended = pd.read_csv('data/tower_mass_results_extended_depth_results.dat', sep=' ', )
    df = data[data.columns[:-1]]
    df_extended = data_extended[data_extended.columns[:-1]]
    df.columns = data.columns[1:]
    df_extended.columns = data_extended.columns[1:]
    df = pd.concat([df, df_extended])
    # filter out 20MW values --> mass surrogate only valid for that rated power!!
    # plt.figure()
    df = df[df['RP'] == rp]
    # plt.scatter(df_cur['WaterDepth'],df_cur['monopile_mass'],label=str(rp)+'MW')
    # plt.legend()
    
    in_cols = ['D', 'HTrans', 'HHub',
               'WaterDepth', 'WaveHeight', 'WavePeriod', 'WindSpeed']
    out_cols = ['monopile_mass', 'tower_mass', 'total_mass']
    df.reset_index(drop=True, inplace=True)
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
        print('...done.')


#%% Create LUT
from CalculateMass import CalculateMass
# varying quantities
depths = np.arange(19,37.1,0.5)
depths = np.arange(10,60.1,1)
rps = np.arange(16,21)

# fixed parameter
V_ave = 10.26158
WavePeriod = 6.75
WaveHeight = 1.5
hh = 170
PlatformHeight = 15
rd = 284

# Make LUT
tot_masses = {}
records = []
for rp in rps:
    trainQLS(rp)
    masses = [] # monopile
    tower_masses = []
    for z in depths:
       cur_mass = CalculateMass(D=rd, HTrans=PlatformHeight, HHub=hh, WaterDepth=z, WaveHeight=WaveHeight, WavePeriod=WavePeriod, WindSpeed=V_ave, IP_item=1)
       masses.append(cur_mass[0][0])
       tower_masses.append(cur_mass[1][0])
       records.append([rp, z, cur_mass[0][0], cur_mass[1][0]])  # Append [Power, Depth, Mass_monopile, Mass_tower]
    tot_masses[rp] = masses

# Create pandas DataFrame
df = pd.DataFrame(records, columns=["Power_MW", "Depth_m", "Mass_mp_kg", "Mass_tower_kg"])
df.sort_values(by=["Depth_m", "Power_MW"], inplace=True)
df.reset_index(drop=True, inplace=True)

#%% Extrapolate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

target_power = 22
predictions = []

poly = PolynomialFeatures(degree=2)

# Group by depth to fit separate regressions
for depth, group in df.groupby("Depth_m"):
    X = group["Power_MW"].values.reshape(-1, 1)
    y1 = group["Mass_mp_kg"].values
    y2 = group["Mass_tower_kg"].values
    
    # monopile
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X, y1)
    M_22_mp = model.predict([[target_power]])[0]
    y1_pred = model.predict(X)
    
    # tower
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X, y2)
    M_22_tower = model.predict([[target_power]])[0]
    y2_pred = model.predict(X)
    
    # model.fit(X_poly, y)
    # M_22 = model.predict(poly.transform([[target_power]]))[0]
    # y_pred = model.predict(X_poly)
    
    # Calculate R² for this fit
    r2_mp = r2_score(y1, y1_pred)
    r2_tower = r2_score(y2, y2_pred)
    
    predictions.append([target_power, depth, M_22_mp, r2_mp, M_22_tower, r2_tower])

# Create a DataFrame for the 22MW results and store
df_22 = pd.DataFrame(predictions, columns=["Power_MW", "Depth_m", "Mass_mp_kg", "R2", "Mass_tower_kg", "R2"])
df_22.to_csv("Mass_Monopile_22MW_extrapolated_detect.csv", index=False)
#
#%% Plot
if True:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # --- LEFT Y-AXIS: Monopile Mass Curves ---
    for rp, group in df.groupby("Power_MW"):
        ax1.plot(
            group["Depth_m"],
            group["Mass_kg"],
            linestyle="-",
            label=f"{rp} MW"
        )
    
    # Plot extrapolated 22 MW
    ax1.plot(
        df_22["Depth_m"],
        df_22["Mass_kg"],
        color="black",
        linestyle="--",
        linewidth=2,
        label="22 MW (Extrapolated)"
    )
    
    # Axis labels & formatting for left axis
    ax1.set_xlabel("Water Depth [m]", fontsize=12)
    ax1.set_ylabel("Monopile Mass [kg]", fontsize=12)
    ax1.set_xlim([min(depths), max(depths)])
    ax1.grid(True)
    
    # Add legend on left axis
    ax1.legend(title="Rated Power", loc="upper left")
    
    # --- RIGHT Y-AXIS: R² Values ---
    ax2 = ax1.twinx()
    ax2.plot(
        df_22["Depth_m"],
        df_22["R2"],
        color="red",
        marker="o",
        linestyle="--",
        linewidth=1.5,
        label="R²"
    )
    ax2.set_ylabel("R² of Linear Fit", fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim([0, 1.05])  # R² always between 0 and 1
    
    # Title & layout
    plt.title("Monopile Mass vs Water Depth with R² of Linear Fits", fontsize=14)
    fig.tight_layout()
    plt.show()