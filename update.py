import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

############################
# WEEKLY FUNCTIONS
############################
def preprocess_data(file_path, input_steps=168, output_steps=168, skiprows=7):
    data = pd.read_csv(file_path, skiprows=skiprows)
    if data.shape[1] == 4:
        data.columns = ["Timestamp", "Electricity Power (kW)", "Air Pressure Power (kW)", "Air Consumption (kW)"]
    elif data.shape[1] == 2:
        data.columns = ["Timestamp", "Electricity Power (kW)"]
        data["Air Pressure Power (kW)"] = 0
        data["Air Consumption (kW)"] = 0
    else:
        data = data.iloc[:, :4]
        data.columns = ["Timestamp", "Electricity Power (kW)", "Air Pressure Power (kW)", "Air Consumption (kW)"]
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data = data[data['Timestamp'].dt.minute == 0]
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
    data['Month'] = data['Timestamp'].dt.month
    data['Electricity Power (Lag1)'] = data['Electricity Power (kW)'].shift(1)
    data['Electricity Power (Lag2)'] = data['Electricity Power (kW)'].shift(2)
    data.dropna(inplace=True)
    features = ['Electricity Power (kW)', 'Air Pressure Power (kW)', 'Air Consumption (kW)',
                'Hour', 'DayOfWeek', 'Month', 'Electricity Power (Lag1)', 'Electricity Power (Lag2)']
    X_data = data[features].values
    y_data = data[['Electricity Power (kW)']].values
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(y_data)
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - input_steps - output_steps + 1):
        X_seq.append(X_scaled[i:i+input_steps])
        y_seq.append(y_scaled[i+input_steps:i+input_steps+output_steps].flatten())
    return np.array(X_seq), np.array(y_seq), scaler_y

def update_model(data_file, model_file='saved_model.h5', input_steps=168, output_steps=168):
    X_seq, y_seq, scaler_y = preprocess_data(data_file, input_steps, output_steps)
    n_features = X_seq.shape[2]
    if os.path.exists(model_file):
        model = load_model(model_file)
        print("Loaded existing weekly model from", model_file)
    else:
        raise FileNotFoundError(f"{model_file} not found.")
    X_finetune = X_seq[:-4]
    y_finetune = y_seq[:-4]
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='mean_squared_error', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_finetune, y_finetune, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stop])
    model.save(model_file)
    print("Updated weekly model saved as", model_file)
    return model, scaler_y, X_seq

def predict_next_month(model, scaler_y, X_seq, input_steps=168):
    predictions = {}
    n_features = X_seq.shape[2]
    for week_num in range(4):
        X_input = X_seq[-(4 - week_num)].reshape(1, input_steps, n_features)
        pred = model.predict(X_input)
        pred_rescaled = scaler_y.inverse_transform(pred)
        # Clamp negative values to zero
        pred_rescaled[pred_rescaled < 0] = 0
        predictions[f"week_{week_num+1}"] = pred_rescaled.flatten().tolist()
    return predictions

############################
# MONTHLY FUNCTIONS
############################
def preprocess_data_monthly_custom(file_path, input_steps=720, output_steps=720):
    try:
        data = pd.read_csv(file_path, skiprows=7)
    except Exception:
        data = pd.read_csv(file_path)
    if data.shape[1] == 4:
        data.columns = ["Timestamp", "Electricity Power (kW)", "Air Pressure Power (kW)", "Air Consumption (kW)"]
    elif data.shape[1] == 2:
        data.columns = ["Timestamp", "Electricity Power (kW)"]
        data["Air Pressure Power (kW)"] = 0
        data["Air Consumption (kW)"] = 0
    else:
        data = data.iloc[:, :4]
        data.columns = ["Timestamp", "Electricity Power (kW)", "Air Pressure Power (kW)", "Air Consumption (kW)"]
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data = data[data['Timestamp'].dt.minute == 0]
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
    data['Month'] = data['Timestamp'].dt.month
    data['Electricity Power (Lag1)'] = data['Electricity Power (kW)'].shift(1)
    data['Electricity Power (Lag2)'] = data['Electricity Power (kW)'].shift(2)
    data['Electricity_Air_Interaction'] = data['Electricity Power (kW)'] * data['Air Pressure Power (kW)']
    data['Electricity_MA'] = data['Electricity Power (kW)'].rolling(window=3).mean()
    data['AirPressure_MA'] = data['Air Pressure Power (kW)'].rolling(window=3).mean()
    data.dropna(inplace=True)
    features = ['Electricity Power (kW)', 'Air Pressure Power (kW)', 'Air Consumption (kW)',
                'Hour', 'DayOfWeek', 'Month', 'Electricity Power (Lag1)', 'Electricity Power (Lag2)',
                'Electricity_Air_Interaction', 'Electricity_MA', 'AirPressure_MA']
    X_data = data[features].values
    y_data = data[['Electricity Power (kW)']].values
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(y_data)
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - input_steps - output_steps + 1):
        X_seq.append(X_scaled[i:i+input_steps])
        y_seq.append(y_scaled[i+input_steps:i+input_steps+output_steps].flatten())
    return np.array(X_seq), np.array(y_seq), scaler_y

def update_model_monthly(data_file, model_file='monthly_model.h5', input_steps=720, output_steps=720):
    X_seq, y_seq, scaler_y = preprocess_data_monthly_custom(data_file, input_steps, output_steps)
    if X_seq.size == 0:
        raise ValueError("Not enough data for monthly predictions. At least {} rows are required.".format(input_steps+output_steps))
    n_features = X_seq.shape[2]
    if os.path.exists(model_file):
        model = load_model(model_file)
        print("Loaded existing monthly model from", model_file)
    else:
        raise FileNotFoundError(f"{model_file} not found.")
    X_finetune = X_seq[:-1]
    y_finetune = y_seq[:-1]
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='mean_squared_error', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_finetune, y_finetune, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stop])
    model.save(model_file)
    print("Updated monthly model saved as", model_file)
    return model, scaler_y, X_seq

def predict_next_monthly(model, scaler_y, X_seq, input_steps=720):
    if X_seq.size == 0:
        raise ValueError("Not enough data for monthly predictions.")
    n_features = X_seq.shape[2]
    X_input = X_seq[-1].reshape(1, input_steps, n_features)
    pred = model.predict(X_input)
    pred_rescaled = scaler_y.inverse_transform(pred)
    pred_rescaled[pred_rescaled < 0] = 0  # Clamp negatives to zero
    return pred_rescaled.flatten().tolist()

############################
# DAILY FUNCTIONS
############################
def preprocess_data_daily(file_path, input_steps=24, output_steps=24, skiprows=7):
    try:
        data = pd.read_csv(file_path, skiprows=skiprows)
    except Exception:
        data = pd.read_csv(file_path)
    if "Timestamp" not in data.columns:
        data.columns = ["Timestamp", "Electricity Power (kW)", "Air Pressure Power (kW)", "Air Consumption (kW)"]
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data = data[data['Timestamp'].dt.minute == 0]
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
    data['Month'] = data['Timestamp'].dt.month
    data['Electricity Power (Lag1)'] = data['Electricity Power (kW)'].shift(1)
    data['Electricity Power (Lag2)'] = data['Electricity Power (kW)'].shift(2)
    data['Interaction'] = data['Electricity Power (kW)'] * data['Air Pressure Power (kW)']
    data['Electricity_MA'] = data['Electricity Power (kW)'].rolling(window=3).mean()
    data['AirPressure_MA'] = data['Air Pressure Power (kW)'].rolling(window=3).mean()
    data.dropna(inplace=True)
    features = ["Electricity Power (kW)", "Air Pressure Power (kW)", "Air Consumption (kW)",
                "Hour", "DayOfWeek", "Month", "Electricity Power (Lag1)", "Electricity Power (Lag2)",
                "Interaction", "Electricity_MA", "AirPressure_MA"]
    X_data = data[features].values
    y_data = data[["Electricity Power (kW)"]].values
    scaler_X = MinMaxScaler(feature_range=(0,1))
    scaler_y = MinMaxScaler(feature_range=(0,1))
    X_scaled = scaler_X.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(y_data)
    X_seq = []
    y_seq = []
    for i in range(len(X_scaled) - input_steps - output_steps + 1):
        X_seq.append(X_scaled[i:i+input_steps])
        y_seq.append(y_scaled[i+input_steps:i+input_steps+output_steps].flatten())
    return np.array(X_seq), np.array(y_seq), scaler_y

def update_model_daily(data_file, model_file='daily_energy_prediction_model.h5', input_steps=24, output_steps=24):
    X_seq, y_seq, scaler_y = preprocess_data_daily(data_file, input_steps, output_steps)
    n_features = X_seq.shape[2]
    if os.path.exists(model_file):
        model = load_model(model_file, custom_objects={'mse': tf.keras.losses.mse})
        print("Loaded existing daily model from", model_file)
    else:
        raise FileNotFoundError(f"{model_file} not found.")
    X_finetune = X_seq[:-2]
    y_finetune = y_seq[:-2]
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_finetune, y_finetune, epochs=20, batch_size=16, validation_split=0.1, callbacks=[early_stop])
    model.save(model_file)
    print("Updated daily model saved as", model_file)
    return model, scaler_y, X_seq

def predict_next_month_daily(model, scaler_y, X_seq, input_steps=24):
    base_input = X_seq[-1].copy()
    data = pd.read_csv("Energy_2022-2024_testing_data.csv", skiprows=7)
    if "Timestamp" not in data.columns:
        data.columns = ["Timestamp", "Electricity Power (kW)", "Air Pressure Power (kW)", "Air Consumption (kW)"]
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data = data[data['Timestamp'].dt.minute == 0]
    last_date = data['Timestamp'].max()
    next_month_days = 31
    future_inputs = []
    for d in range(next_month_days):
        new_input = base_input.copy()
        for i in range(24):
            new_time = last_date + timedelta(days=d+1, hours=i - (24-1))
            new_input[i, 3] = new_time.hour / 23.0
            new_input[i, 4] = new_time.weekday() / 6.0
            new_input[i, 5] = new_time.month / 12.0
        future_inputs.append(new_input)
    future_inputs = np.array(future_inputs)
    preds_scaled = model.predict(future_inputs)
    preds = np.array([scaler_y.inverse_transform(day_pred.reshape(-1,1)).flatten() for day_pred in preds_scaled])
    predictions = {f"day_{d+1}": preds[d].tolist() for d in range(next_month_days)}
    return predictions
