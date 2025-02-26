import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def preprocess_data_15min(file_path, input_steps=168, output_steps=168, skiprows=7):
    try:
        data = pd.read_csv(file_path, skiprows=skiprows)
    except Exception as e:
        print(f"Error reading with skiprows={skiprows}: {e}. Trying without skipping rows.")
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
    data.dropna(inplace=True)
    
    features = ["Electricity Power (kW)", "Air Pressure Power (kW)", "Air Consumption (kW)",
                "Hour", "DayOfWeek", "Month", "Electricity Power (Lag1)", "Electricity Power (Lag2)"]
    X_data = data[features].values
    y_data = data[["Electricity Power (kW)"]].values

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(y_data)
    
    X_seq = []
    y_seq = []
    for i in range(len(X_scaled) - input_steps - output_steps + 1):
        X_seq.append(X_scaled[i:i+input_steps])
        y_seq.append(y_scaled[i+input_steps:i+input_steps+output_steps].flatten())
    
    return np.array(X_seq), np.array(y_seq), scaler_y

def update_model(data_file, model_file='saved_model.h5', input_steps=168, output_steps=168):
    X_seq, y_seq, scaler_y = preprocess_data_15min(data_file, input_steps, output_steps)
    n_features = X_seq.shape[2]
    
    if os.path.exists(model_file):
        model = load_model(model_file)
        print("Loaded existing model from", model_file)
    else:
        raise FileNotFoundError(f"{model_file} not found.")
    
    # Fine-tuning using all but the last 4 sequences
    X_finetune = X_seq[:-4]
    y_finetune = y_seq[:-4]
    
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='mean_squared_error', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_finetune, y_finetune, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stop])
    
    model.save(model_file)
    print("Updated model saved as", model_file)
    return model, scaler_y, X_seq

def predict_next_month(model, scaler_y, X_seq, input_steps=168):
    predictions = {}
    n_features = X_seq.shape[2]
    for week_num in range(4):
        X_input = X_seq[-(4 - week_num)].reshape(1, input_steps, n_features)
        pred = model.predict(X_input)
        pred_rescaled = scaler_y.inverse_transform(pred)
        predictions[f"week_{week_num+1}"] = pred_rescaled.flatten().tolist()
    return predictions

# New monthly preprocessing function that replicates your training pipeline
def preprocess_data_monthly_custom(file_path, input_steps=720, output_steps=720):
    try:
        data = pd.read_csv(file_path, skiprows=7)
    except Exception as e:
        print(f"Error reading with skiprows=7: {e}. Trying without skipping rows.")
        data = pd.read_csv(file_path)
    
    # Check number of columns and adjust if needed
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
    # Extra features for monthly training
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
    
    X_seq = []
    y_seq = []
    for i in range(len(X_scaled) - input_steps - output_steps + 1):
        X_seq.append(X_scaled[i:i+input_steps])
        y_seq.append(y_scaled[i+input_steps:i+input_steps+output_steps].flatten())
    
    return np.array(X_seq), np.array(y_seq), scaler_y

def update_model_monthly(data_file, model_file='monthly_model.h5', input_steps=720, output_steps=720):
    X_seq, y_seq, scaler_y = preprocess_data_monthly_custom(data_file, input_steps, output_steps)
    
    # Check if any sequence was created
    if X_seq.size == 0:
        raise ValueError("Not enough data for monthly predictions. At least {} rows are required.".format(input_steps + output_steps))
    
    n_features = X_seq.shape[2]
    
    if os.path.exists(model_file):
        model = load_model(model_file)
        print("Loaded existing monthly model from", model_file)
    else:
        raise FileNotFoundError(f"{model_file} not found.")
    
    # Fine-tuning using all but the last sequence
    X_finetune = X_seq[:-1]
    y_finetune = y_seq[:-1]
    
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='mean_squared_error', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_finetune, y_finetune, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stop])
    
    model.save(model_file)
    print("Updated monthly model saved as", model_file)
    return model, scaler_y, X_seq

def predict_next_monthly(model, scaler_y, X_seq, input_steps=720):
    # Check if there is at least one sequence
    if X_seq.size == 0:
        raise ValueError("Not enough data for monthly predictions. Please provide at least {} rows.".format(input_steps))
    n_features = X_seq.shape[2]
    X_input = X_seq[-1].reshape(1, input_steps, n_features)
    pred = model.predict(X_input)
    pred_rescaled = scaler_y.inverse_transform(pred)
    prediction = pred_rescaled.flatten().tolist()
    return prediction
