# app/routes/daily.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import timedelta
from flask import Blueprint, jsonify, render_template, request, send_file

from app.utils.prediction_utils import (
    preprocess_data_daily,
    update_model_daily,
    predict_next_month_daily
)

daily_bp = Blueprint('daily', __name__, template_folder='templates', url_prefix='/daily')

# Load daily model (with custom objects)
DAILY_MODEL_PATH = os.path.join(os.getcwd(), 'daily_energy_prediction_model.h5')
daily_model = tf.keras.models.load_model(DAILY_MODEL_PATH, custom_objects={'mse': tf.keras.losses.mse})
print("Daily model loaded successfully.")

@daily_bp.route('/', methods=['GET'])
def daily_page():
    return render_template('daily.html')

@daily_bp.route('/predict_daily', methods=['GET'])
def predict_daily():
    try:
        file_path = 'Energy_2022-2024_testing_data.csv'
        input_steps = 24
        output_steps = 24
        X, _, scaler_y_daily = preprocess_data_daily(file_path, input_steps, output_steps)
        
        base_input = X[-1].copy()
        data = pd.read_csv(file_path, skiprows=7)
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
                # Hour, weekday, month normalization
                new_input[i, 3] = new_time.hour / 23.0
                new_input[i, 4] = new_time.weekday() / 6.0
                new_input[i, 5] = new_time.month / 12.0
            future_inputs.append(new_input)
        
        future_inputs = np.array(future_inputs)
        preds_scaled = daily_model.predict(future_inputs)
        preds = np.array([
            scaler_y_daily.inverse_transform(day_pred.reshape(-1,1)).flatten() 
            for day_pred in preds_scaled
        ])
        
        predictions = {
            f"day_{d+1}": preds[d].tolist() for d in range(next_month_days)
        }
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@daily_bp.route('/upload_daily', methods=['POST'])
def upload_daily():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    try:
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        updated_model, scaler_y_daily, X_seq_daily = update_model_daily(file_path)
        predictions = predict_next_month_daily(updated_model, scaler_y_daily, X_seq_daily)
        
        updated_model.save(DAILY_MODEL_PATH)
        global daily_model
        daily_model = updated_model
        
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
