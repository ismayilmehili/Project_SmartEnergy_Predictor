# app/routes/monthly.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Blueprint, jsonify, render_template, request, send_file

from app.utils.prediction_utils import (
    preprocess_data_monthly_custom,
    update_model_monthly,
    predict_next_monthly
)

monthly_bp = Blueprint('monthly', __name__, template_folder='templates', url_prefix='/monthly')

MONTHLY_MODEL_PATH = os.path.join(os.getcwd(), 'monthly_model.h5')
monthly_model = tf.keras.models.load_model(MONTHLY_MODEL_PATH)
print("Monthly model loaded successfully.")

@monthly_bp.route('/', methods=['GET'])
def monthly_page():
    return render_template('monthly.html')

@monthly_bp.route('/predict_monthly', methods=['GET'])
def predict_monthly():
    file_path = os.path.join(os.getcwd(), 'Energy_2022-2024_testing_data.csv')
    input_steps = 720
    output_steps = 720
    X, _, scaler_y = preprocess_data_monthly_custom(file_path, input_steps, output_steps)
    if X.shape[0] == 0:
        return jsonify({"error": f"Not enough data for monthly predictions. Need at least {input_steps+output_steps} rows."})
    
    X_input = X[-1].reshape(1, input_steps, X.shape[2])
    pred = monthly_model.predict(X_input)
    pred_rescaled = scaler_y.inverse_transform(pred)
    pred_rescaled[pred_rescaled < 0] = 0
    prediction = pred_rescaled.flatten().tolist()
    
    # Save CSV using an absolute path
    csv_path = os.path.join(os.getcwd(), "monthly_predictions.csv")
    pd.DataFrame({
        "Time Step": list(range(1, output_steps+1)),
        "Predicted Electricity Power (kW)": prediction
    }).to_csv(csv_path, index=False)
    
    return jsonify({"monthly_prediction": prediction})

@monthly_bp.route('/download_monthly', methods=['GET'])
def download_monthly():
    csv_path = os.path.join(os.getcwd(), "monthly_predictions.csv")
    return send_file(csv_path, as_attachment=True)

@monthly_bp.route('/upload_monthly', methods=['POST'])
def upload_monthly():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    try:
        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)
        
        updated_model, scaler_y, X_seq = update_model_monthly(file_path)
        if X_seq.shape[0] == 0:
            return jsonify({"error": f"Not enough data for monthly prediction update. Need at least {720+720} rows."})
        
        prediction = predict_next_monthly(updated_model, scaler_y, X_seq)
        csv_path = os.path.join(os.getcwd(), "monthly_predictions.csv")
        pd.DataFrame({
            "Time Step": list(range(1, len(prediction)+1)),
            "Predicted Electricity Power (kW)": prediction
        }).to_csv(csv_path, index=False)
        
        return jsonify({"monthly_prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
