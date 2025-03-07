# app/routes/weekly.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Blueprint, jsonify, render_template, request, send_file

from app.utils.prediction_utils import (
    preprocess_data,
    update_model,
    predict_next_month
)

weekly_bp = Blueprint('weekly', __name__, template_folder='templates', url_prefix='/weekly')

WEEKLY_MODEL_PATH = os.path.join(os.getcwd(), 'saved_model.h5')
weekly_model = tf.keras.models.load_model(WEEKLY_MODEL_PATH)
print("Weekly model loaded successfully.")

@weekly_bp.route('/', methods=['GET'])
def weekly_page():
    return render_template('weekly.html')

@weekly_bp.route('/predict', methods=['GET'])
def predict():
    file_path = os.path.join(os.getcwd(), 'Energy_2022-2024_testing_data.csv')
    input_steps = 168
    output_steps = 168
    X, _, scaler_y = preprocess_data(file_path, input_steps, output_steps)
    
    predictions = {}
    all_data = []
    for week_num in range(4):
        X_input = X[-(4 - week_num)].reshape(1, input_steps, X.shape[2])
        pred = weekly_model.predict(X_input)
        pred_rescaled = scaler_y.inverse_transform(pred)
        pred_rescaled[pred_rescaled < 0] = 0
        week_data = pred_rescaled.flatten().tolist()
        predictions[f'week_{week_num+1}'] = week_data
        all_data.append({"Week": f"Week {week_num+1}", "Predictions": week_data})
    
    # Save CSV using an absolute path
    csv_path = os.path.join(os.getcwd(), "weekly_predictions.csv")
    pd.DataFrame(all_data).to_csv(csv_path, index=False)
    
    return jsonify(predictions)

@weekly_bp.route('/download/<week_num>', methods=['GET'])
def download_csv(week_num):
    csv_path = os.path.join(os.getcwd(), "weekly_predictions.csv")
    data = pd.read_csv(csv_path)
    week_data = data[data['Week'] == f"Week {week_num}"]
    temp_path = os.path.join(os.getcwd(), f"week_{week_num}_predictions.csv")
    week_data.to_csv(temp_path, index=False)
    return send_file(temp_path, as_attachment=True)

@weekly_bp.route('/download_all', methods=['GET'])
def download_all():
    csv_path = os.path.join(os.getcwd(), "weekly_predictions.csv")
    return send_file(csv_path, as_attachment=True)

@weekly_bp.route('/analyze', methods=['GET'])
def analyze():
    csv_path = os.path.join(os.getcwd(), "weekly_predictions.csv")
    if not os.path.exists(csv_path):
        return jsonify({"error": "No predictions found. Run /weekly/predict first."}), 400
    data = pd.read_csv(csv_path)
    analysis_results = []
    for week in ["Week 1", "Week 2", "Week 3", "Week 4"]:
        week_data = data[data["Week"] == week]
        if not week_data.empty:
            preds = eval(week_data["Predictions"].values[0])
            analysis_results.append({
                "Week": week,
                "Mean": np.mean(preds),
                "Min": np.min(preds),
                "Max": np.max(preds)
            })
    analysis_csv = os.path.join(os.getcwd(), "analysis_report.csv")
    pd.DataFrame(analysis_results).to_csv(analysis_csv, index=False)
    return jsonify(analysis_results)

@weekly_bp.route('/download_analysis', methods=['GET'])
def download_analysis():
    analysis_csv = os.path.join(os.getcwd(), "analysis_report.csv")
    return send_file(analysis_csv, as_attachment=True)

@weekly_bp.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    try:
        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)
        
        updated_model, scaler_y, X_seq = update_model(file_path)
        predictions = predict_next_month(updated_model, scaler_y, X_seq)
        
        all_data = []
        for week_num in range(4):
            week_data = predictions[f'week_{week_num+1}']
            all_data.append({"Week": f"Week {week_num+1}", "Predictions": week_data})
        
        csv_path = os.path.join(os.getcwd(), "weekly_predictions.csv")
        pd.DataFrame(all_data).to_csv(csv_path, index=False)
        
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
