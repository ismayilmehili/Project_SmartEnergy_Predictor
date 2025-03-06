import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# Import helper functions from update.py
from update import (
    preprocess_data, preprocess_data_monthly_custom,
    update_model, predict_next_month,
    update_model_monthly, predict_next_monthly,
    preprocess_data_daily, update_model_daily, predict_next_month_daily
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = 'supersecretkey'

print("✅ Flask is using template folder:", TEMPLATE_DIR)

# Load models
weekly_model = tf.keras.models.load_model('saved_model.h5')
print("Weekly model loaded successfully.")

monthly_model = tf.keras.models.load_model('monthly_model.h5')
print("Monthly model loaded successfully.")

daily_model = tf.keras.models.load_model('daily_energy_prediction_model.h5',
                                         custom_objects={'mse': tf.keras.losses.mse})
print("Daily model loaded successfully.")

# ---------------------- Main Navigation ----------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/weekly')
def weekly_page():
    return render_template('weekly.html')

@app.route('/monthly')
def monthly_page():
    return render_template('monthly.html')

@app.route('/daily')
def daily_page():
    return render_template('daily.html')

# ---------------------- Weekly Endpoints ----------------------
@app.route('/predict', methods=['GET'])
def predict():
    file_path = 'Energy_2022-2024_testing_data.csv'
    input_steps = 168
    output_steps = 168
    X, _, scaler_y = preprocess_data(file_path, input_steps, output_steps)
    n_features = X.shape[2]
    predictions = {}
    all_data = []
    for week_num in range(4):
        X_input = X[-(4 - week_num)].reshape(1, input_steps, n_features)
        pred = weekly_model.predict(X_input)
        pred_rescaled = scaler_y.inverse_transform(pred)
        pred_rescaled[pred_rescaled < 0] = 0  # Clamp negatives to zero
        week_data = pred_rescaled.flatten().tolist()
        predictions[f'week_{week_num+1}'] = week_data
        all_data.append({"Week": f"Week {week_num+1}", "Predictions": week_data})
    df = pd.DataFrame(all_data)
    df.to_csv("weekly_predictions.csv", index=False)
    return jsonify(predictions)

@app.route('/download/<week_num>')
def download_csv(week_num):
    file_path = f"week_{week_num}_predictions.csv"
    data = pd.read_csv("weekly_predictions.csv")
    week_data = data[data['Week'] == f"Week {week_num}"]
    week_data.to_csv(file_path, index=False)
    return send_file(file_path, as_attachment=True)

@app.route('/download_all')
def download_all():
    return send_file("weekly_predictions.csv", as_attachment=True)

@app.route('/analyze', methods=['GET'])
def analyze():
    file_path = "weekly_predictions.csv"
    if not os.path.exists(file_path):
        return jsonify({"error": "No predictions found. Run /predict first."}), 400
    data = pd.read_csv(file_path)
    analysis_results = []
    weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
    for week in weeks:
        week_data = data[data["Week"] == week]
        if not week_data.empty:
            preds = eval(week_data["Predictions"].values[0])
            mean_val = np.mean(preds)
            min_val = np.min(preds)
            max_val = np.max(preds)
            analysis_results.append({"Week": week, "Mean": mean_val, "Min": min_val, "Max": max_val})
    df_analysis = pd.DataFrame(analysis_results)
    df_analysis.to_csv("analysis_report.csv", index=False)
    return jsonify(analysis_results)

@app.route('/download_analysis')
def download_analysis():
    return send_file("analysis_report.csv", as_attachment=True)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    try:
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        updated_model, scaler_y, X_seq = update_model(file_path)
        predictions = predict_next_month(updated_model, scaler_y, X_seq)
        all_data = []
        for week_num in range(4):
            week_key = f'week_{week_num+1}'
            week_data = predictions[week_key]
            all_data.append({"Week": f"Week {week_num+1}", "Predictions": week_data})
        df = pd.DataFrame(all_data)
        df.to_csv("weekly_predictions.csv", index=False)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------- Monthly Endpoints ----------------------
@app.route('/predict_monthly', methods=['GET'])
def predict_monthly():
    file_path = 'Energy_2022-2024_testing_data.csv'
    input_steps = 720
    output_steps = 720
    X, _, scaler_y = preprocess_data_monthly_custom(file_path, input_steps, output_steps)
    if X.shape[0] == 0:
        return jsonify({"error": f"Not enough data for monthly predictions. Need at least {input_steps+output_steps} rows."})
    n_features = X.shape[2]
    X_input = X[-1].reshape(1, input_steps, n_features)
    pred = monthly_model.predict(X_input)
    pred_rescaled = scaler_y.inverse_transform(pred)
    pred_rescaled[pred_rescaled < 0] = 0  # Clamp negatives
    prediction = pred_rescaled.flatten().tolist()
    df = pd.DataFrame({
        "Time Step": range(1, output_steps+1),
        "Predicted Electricity Power (kW)": prediction
    })
    df.to_csv("monthly_predictions.csv", index=False)
    return jsonify({"monthly_prediction": prediction})

@app.route('/download_monthly')
def download_monthly():
    return send_file("monthly_predictions.csv", as_attachment=True)

@app.route('/upload_monthly', methods=['POST'])
def upload_monthly():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    try:
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        updated_model, scaler_y, X_seq = update_model_monthly(file_path)
        if X_seq.shape[0] == 0:
            return jsonify({"error": f"Not enough data for monthly prediction update. Need at least {720+720} rows."})
        prediction = predict_next_monthly(updated_model, scaler_y, X_seq)
        df = pd.DataFrame({
            "Time Step": range(1, len(prediction)+1),
            "Predicted Electricity Power (kW)": prediction
        })
        df.to_csv("monthly_predictions.csv", index=False)
        return jsonify({"monthly_prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------- Daily Endpoints ----------------------
@app.route('/predict_daily', methods=['GET'])
def predict_daily():
    try:
        file_path = 'Energy_2022-2024_testing_data.csv'
        input_steps = 24
        output_steps = 24
        X, _, scaler_y_daily = preprocess_data_daily(file_path, input_steps, output_steps)
        n_features = X.shape[2]
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
                new_input[i, 3] = new_time.hour / 23.0
                new_input[i, 4] = new_time.weekday() / 6.0
                new_input[i, 5] = new_time.month / 12.0
            future_inputs.append(new_input)
        future_inputs = np.array(future_inputs)
        preds_scaled = daily_model.predict(future_inputs)
        preds = np.array([scaler_y_daily.inverse_transform(day_pred.reshape(-1,1)).flatten() for day_pred in preds_scaled])
        predictions = {f"day_{d+1}": preds[d].tolist() for d in range(next_month_days)}
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_daily', methods=['POST'])
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
        updated_model.save("daily_energy_prediction_model.h5")
        global daily_model
        daily_model = updated_model
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))

