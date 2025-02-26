import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Import functions from update.py
from update import update_model, predict_next_month, update_model_monthly, predict_next_monthly, preprocess_data_monthly_custom

# Set up Flask and template folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = 'supersecretkey'

print("✅ Flask is using template folder:", TEMPLATE_DIR)

# Load weekly model
model = tf.keras.models.load_model('saved_model.h5')
print("Weekly model loaded successfully.")

# Load monthly model
monthly_model = tf.keras.models.load_model('monthly_model.h5')
print("Monthly model loaded successfully.")

# ---------------------- Helper Function for Weekly ----------------------
def preprocess_data(file_path, input_steps=168, output_steps=168):
    data = pd.read_csv(file_path, skiprows=7)
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

# ---------------------- Main Navigation Routes ----------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/weekly')
def weekly_page():
    return render_template('weekly.html')

@app.route('/monthly')
def monthly_page():
    return render_template('monthly.html')

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
        pred = model.predict(X_input)
        pred_rescaled = scaler_y.inverse_transform(pred)
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
            predictions = eval(week_data["Predictions"].values[0])
            mean_val = np.mean(predictions)
            min_val = np.min(predictions)
            max_val = np.max(predictions)
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

    # Use the custom monthly preprocessing function
    X, _, scaler_y = preprocess_data_monthly_custom(file_path, input_steps, output_steps)
    if X.shape[0] == 0:
        return jsonify({"error": "Not enough data for monthly predictions. At least {} rows required.".format(input_steps + output_steps)})
    n_features = X.shape[2]
    X_input = X[-1].reshape(1, input_steps, n_features)
    pred = monthly_model.predict(X_input)
    pred_rescaled = scaler_y.inverse_transform(pred)
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
            return jsonify({"error": "Not enough data for monthly prediction update. At least {} rows required.".format(720+720)})
        prediction = predict_next_monthly(updated_model, scaler_y, X_seq)
        df = pd.DataFrame({
            "Time Step": range(1, len(prediction)+1),
            "Predicted Electricity Power (kW)": prediction
        })
        df.to_csv("monthly_predictions.csv", index=False)
        return jsonify({"monthly_prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server... Access it at: http://127.0.0.1:5001/")
    app.run(host='0.0.0.0', port=5001, debug=True)
