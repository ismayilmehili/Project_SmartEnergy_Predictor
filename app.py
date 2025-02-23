import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, send_file
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Force Flask to use the correct templates folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of app.py
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")  # Path to templates

app = Flask(__name__, template_folder=TEMPLATE_DIR)

print("✅ Flask is using template folder:", TEMPLATE_DIR)

# Load the saved model once when the app starts
model = tf.keras.models.load_model('saved_model.h5')
print("Model loaded successfully.")

def preprocess_data(file_path, input_steps=168, output_steps=168):
    # Load CSV data and set column names
    data = pd.read_csv(file_path, skiprows=7)
    data.columns = ["Timestamp", "Electricity Power (kW)", "Air Pressure Power (kW)", "Air Consumption (kW)"]
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    
    # Filter for rows at the top of the hour
    data = data[data['Timestamp'].dt.minute == 0]

    # Create time-based features
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
    data['Month'] = data['Timestamp'].dt.month

    # Create lagged features
    data['Electricity Power (Lag1)'] = data['Electricity Power (kW)'].shift(1)
    data['Electricity Power (Lag2)'] = data['Electricity Power (kW)'].shift(2)
    data.dropna(inplace=True)

    # Select features and target
    features = ['Electricity Power (kW)', 'Air Pressure Power (kW)', 'Air Consumption (kW)',
                'Hour', 'DayOfWeek', 'Month', 'Electricity Power (Lag1)', 'Electricity Power (Lag2)']
    X_data = data[features].values
    y_data = data[['Electricity Power (kW)']].values

    # Normalize the data (this must match the training phase)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(y_data)

    # Create sequences for weekly prediction
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - input_steps - output_steps + 1):
        X_seq.append(X_scaled[i:i+input_steps])
        y_seq.append(y_scaled[i+input_steps:i+input_steps+output_steps].flatten())
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    return X_seq, y_seq, scaler_y

@app.route('/predict', methods=['GET'])
def predict():
    file_path = 'Energy_2022-2024_testing_data.csv'
    input_steps = 168
    output_steps = 168

    # Preprocess the data
    X, _, scaler_y = preprocess_data(file_path, input_steps, output_steps)
    n_features = X.shape[2]

    predictions = {}
    all_data = []
    
    # Generate predictions for the last 4 weekly sequences
    for week_num in range(4):
        X_input = X[-(4 - week_num)].reshape(1, input_steps, n_features)
        pred = model.predict(X_input)
        pred_rescaled = scaler_y.inverse_transform(pred)
        
        week_data = pred_rescaled.flatten().tolist()
        predictions[f'week_{week_num+1}'] = week_data
        
        # Store for CSV download
        all_data.append({"Week": f"Week {week_num+1}", "Predictions": week_data})

    # Save all weeks data to CSV
    df = pd.DataFrame(all_data)
    df.to_csv("weekly_predictions.csv", index=False)

    return jsonify(predictions)

@app.route('/download/<week_num>')
def download_csv(week_num):
    """Download CSV file for a specific week"""
    file_path = f"week_{week_num}_predictions.csv"

    # Read predictions from the last API call
    data = pd.read_csv("weekly_predictions.csv")

    # Filter for the requested week
    week_data = data[data['Week'] == f"Week {week_num}"]

    # Save as separate CSV
    week_data.to_csv(file_path, index=False)

    return send_file(file_path, as_attachment=True)

@app.route('/download_all')
def download_all():
    """Download all weekly predictions as a single CSV"""
    return send_file("weekly_predictions.csv", as_attachment=True)

@app.route('/analyze', methods=['GET'])
def analyze():
    """Generate analysis report for predictions"""
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

    # Save the analysis to CSV
    df_analysis = pd.DataFrame(analysis_results)
    df_analysis.to_csv("analysis_report.csv", index=False)

    return jsonify(analysis_results)

@app.route('/download_analysis')
def download_analysis():
    """Download the analysis report as CSV"""
    return send_file("analysis_report.csv", as_attachment=True)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("🚀 Starting Flask server... Access it at: http://127.0.0.1:5001/")
    app.run(host='0.0.0.0', port=5001, debug=True)
