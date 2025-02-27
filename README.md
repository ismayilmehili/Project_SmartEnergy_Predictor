# Smart Energy Predictor

Smart Energy Predictor is a Flask-based web application that leverages LSTM models to forecast electricity power consumption on both a weekly and monthly basis. The application allows users to update models by uploading new CSV files and provides predictions along with analysis reports.

## Repository Link
[Project_SmartEnergyPredictor](https://github.com/ismayilmehili/Project_SmartEnergyPredictor.git)

## Features

- **Weekly Predictions:**  
  Forecasts energy consumption for the upcoming 4 weeks using an LSTM model.  
  - **Input Steps:** 168  
  - **Output Steps:** 168  
  - **Required Rows:** 336

- **Monthly Predictions:**  
  Predicts the next month's energy consumption with additional engineered features (lag features, moving averages, interaction terms).  
  - **Input Steps:** 720  
  - **Output Steps:** 720  
  - **Required Rows:** 1440

- **Model Updates:**  
  Update both weekly and monthly models using new CSV uploads through dedicated endpoints.

- **Data Analysis and Reporting:**  
  Generates downloadable CSV reports containing predictions and statistical analysis (mean, min, max values).

## File Structure

- `app.py`  
  Main Flask application handling routes for predictions, uploads, downloads, and analysis.

- `update.py`  
  Contains functions for data preprocessing, updating the models, and sequence creation.

- `main.py`  
  Entry point for running the Flask server.

- `templates/`  
  HTML templates for the web interface.

- `requirements.txt`  
  Lists the project dependencies.

- `monthly_model.h5` & `saved_model.h5`  
  Pre-trained TensorFlow LSTM models for monthly and weekly predictions, respectively.

## Installation and Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ismayilmehili/Project_SmartEnergyPredictor.git
   cd Project_SmartEnergyPredictor
