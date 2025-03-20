# Smart Energy Predictor

**Smart Energy Predictor** is a Flask-based web application leveraging pre-trained LSTM models to forecast electricity consumption. It provides comprehensive daily, weekly, and monthly predictions, along with data analysis capabilities and convenient model updating via CSV uploads.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Testing](#api-testing)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Features

- **Daily Predictions (24-Hour Forecast)**
  - Predict hourly electricity consumption for the next day.
  - **Input Steps:** 24 | **Output Steps:** 24
  - **Minimum Data Required:** 48 rows

- **Weekly Predictions (4-Week Forecast)**
  - Generate weekly forecasts (hourly predictions for each week).
  - **Input Steps:** 168 | **Output Steps:** 168
  - **Minimum Data Required:** 336 rows

- **Monthly Predictions (Upcoming Month Forecast)**
  - Predict next month's consumption using advanced features (lags, moving averages, interactions).
  - **Input Steps:** 720 | **Output Steps:** 720
  - **Minimum Data Required:** 1440 rows

- **Model Updates via CSV**
  - Easily update daily, weekly, and monthly prediction models using dedicated CSV upload endpoints.

- **Data Analysis & Reporting**
  - Generate and download CSV reports containing prediction statistics (mean, min, max).

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/ismayilmehili/Project_SmartEnergyPredictor.git
cd Project_SmartEnergyPredictor
```

### Step 2: Set up Virtual Environment

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Add Pre-trained Models
Place the following TensorFlow models into the `models/` directory:
- `daily_energy_prediction_model.h5`
- `saved_model.h5` (weekly predictions)
- `monthly_model.h5`

## Usage

### Start the Flask Server
```bash
python run.py
```

Server starts at: [http://127.0.0.1:5001](http://127.0.0.1:5001)

### Navigating the Web App
- **Dashboard:** Select Daily, Weekly, or Monthly prediction views.

- **Daily Predictions:**
  - Upload CSV to update daily model.
  - View forecasts and download detailed reports.

- **Weekly Predictions:**
  - Upload CSV to refresh weekly model.
  - Analyze weekly trends and download individual or aggregate reports.

- **Monthly Predictions:**
  - Upload CSV to update monthly forecasts.
  - Access detailed monthly predictions and reports.

## API Testing
Run the included API test script to verify endpoint responses:

```bash
python testapi.py
```
This sends a test GET request to the prediction endpoint and outputs the response JSON.

-------------------------------------------------------------------------------------------------
