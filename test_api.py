import requests

# API endpoint URL
api_url = "http://127.0.0.1:5001/predict"

try:
    headers = {"Content-Type": "application/json"}
    response = requests.get(api_url, headers=headers)
    
    if response.status_code == 200:
        predictions = response.json()
        print("Predictions from API:")
        print(predictions)
    else:
        print(f"Error: {response.status_code} - {response.text}")
except requests.exceptions.RequestException as e:
    print("API Request Failed:", e)
