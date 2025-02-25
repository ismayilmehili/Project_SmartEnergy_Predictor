# main.py
from app import app
import os

if __name__ == '__main__':
    print("🚀 Flask server is starting... Access it at: http://127.0.0.1:5001/")
    print("Current working directory:", os.getcwd())
    print("Directory contents:", os.listdir('.'))
    app.run(host='0.0.0.0', port=5001, debug=True)
