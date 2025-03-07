# run.py
from app import create_app

app = create_app()

if __name__ == "__main__":
    print("🚀 Starting Flask server... Access it at: http://127.0.0.1:5001/")
    app.run(host='0.0.0.0', port=5001, debug=True)
