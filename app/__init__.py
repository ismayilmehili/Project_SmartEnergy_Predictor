# app/__init__.py
import os
from flask import Flask, render_template

def create_app():
    # Set template and static folders to use the project root directories.
    app = Flask(__name__,
                template_folder=os.path.join(os.getcwd(), 'templates'),
                static_folder=os.path.join(os.getcwd(), 'static'))
    app.secret_key = 'supersecretkey'
    
    # Import and register Blueprints
    from app.routes.weekly import weekly_bp
    from app.routes.monthly import monthly_bp
    from app.routes.daily import daily_bp

    app.register_blueprint(weekly_bp)
    app.register_blueprint(monthly_bp)
    app.register_blueprint(daily_bp)
    
    # Main index route
    @app.route('/')
    def index():
        return render_template("index.html")
    
    return app
