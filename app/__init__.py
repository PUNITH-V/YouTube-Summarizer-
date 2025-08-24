"""Flask application initialization."""
from flask import Flask
import os

# Initialize Flask app with proper static and template folders
flask_app = Flask(__name__,
                 static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'),
                 template_folder='templates')

# Set maximum content length to 16MB
flask_app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
