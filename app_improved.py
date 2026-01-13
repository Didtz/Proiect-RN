"""
Plant Identification Web Application
Flask-based web interface for plant identification using deep learning
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'models/plant_model.h5'
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Plant information database
PLANT_DATABASE = {
    'rose': {
        'scientific_name': 'Rosa spp.',
        'description': 'Beautiful flowering plant with thorns',
        'watering': 'Water deeply when soil is dry 1-2 inches below surface',
        'light': 'Full sun (6+ hours daily)',
        'difficulty': 'Intermediate'
    },
    'sunflower': {
        'scientific_name': 'Helianthus annuus',
        'description': 'Large golden flowers that follow the sun',
        'watering': 'Regular watering, 1-2 inches per week',
        'light': 'Full sun (6-8 hours daily)',
        'difficulty': 'Easy'
    },
    'tulip': {
        'scientific_name': 'Tulipa spp.',
        'description': 'Spring-flowering bulb with colorful blooms',
        'watering': 'Moderate, allow soil to dry between watering',
        'light': 'Full sun to partial shade',
        'difficulty': 'Easy'
    },
    'cactus': {
        'scientific_name': 'Cactaceae',
        'description': 'Desert plant adapted to dry conditions',
        'watering': 'Sparingly, only when soil is completely dry',
        'light': 'Full sun',
        'difficulty': 'Easy'
    },
    'orchid': {
        'scientific_name': 'Orchidaceae',
        'description': 'Elegant exotic flowers',
        'watering': 'Once per week, avoid standing water',
        'light': 'Bright, indirect light',
        'difficulty': 'Advanced'
    }
}

# Global model variable
model = None
class_names = list(PLANT_DATABASE.keys())
image_size = (224, 224)


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load the trained model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found. Training required.")
            model = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None


def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(image_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', plant_count=len(PLANT_DATABASE))


@app.route('/identify')
def identify():
    """Plant identification page"""
    return render_template('identify.html')


@app.route('/database')
def database():
    """Plant database page"""
    return render_template('database.html', plants=PLANT_DATABASE)


@app.route('/guide')
def guide():
    """Care guide page"""
    return render_template('guide.html')


@app.route('/training')
def training():
    """Training status page"""
    return render_template('training.html', model_exists=model is not None)


# API Routes
@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict plant from uploaded image"""
    if model is None:
        return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess and predict
        img_array = preprocess_image(filepath)
        if img_array is None:
            return jsonify({'error': 'Failed to process image'}), 400

        predictions = model.predict(img_array, verbose=0)
        confidence = float(np.max(predictions[0]))
        predicted_class = class_names[np.argmax(predictions[0])]

        if confidence < 0.3:
            return jsonify({
                'success': False,
                'message': 'Cannot confidently identify the plant',
                'confidence': confidence
            })

        plant_info = PLANT_DATABASE.get(predicted_class, {})

        return jsonify({
            'success': True,
            'plant': predicted_class,
            'confidence': confidence,
            'info': plant_info
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/plants')
def get_plants():
    """Get all plants in database"""
    return jsonify(PLANT_DATABASE)


@app.route('/api/plant/<plant_name>')
def get_plant(plant_name):
    """Get specific plant information"""
    plant = PLANT_DATABASE.get(plant_name.lower())
    if plant:
        return jsonify({plant_name: plant})
    return jsonify({'error': 'Plant not found'}), 404


@app.route('/api/model/status')
def model_status():
    """Get model training status"""
    return jsonify({
        'trained': model is not None,
        'model_path': MODEL_PATH,
        'classes': class_names,
        'num_classes': len(class_names)
    })


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Trigger model training"""
    return jsonify({
        'message': 'Training initiated. Check the training page or terminal for progress.',
        'note': 'Run train_model.py from the command line to train the model.'
    }), 202


if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000, host='0.0.0.0')
