"""
Plant Identification using TensorFlow
A project to identify plants from images and provide care guides
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# Configuration
CONFIG = {
    'image_size': (224, 224),
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'test_split': 0.1,
}

# Plant Care Guides Database
PLANT_CARE_GUIDES = {
    'rose': {
        'scientific_name': 'Rosa spp.',
        'watering': 'Water deeply when soil is dry 1-2 inches below surface. Daily in hot weather.',
        'light': 'Full sun (6+ hours daily)',
        'humidity': '40-70%',
        'temperature': '65-75°F (18-24°C)',
        'soil': 'Well-draining, pH 6.0-6.5',
        'fertilizer': 'Monthly during growing season',
        'propagation': 'Cuttings in spring',
        'common_issues': ['Powdery mildew', 'Black spot', 'Aphids'],
        'difficulty': 'Intermediate'
    },
    'sunflower': {
        'scientific_name': 'Helianthus annuus',
        'watering': 'Regular watering, 1-2 inches per week',
        'light': 'Full sun (6-8 hours daily)',
        'humidity': '40-60%',
        'temperature': '70-85°F (21-29°C)',
        'soil': 'Well-draining, neutral pH',
        'fertilizer': 'Monthly balanced fertilizer',
        'propagation': 'Seeds in spring',
        'common_issues': ['Sunflower moths', 'Rust', 'Root rot'],
        'difficulty': 'Easy'
    },
    'orchid': {
        'scientific_name': 'Orchidaceae',
        'watering': 'Once per week, avoid standing water',
        'light': 'Bright, indirect light',
        'humidity': '50-70%',
        'temperature': '65-75°F day, 55-65°F night',
        'soil': 'Orchid bark mix',
        'fertilizer': 'Diluted weekly during growing season',
        'propagation': 'Division or keiki',
        'common_issues': ['Root rot', 'Scale insects', 'Flower drop'],
        'difficulty': 'Advanced'
    },
    'tulip': {
        'scientific_name': 'Tulipa spp.',
        'watering': 'Moderate, allow soil to dry between watering',
        'light': 'Full sun to partial shade',
        'humidity': '30-40%',
        'temperature': '55-70°F (13-21°C)',
        'soil': 'Well-draining, sandy loam',
        'fertilizer': 'Spring and fall',
        'propagation': 'Bulb division',
        'common_issues': ['Tulip breaking virus', 'Botrytis', 'Slugs'],
        'difficulty': 'Easy'
    },
    'cactus': {
        'scientific_name': 'Cactaceae',
        'watering': 'Sparingly, only when soil is completely dry',
        'light': 'Full sun',
        'humidity': '20-30%',
        'temperature': '70-90°F (21-32°C)',
        'soil': 'Well-draining, sandy/rocky',
        'fertilizer': 'Rarely, light during growing season',
        'propagation': 'Cuttings or seeds',
        'common_issues': ['Root rot', 'Scale', 'Mealybugs'],
        'difficulty': 'Easy'
    }
}


class PlantIdentificationModel:
    """Handles model creation, training, and inference for plant identification"""
    
    def __init__(self, num_classes=len(PLANT_CARE_GUIDES)):
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = list(PLANT_CARE_GUIDES.keys())
        
    def build_model(self):
        """Build transfer learning model using MobileNetV2"""
        base_model = MobileNetV2(
            input_shape=(*CONFIG['image_size'], 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers
        model = keras.Sequential([
            layers.Input(shape=(*CONFIG['image_size'], 3)),
            layers.Rescaling(1./255),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, train_generator, validation_generator, epochs=None):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        epochs = epochs or CONFIG['epochs']
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        ]
        
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, image_path, confidence_threshold=0.5):
        """Predict plant type from image"""
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(
            image_path,
            target_size=CONFIG['image_size']
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Get prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        if confidence < confidence_threshold:
            return {
                'plant': 'Unknown',
                'confidence': float(confidence),
                'care_guide': None
            }
        
        plant_name = self.class_names[predicted_class_idx]
        
        return {
            'plant': plant_name,
            'confidence': float(confidence),
            'care_guide': PLANT_CARE_GUIDES.get(plant_name),
            'all_predictions': {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
        }
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


class PlantCareGuide:
    """Provides plant care information and recommendations"""
    
    @staticmethod
    def display_care_guide(plant_name):
        """Display comprehensive care guide for a plant"""
        if plant_name not in PLANT_CARE_GUIDES:
            return f"Care guide for '{plant_name}' not available."
        
        guide = PLANT_CARE_GUIDES[plant_name]
        
        guide_text = f"""
╔════════════════════════════════════════════╗
║        PLANT CARE GUIDE: {plant_name.upper():^30} ║
╚════════════════════════════════════════════╝

📋 BASIC INFORMATION
  Scientific Name: {guide['scientific_name']}
  Difficulty Level: {guide['difficulty']}

💧 WATERING
  {guide['watering']}

☀️ LIGHT REQUIREMENTS
  {guide['light']}

💨 HUMIDITY
  {guide['humidity']}

🌡️ TEMPERATURE
  {guide['temperature']}

🌱 SOIL
  {guide['soil']}

🍽️ FERTILIZER
  {guide['fertilizer']}

🌿 PROPAGATION
  {guide['propagation']}

⚠️ COMMON ISSUES
  {', '.join(guide['common_issues'])}

"""
        return guide_text
    
    @staticmethod
    def get_watering_schedule(plant_name):
        """Get specific watering schedule"""
        if plant_name in PLANT_CARE_GUIDES:
            return PLANT_CARE_GUIDES[plant_name]['watering']
        return None
    
    @staticmethod
    def export_guides_to_json(output_path):
        """Export all care guides to JSON"""
        with open(output_path, 'w') as f:
            json.dump(PLANT_CARE_GUIDES, f, indent=2)
        print(f"Care guides exported to {output_path}")


def create_data_generators(train_dir, validation_split=0.2):
    """Create train and validation data generators"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator


def main():
    """Main function to demonstrate the plant identification system"""
    print("🌿 Plant Identification System with TensorFlow 🌿\n")
    
    # Example: Display care guides
    print("Available Plants:")
    for plant in PLANT_CARE_GUIDES.keys():
        print(f"  - {plant}")
    
    print("\n" + "="*50)
    
    # Display sample care guide
    sample_plant = 'rose'
    guide = PlantCareGuide.display_care_guide(sample_plant)
    print(guide)
    
    # Initialize model (training would require actual image data)
    print("\n" + "="*50)
    print("Model Initialization Example:")
    model_handler = PlantIdentificationModel()
    model_handler.build_model()
    print("✓ Model architecture built successfully")
    print(f"  Total trainable parameters: {model_handler.model.count_params():,}")
    
    # To train the model, you would do:
    # train_gen, val_gen = create_data_generators('path/to/plant/images')
    # model_handler.train(train_gen, val_gen)
    # model_handler.save_model('plant_model.h5')
    
    print("\n✓ System ready! Load images and train the model with your plant dataset.")


if __name__ == "__main__":
    main()
