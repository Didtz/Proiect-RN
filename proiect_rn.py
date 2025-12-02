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
import zipfile
import shutil

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


def extract_images_from_archives(downloads_path=None):
    """
    Extract plant images from plante1.zip & plante2.zip in Downloads folder
    
    Args:
        downloads_path: Path to Downloads folder (auto-detects if None)
    
    Returns:
        Path to organized plant_images folder
    """
    if downloads_path is None:
        # Auto-detect Downloads folder
        downloads_path = Path.home() / "Downloads"
    
    downloads_path = Path(downloads_path)
    project_path = Path("d:\\Facultate\\Anul III\\RN")
    images_dir = project_path / "plant_images"
    
    print(f"🔍 Looking for archives in: {downloads_path}")
    
    # Look for specific archive files
    archive_names = ["plante1.zip", "plante2.zip"]
    zip_files = []
    
    for archive_name in archive_names:
        archive_path = downloads_path / archive_name
        if archive_path.exists():
            zip_files.append(archive_path)
            print(f"   ✓ Found: {archive_name}")
        else:
            print(f"   ⚠️  Missing: {archive_name}")
    
    if not zip_files:
        print("\n❌ Required archives not found in Downloads!")
        print(f"   Expected: plante1.zip & plante2.zip")
        print(f"   Location: {downloads_path}")
        return None
    
    print(f"\n✓ Found {len(zip_files)} archive(s)")
    
    # Create plant_images directory
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract all zip files
    for zip_path in zip_files:
        print(f"\n📦 Extracting: {zip_path.name}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(images_dir)
            print(f"   ✓ Extracted successfully")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Organize files by plant type (if needed)
    print("\n📂 Organizing plant folders...")
    organize_plant_folders(images_dir)
    
    print(f"\n✓ Images ready at: {images_dir}")
    return str(images_dir)


def organize_plant_folders(images_dir):
    """Organize extracted images into plant-type folders"""
    images_dir = Path(images_dir)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    
    for item in images_dir.iterdir():
        if item.is_file() and item.suffix.lower() in image_extensions:
            # Try to determine plant type from filename
            filename = item.stem.lower()
            plant_type = None
            
            for known_plant in PLANT_CARE_GUIDES.keys():
                if known_plant in filename:
                    plant_type = known_plant
                    break
            
            # If no match, try to extract from folder name
            if not plant_type:
                # Create a generic folder for unclassified images
                plant_type = 'unclassified'
            
            plant_folder = images_dir / plant_type
            plant_folder.mkdir(exist_ok=True)
            
            # Move file to plant folder
            new_path = plant_folder / item.name
            if not new_path.exists():
                shutil.move(str(item), str(new_path))
    
    # Display folder structure
    print("\n📋 Plant folders created:")
    for folder in sorted(images_dir.iterdir()):
        if folder.is_dir():
            image_count = len(list(folder.glob('*.*')))
            print(f"   {folder.name}: {image_count} images")


def create_sample_images(output_dir, num_samples_per_class=10):
    """
    Create sample training images for demonstration
    In production, use your actual plant images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🖼️  Creating sample training images...")
    
    # Create folders for each plant
    for plant_name in PLANT_CARE_GUIDES.keys():
        plant_folder = output_dir / plant_name
        plant_folder.mkdir(exist_ok=True)
        
        # Create synthetic images
        for i in range(num_samples_per_class):
            # Generate random image array
            img_array = np.random.randint(0, 256, (*CONFIG['image_size'], 3), dtype=np.uint8)
            
            # Make the image somewhat realistic (green-ish for plants)
            img_array[:, :, 0] = img_array[:, :, 0] * 0.6  # Reduce red
            img_array[:, :, 1] = (img_array[:, :, 1] * 0.8 + 50).astype(np.uint8)  # Boost green
            img_array[:, :, 2] = img_array[:, :, 2] * 0.6  # Reduce blue
            
            # Convert to PIL image and save
            img = keras.preprocessing.image.array_to_img(img_array)
            img_path = plant_folder / f"{plant_name}_{i:03d}.jpg"
            img.save(str(img_path))
        
        print(f"   ✓ Created {num_samples_per_class} sample images for {plant_name}")
    
    print(f"✓ Sample images created in: {output_dir}")
    return str(output_dir)


def create_data_generators(train_dir, validation_split=0.2):
    """Create train and validation data generators"""
    train_dir = Path(train_dir)
    
    # Check if directory has subdirectories with images
    subdirs = [d for d in train_dir.iterdir() if d.is_dir()]
    image_files = list(train_dir.glob('**/*.jpg')) + list(train_dir.glob('**/*.png'))
    
    if not subdirs or len(image_files) == 0:
        print("\n⚠️  No images found in subdirectories!")
        print("   Creating sample images for demonstration...")
        train_dir = Path(create_sample_images(train_dir))
    
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
        str(train_dir),
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator


def main():
    """Main function to demonstrate the plant identification system"""
    print("🌿 Plant Identification System with TensorFlow 🌿\n")
    
    # Step 1: Extract images from archives
    print("STEP 1: Loading images from archives")
    print("="*50)
    images_dir = extract_images_from_archives()
    
    if not images_dir:
        print("\n⚠️  Please download plant images to your Downloads folder first!")
        print("   Then run this script again.")
        return
    
    print("\n" + "="*50)
    
    # Step 2: Display available plants
    print("\nSTEP 2: Available Plants")
    print("="*50)
    print("Database contains:")
    for plant in PLANT_CARE_GUIDES.keys():
        print(f"  - {plant}")
    
    print("\n" + "="*50)
    
    # Step 3: Display sample care guide
    print("\nSTEP 3: Sample Care Guide")
    print("="*50)
    sample_plant = 'rose'
    guide = PlantCareGuide.display_care_guide(sample_plant)
    print(guide)
    
    # Step 4: Initialize model
    print("\n" + "="*50)
    print("STEP 4: Model Initialization")
    print("="*50)
    model_handler = PlantIdentificationModel()
    model_handler.build_model()
    print("✓ Model architecture built successfully")
    print(f"  Total trainable parameters: {model_handler.model.count_params():,}")
    
    print("\n" + "="*50)
    print("✓ System ready!")
    print("\n📚 NEXT STEPS:")
    print("  1. Images are loaded from: " + images_dir)
    print("  2. Train the model:")
    print("     train_gen, val_gen = create_data_generators('" + images_dir + "')")
    print("     model_handler.train(train_gen, val_gen)")
    print("     model_handler.save_model('plant_model.h5')")
    print("\n  3. Make predictions on new images:")
    print("     result = model_handler.predict('path/to/image.jpg')")
    print("     print(result['plant'], result['confidence'])")


def train_model(images_dir, model_name='plant_model.h5', epochs=20):
    """Train the plant identification model"""
    print("\n" + "="*60)
    print("TRAINING THE MODEL")
    print("="*60)
    
    # Create data generators
    print(f"\n📊 Creating data generators from: {images_dir}")
    try:
        train_gen, val_gen = create_data_generators(images_dir)
    except Exception as e:
        print(f"❌ Error creating data generators: {e}")
        return None
    
    # Initialize model
    print("\n🤖 Building model...")
    model_handler = PlantIdentificationModel()
    model_handler.build_model()
    print(f"✓ Model ready with {model_handler.model.count_params():,} parameters")
    
    # Train model
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print("   (This may take several minutes)\n")
    
    history = model_handler.train(train_gen, val_gen, epochs=epochs)
    
    # Save model
    print(f"\n💾 Saving model to: {model_name}")
    model_handler.save_model(model_name)
    
    # Plot training history
    plot_training_history(history)
    
    return model_handler, history


def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    if history is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\n📈 Training history saved to: training_history.png")
    plt.show()


def predict_on_image(model_handler, image_path):
    """Make a prediction on a single image and display results"""
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    print(f"\n🔍 Analyzing image: {image_path}")
    result = model_handler.predict(image_path)
    
    print(f"\n📊 PREDICTION RESULTS:")
    print(f"   Plant: {result['plant']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    
    if result['care_guide']:
        print(f"\n🌿 CARE GUIDE:")
        print(PlantCareGuide.display_care_guide(result['plant']))
    
    # Show all predictions
    print("\n📈 All predictions:")
    for plant, confidence in sorted(result['all_predictions'].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(confidence * 20)
        print(f"   {plant:12} {confidence:6.2%} {bar}")
    
    return result


if __name__ == "__main__":
    main()
    
    # UNCOMMENT THE FOLLOWING TO TRAIN THE MODEL:
    # images_dir = 'd:\\Facultate\\Anul III\\RN\\plant_images'
    # model_handler, history = train_model(images_dir, epochs=20)
    # model_handler.save_model('plant_model.h5')
    
    # UNCOMMENT TO MAKE PREDICTIONS:
    # model_handler = PlantIdentificationModel()
    # model_handler.load_model('plant_model.h5')
    # result = predict_on_image(model_handler, 'path/to/image.jpg')
