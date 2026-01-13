"""
Train the plant identification model using Oxford Flowers dataset
This script downloads and trains a CNN model for plant classification
"""

import os
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'image_size': (224, 224),
    'batch_size': 32,
    'epochs': 15,
    'learning_rate': 0.001,
    'validation_split': 0.2,
}

# Oxford Flowers has 102 flower classes
NUM_CLASSES = 102

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('training_data', exist_ok=True)

class PlantModelTrainer:
    """Train plant identification model"""

    def __init__(self):
        self.model = None
        self.class_names = list(range(102))  # Oxford Flowers has 102 classes
        self.history = None

    def build_model(self, num_classes=NUM_CLASSES):
        """Build transfer learning model using MobileNetV2"""
        logger.info("Building model...")
        
        base_model = MobileNetV2(
            input_shape=(*CONFIG['image_size'], 3),
            include_top=False,
            weights='imagenet'
        )

        # Freeze base model layers
        base_model.trainable = False

        # Build custom model
        model = keras.Sequential([
            layers.Input(shape=(*CONFIG['image_size'], 3)),
            layers.Rescaling(1./255),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        # Compile model
        optimizer = Adam(learning_rate=CONFIG['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        logger.info("Model built successfully")
        return model

    def create_synthetic_data(self):
        """Download Oxford Flowers dataset from TensorFlow Datasets"""
        logger.info("Downloading Oxford Flowers dataset...")
        
        try:
            # Download Oxford Flowers 102 dataset
            self.train_data, self.test_data = tfds.load(
                'oxford_flowers102',
                split=['train', 'test'],
                as_supervised=True,
                with_info=False,
                download=True
            )
            
            logger.info("Dataset downloaded successfully")
            
            # Count images
            train_count = sum(1 for _ in self.train_data)
            test_count = sum(1 for _ in self.test_data)
            logger.info(f"Training images: {train_count}")
            logger.info(f"Test images: {test_count}")
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            logger.info("Make sure you have internet connection and sufficient disk space")
            raise

    def train(self):
        """Train the model"""
        if self.model is None:
            self.build_model()

        logger.info("Preparing Oxford Flowers dataset...")
        self.prepare_data()

        logger.info(f"Starting training for {CONFIG['epochs']} epochs...")

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train model
        self.history = self.model.fit(
            self.train_data,
            epochs=CONFIG['epochs'],
            validation_data=self.test_data,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Training completed")
        return self.history

    def prepare_data(self):
        """Prepare and preprocess Oxford Flowers dataset"""
        logger.info("Preparing dataset for training...")
        
        def preprocess_image(image, label):
            # Resize image
            image = tf.image.resize(image, CONFIG['image_size'])
            # Normalize to [0, 1]
            image = image / 255.0
            return image, label
        
        def augment_image(image, label):
            # Data augmentation
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            return image, label
        
        # Prepare training data
        self.train_data = self.train_data.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        self.train_data = self.train_data.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        self.train_data = self.train_data.batch(CONFIG['batch_size'])
        self.train_data = self.train_data.prefetch(tf.data.AUTOTUNE)
        
        # Prepare validation data
        self.test_data = self.test_data.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        self.test_data = self.test_data.batch(CONFIG['batch_size'])
        self.test_data = self.test_data.prefetch(tf.data.AUTOTUNE)
        
        logger.info("Dataset ready for training")

        logger.info("Training completed")
        return self.history

    def evaluate(self):
        """Evaluate the model on test set"""
        if self.model is None:
            logger.error("Model not trained. Please train first.")
            return None

        logger.info("Evaluating model on test set...")
        
        # Evaluate
        loss, accuracy = self.model.evaluate(self.test_data, verbose=1)
        logger.info(f"Test Loss: {loss:.4f}")
        logger.info(f"Test Accuracy: {accuracy:.4f}")

        return {'loss': loss, 'accuracy': accuracy}

    def save_model(self, model_path='models/plant_model.h5'):
        """Save the trained model"""
        if self.model is None:
            logger.error("No model to save")
            return False

        try:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save model metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'classes': self.class_names,
                'image_size': CONFIG['image_size'],
                'config': CONFIG
            }
            
            metadata_path = model_path.replace('.h5', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved to {metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def plot_history(self):
        """Plot training history"""
        if self.history is None:
            logger.error("No training history to plot")
            return

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Plot loss
        ax2.plot(self.history.history['loss'], label='Train Loss')
        ax2.plot(self.history.history['val_loss'], label='Val Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=100)
        logger.info("Training history plot saved to models/training_history.png")
        plt.show()


def main():
    """Main training function"""
    logger.info("=" * 50)
    logger.info("Plant Identification Model Training")
    logger.info("Using Oxford Flowers Dataset")
    logger.info("=" * 50)

    # Initialize trainer
    trainer = PlantModelTrainer()

    # Download dataset
    logger.info("\nStep 1: Downloading Oxford Flowers dataset...")
    trainer.create_synthetic_data()

    # Build model
    logger.info("\nStep 2: Building model architecture...")
    trainer.build_model()

    # Train model
    logger.info("\nStep 3: Training the model...")
    trainer.train()

    # Evaluate model
    logger.info("\nStep 4: Evaluating model...")
    trainer.evaluate()

    # Save model
    logger.info("\nStep 5: Saving model...")
    trainer.save_model('models/plant_model.h5')

    # Plot training history
    logger.info("\nStep 6: Generating training plots...")
    try:
        trainer.plot_history()
    except Exception as e:
        logger.warning(f"Could not plot history: {e}")

    logger.info("\n" + "=" * 50)
    logger.info("Training completed successfully!")
    logger.info("Model saved to: models/plant_model.h5")
    logger.info("Dataset: Oxford Flowers 102")
    logger.info("Classes: 102 flower species")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
