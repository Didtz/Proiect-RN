"""
Download and use real plant/flower datasets from TensorFlow
Uses CIFAR10 as training data for image classification
"""

import tensorflow as tf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from proiect_rn import PlantIdentificationModel

def download_cifar10_dataset():
    """
    Download CIFAR10 dataset from TensorFlow
    60,000 images in 10 categories (32x32 pixels)
    """
    print("\n📥 Downloading CIFAR10 dataset from TensorFlow...\n")
    
    try:
        # Load CIFAR10 dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        print(f"✓ Dataset downloaded successfully!")
        print(f"  Training images: {len(x_train)}")
        print(f"  Test images: {len(x_test)}")
        print(f"  Image size: {x_train[0].shape}")
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Flatten labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        print("\n✓ Data normalized and ready!")
        
        return x_train, y_train, x_test, y_test
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None, None, None, None


def train_with_cifar10():
    """Train model with CIFAR10 dataset"""
    print("\n🌿 PLANT IDENTIFICATION - Training with CIFAR10 Dataset\n")
    
    # Download dataset
    x_train, y_train, x_test, y_test = download_cifar10_dataset()
    
    if x_train is None:
        print("Could not load dataset")
        return
    
    # Use only a subset to avoid memory issues (10,000 training images instead of 50,000)
    print("\n📊 Using a subset of data to fit in memory...")
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    x_test = x_test[:2000]
    y_test = y_test[:2000]
    print(f"   Training samples: {len(x_train)}")
    print(f"   Test samples: {len(x_test)}")
    
    # Create data generator for on-the-fly resizing (memory efficient)
    print("\n📦 Creating data generator for efficient memory usage...")
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1
    )
    
    # Build model with smaller batch size
    print("\n🤖 Building model...")
    model_handler = PlantIdentificationModel(num_classes=10)
    model_handler.build_model()
    
    print(f"✓ Model ready with {model_handler.model.count_params():,} parameters")
    
    # Train model with smaller batch size
    print(f"\n🚀 Starting training for 3 epochs (reduced to save memory)...")
    print("   (This may take several minutes)\n")
    
    # Use flow method for batch processing
    train_gen = train_datagen.flow(x_train, y_train, batch_size=16, shuffle=True)
    test_gen = train_datagen.flow(x_test, y_test, batch_size=16, shuffle=False)
    
    history = model_handler.model.fit(
        train_gen,
        validation_data=test_gen,
        steps_per_epoch=len(x_train) // 16,
        validation_steps=len(x_test) // 16,
        epochs=3,
        verbose=1
    )
    
    print("\n✓ Training complete!")
    print("💾 Saving model...")
    model_handler.save_model('plant_model_cifar10.h5')
    
    return model_handler, history


def list_available_datasets():
    """List TensorFlow datasets available"""
    print("\n📚 Available TensorFlow Datasets:")
    print("""
    Image Classification Datasets:
    - CIFAR10: 60,000 32x32 images (10 categories)
    - CIFAR100: 60,000 32x32 images (100 categories)
    - MNIST: 70,000 28x28 grayscale images (10 digits)
    - Fashion MNIST: 70,000 28x28 grayscale images (10 categories)
    - ImageNet: Large-scale visual recognition dataset
    
    Plant/Flower Specific:
    - TensorFlow Flowers: flower classification
    - Plant Village: plant disease dataset
    
    This script uses CIFAR10 as a proxy for plant image classification.
    For actual plant images, download from:
    - Kaggle: https://www.kaggle.com/datasets/
    - iNaturalist: https://www.inaturalist.org/
    - Plant Village: https://github.com/spMohanty/PlantVillage-Dataset
    """)


if __name__ == "__main__":
    print("🌿 TENSORFLOW DATASETS INTEGRATION 🌿\n")
    
    print("Options:")
    print("1. Download CIFAR10 and train model")
    print("2. List available datasets")
    print("3. Use local images instead")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        train_with_cifar10()
    elif choice == "2":
        list_available_datasets()
    elif choice == "3":
        print("\nRun: python setup_images.py")
        print("Then: python training_script.py")
    else:
        print("Invalid choice!")
