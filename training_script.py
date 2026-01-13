"""
Training script for plant identification model
Run this to train the model on your plant images
"""

import sys
from pathlib import Path

# Add project directory to path
sys.path.insert(0, str(Path(__name__).parent))

from proiect_rn import train_model, PlantIdentificationModel

if __name__ == "__main__":
    images_dir = 'd:\\Facultate\\Anul III\\RN\\plant_images'
    
    print("🌿 PLANT IDENTIFICATION - TRAINING SCRIPT 🌿\n")
    
    # Train the model
    model_handler, history = train_model(images_dir, model_name='plant_model.h5', epochs=20)
    
    if model_handler:
        print("\n✓ Training complete!")
        print("✓ Model saved as: plant_model.h5")
        print("\n📚 Next step: Use predict.py to make predictions on new images")
