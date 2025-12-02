"""
Prediction script for plant identification
Use this to identify plants from images
"""

import sys
from pathlib import Path

# Add project directory to path
sys.path.insert(0, str(Path(__file__).parent))

from proiect_rn import PlantIdentificationModel, predict_on_image

if __name__ == "__main__":
    # Load the trained model
    print("🌿 PLANT IDENTIFICATION - PREDICTION 🌿\n")
    
    model_handler = PlantIdentificationModel()
    
    try:
        model_handler.load_model('plant_model.h5')
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Please train the model first using: train.py")
        sys.exit(1)
    
    # Get image path from user
    image_path = input("\n📸 Enter the path to your plant image: ").strip()
    
    # Make prediction
    result = predict_on_image(model_handler, image_path)
