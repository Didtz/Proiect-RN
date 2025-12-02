"""
Setup single plant images folder with sample data
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

def setup_single_folder():
    """Create a single organized folder with sample plant images"""
    
    base_dir = Path("d:\\Facultate\\Anul III\\RN\\plant_images")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Plant categories
    plants = ['rose', 'sunflower', 'orchid', 'tulip', 'cactus']
    
    print("🌿 Setting up plant images folder...\n")
    
    # Create subfolders for each plant
    for plant_name in plants:
        plant_dir = base_dir / plant_name
        plant_dir.mkdir(exist_ok=True)
        
        # Create 30 sample images per plant type
        print(f"📸 Generating {plant_name} images...")
        for i in range(30):
            # Generate realistic green-ish image
            img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            
            # Add plant-specific color adjustments
            if plant_name == 'rose':
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.3, 0, 255).astype(np.uint8)  # More red
                img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 0.7, 0, 255).astype(np.uint8)  # Less green
            elif plant_name == 'sunflower':
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.2, 0, 255).astype(np.uint8)  # Red/yellow
                img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.3, 0, 255).astype(np.uint8)  # More green
                img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.5, 0, 255).astype(np.uint8)  # Less blue
            elif plant_name == 'orchid':
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.4, 0, 255).astype(np.uint8)  # Purple/pink
            elif plant_name == 'tulip':
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.2, 0, 255).astype(np.uint8)  # Red/orange
            elif plant_name == 'cactus':
                img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.2, 0, 255).astype(np.uint8)  # Green
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 0.8, 0, 255).astype(np.uint8)  # Less red
            
            # Convert to image and save
            img = keras.preprocessing.image.array_to_img(img_array)
            img_path = plant_dir / f"{plant_name}_{i:03d}.jpg"
            img.save(str(img_path))
        
        print(f"   ✓ Created 30 {plant_name} images")
    
    print(f"\n✓ Plant images folder ready!")
    print(f"Location: {base_dir}")
    print("\nFolder structure:")
    for plant in plants:
        plant_dir = base_dir / plant
        img_count = len(list(plant_dir.glob('*.jpg')))
        print(f"  {plant}/: {img_count} images")
    
    return str(base_dir)


if __name__ == "__main__":
    setup_single_folder()
    print("\n✓ Ready to train! Run: python training_script.py")
