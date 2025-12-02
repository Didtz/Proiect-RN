"""
Script to download plant datasets
Choose your preferred dataset and it will be organized automatically
"""

import os
import urllib.request
import zipfile
from pathlib import Path

def download_oxford_flowers():
    """Download Oxford Flowers 102 dataset"""
    print("\n📥 Downloading Oxford Flowers Dataset (102 categories, ~400MB)...")
    print("   This is a high-quality flower dataset")
    
    base_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    downloads_dir = Path.home() / "Downloads"
    
    # Download image set
    image_file = downloads_dir / "102flowers.tgz"
    labels_file = downloads_dir / "imagelabels.mat"
    setid_file = downloads_dir / "setid.mat"
    
    print("\n   Step 1: Downloading images (this may take a few minutes)...")
    try:
        urllib.request.urlretrieve(base_url + "102flowers.tgz", str(image_file))
        print("   ✓ Images downloaded")
    except Exception as e:
        print(f"   ❌ Error downloading images: {e}")
        return False
    
    print("   Step 2: Downloading labels...")
    try:
        urllib.request.urlretrieve(base_url + "imagelabels.mat", str(labels_file))
        urllib.request.urlretrieve(base_url + "setid.mat", str(setid_file))
        print("   ✓ Labels downloaded")
    except Exception as e:
        print(f"   ❌ Error downloading labels: {e}")
        return False
    
    print("\n✓ Dataset downloaded to Downloads folder")
    print("   Files ready for training!")
    return True


def setup_plant_images_structure():
    """
    Setup the directory structure for plant images
    Instructions for organizing downloaded datasets
    """
    print("\n📂 RECOMMENDED FOLDER STRUCTURE:")
    print("""
    d:\\Facultate\\Anul III\\RN\\plant_images\\
    ├── rose/
    │   ├── rose_001.jpg
    │   ├── rose_002.jpg
    │   └── ...
    ├── sunflower/
    │   ├── sunflower_001.jpg
    │   └── ...
    ├── orchid/
    ├── tulip/
    └── cactus/
    """)
    
    print("\n📋 STEPS TO ORGANIZE YOUR DATASET:")
    print("""
    1. Download a dataset from:
       - Oxford Flowers: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/
       - Plant Village: https://github.com/spMohanty/PlantVillage-Dataset
       - Kaggle Plants: https://www.kaggle.com/datasets/
    
    2. Extract the archive
    
    3. Create subfolders in d:\\Facultate\\Anul III\\RN\\plant_images\\
       for each plant type (rose, sunflower, orchid, tulip, cactus)
    
    4. Move images into corresponding plant folders
    
    5. Run training_script.py
    """)


if __name__ == "__main__":
    print("🌿 PLANT IMAGE DATASET SETUP 🌿\n")
    
    print("Available options:")
    print("1. Download Oxford Flowers Dataset (102 flower types, high quality)")
    print("2. Manual setup with your own images")
    print("3. Use sample synthetic images for testing")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        download_oxford_flowers()
    elif choice == "2":
        setup_plant_images_structure()
    elif choice == "3":
        print("\nSynthetic images will be created automatically when training starts.")
    else:
        print("Invalid choice!")
