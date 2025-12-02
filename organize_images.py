"""
Simple dataset organizer for plant images
Helps organize images you already have or download from reliable sources
"""

import os
import shutil
from pathlib import Path

def create_plant_folders():
    """Create the directory structure for plant images"""
    base_dir = Path("d:\\Facultate\\Anul III\\RN\\plant_images")
    
    plants = ['rose', 'sunflower', 'orchid', 'tulip', 'cactus']
    
    for plant in plants:
        plant_dir = base_dir / plant
        plant_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created folder: {plant_dir}")
    
    print(f"\n✓ All plant folders created at: {base_dir}")
    print("\n📋 NEXT STEPS:")
    print("""
1. Download plant images from Kaggle:
   https://www.kaggle.com/datasets/
   
2. Extract images into the folders:
   - rose_images.zip → plant_images/rose/
   - sunflower_images.zip → plant_images/sunflower/
   - orchid_images.zip → plant_images/orchid/
   - tulip_images.zip → plant_images/tulip/
   - cactus_images.zip → plant_images/cactus/

3. Make sure each folder has at least 10-20 images

4. Run training_script.py

NOTE: For testing, the system will create synthetic images automatically
if it doesn't find real images.
    """)


def organize_images_from_flat_folder(source_folder, keyword_mapping=None):
    """
    Organize images from a flat folder into plant-type subfolders
    
    Args:
        source_folder: Path to folder with all images mixed together
        keyword_mapping: Dict mapping plant names to keywords in filenames
                        e.g. {'rose': ['rose', 'red_flower'], 'sunflower': ['sunflower', 'yellow']}
    """
    if keyword_mapping is None:
        keyword_mapping = {
            'rose': ['rose', 'rosa'],
            'sunflower': ['sunflower', 'girasol'],
            'orchid': ['orchid', 'orchidea'],
            'tulip': ['tulip', 'tulipa'],
            'cactus': ['cactus', 'cacti']
        }
    
    source_path = Path(source_folder)
    base_dir = Path("d:\\Facultate\\Anul III\\RN\\plant_images")
    
    if not source_path.exists():
        print(f"❌ Source folder not found: {source_folder}")
        return
    
    print(f"📂 Organizing images from: {source_path}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    moved_count = 0
    
    for image_file in source_path.iterdir():
        if image_file.suffix.lower() not in image_extensions:
            continue
        
        filename_lower = image_file.name.lower()
        plant_type = None
        
        # Check filename for plant keywords
        for plant, keywords in keyword_mapping.items():
            if any(keyword in filename_lower for keyword in keywords):
                plant_type = plant
                break
        
        if plant_type:
            dest_folder = base_dir / plant_type
            dest_folder.mkdir(parents=True, exist_ok=True)
            dest_path = dest_folder / image_file.name
            
            shutil.copy2(image_file, dest_path)
            moved_count += 1
            print(f"   ✓ {image_file.name} → {plant_type}/")
        else:
            print(f"   ⊘ {image_file.name} (no plant match)")
    
    print(f"\n✓ Moved {moved_count} images to plant folders")


if __name__ == "__main__":
    print("🌿 PLANT IMAGE ORGANIZER 🌿\n")
    
    create_plant_folders()
    
    print("\n" + "="*60)
    print("MANUAL SETUP OPTION:")
    print("="*60)
    print("""
To download plant images, visit:
- Kaggle: https://www.kaggle.com/search?q=plants
- Google Images: https://images.google.com/
- Unsplash: https://unsplash.com/
- Pixabay: https://pixabay.com/

Download 20-50 images per plant type and save them to:
  d:\\Facultate\\Anul III\\RN\\plant_images\\rose\\
  d:\\Facultate\\Anul III\\RN\\plant_images\\sunflower\\
  etc.
    """)
