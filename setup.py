"""
Quick setup script to prepare the environment
"""

import os
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    dirs = ['models', 'training_data', 'uploads', 'templates', 'static']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        logger.info(f"✓ Created {dir_name}/ directory")

def install_requirements():
    """Install Python dependencies"""
    logger.info("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '-q'])
        logger.info("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        logger.error("✗ Failed to install dependencies")
        return False

def main():
    logger.info("=" * 50)
    logger.info("Plant Identifier - Setup")
    logger.info("=" * 50)
    
    # Create directories
    logger.info("\n📁 Creating directories...")
    create_directories()
    
    # Install requirements
    if not install_requirements():
        logger.error("\nSetup failed. Please install dependencies manually:")
        logger.error("  pip install -r requirements.txt")
        sys.exit(1)
    
    logger.info("\n" + "=" * 50)
    logger.info("Setup completed! ✓")
    logger.info("=" * 50)
    logger.info("\nNext steps:")
    logger.info("1. Train the model:")
    logger.info("   python train_model.py")
    logger.info("\n2. Run the app:")
    logger.info("   python app_improved.py")
    logger.info("\n3. Open browser:")
    logger.info("   http://localhost:5000")
    logger.info("=" * 50)

if __name__ == '__main__':
    main()
