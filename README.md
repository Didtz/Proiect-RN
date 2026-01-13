# Plant Identifier - Web Application

A modern Flask web application for plant identification using deep learning with TensorFlow.

## Features

- 🌿 **Plant Identification**: Upload images to identify plant species
- 📚 **Plant Database**: Browse and search plant information
- 💧 **Care Guides**: Get detailed care tips for each plant
- 🎨 **Clean UI**: Simple black and white minimalist design
- 🤖 **Deep Learning**: Uses MobileNetV2 for accurate classification

## Project Structure

```
.
├── app_improved.py              # Main Flask application
├── train_model.py               # Model training script
├── templates/                   # HTML templates
│   ├── index.html              # Home page
│   ├── identify.html           # Plant identification page
│   ├── database.html           # Plant database
│   └── guide.html              # Care guide
├── static/
│   └── style.css               # Minimalist black and white styles
├── models/                      # Trained models (created during training)
├── training_data/              # Training dataset (created during training)
└── uploads/                    # Temporary uploaded images
```

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Required Packages

The application requires:
- Flask
- TensorFlow
- Keras
- NumPy
- Pillow
- matplotlib (for training plots)

## Quick Start

### Step 1: Train the Model

Before running the web app, you need to train the model:

```bash
python train_model.py
```

This script will:
1. Create synthetic training data in the `training_data` folder
2. Build the MobileNetV2 model with custom layers
3. Train the model (20 epochs by default)
4. Evaluate performance
5. Save the trained model to `models/plant_model.h5`

**Training takes 5-15 minutes depending on your hardware.**

### Step 2: Run the Web Application

```bash
python app_improved.py
```

The application will start at: **http://localhost:5000**

## Usage

### Home Page
Access the main page with feature overview and quick links.

### Identify Page
1. Click "Identify" in the navigation
2. Drag and drop or click to upload a plant image
3. The model will analyze and display:
   - Plant name
   - Confidence percentage
   - Scientific information
   - Care instructions

### Database
Browse all 5 plant types with their characteristics:
- Rose
- Sunflower
- Tulip
- Cactus
- Orchid

### Care Guide
Get comprehensive plant care tips including:
- Watering requirements
- Light needs
- Humidity preferences
- Common problems and solutions

## Configuration

Edit configuration in `train_model.py`:

```python
CONFIG = {
    'image_size': (224, 224),      # Input image size
    'batch_size': 32,               # Training batch size
    'epochs': 20,                   # Number of training epochs
    'learning_rate': 0.001,         # Model learning rate
    'validation_split': 0.2,        # Validation data percentage
}
```

## API Endpoints

### Prediction
- **POST** `/api/predict` - Upload image for plant identification
  - Form data: `file` (image file)
  - Returns: Plant name, confidence, and information

### Database
- **GET** `/api/plants` - Get all plants
- **GET** `/api/plant/<name>` - Get specific plant info
- **GET** `/api/categories` - Get plant categories

### Model Status
- **GET** `/api/model/status` - Check if model is trained

## Browser Compatibility

- Chrome/Edge (Recommended)
- Firefox
- Safari
- Mobile browsers

## Performance

The application uses MobileNetV2, which provides:
- Fast inference (< 1 second)
- Lightweight model (~100 MB)
- Good accuracy for plant classification

## Troubleshooting

### Model not found error
**Solution**: Run `train_model.py` first to train and save the model

### Port already in use
**Solution**: Change port in `app_improved.py`:
```python
app.run(debug=True, port=5001, host='0.0.0.0')
```

### Image upload fails
**Solution**: Check file size (max 5MB) and format (PNG, JPG, JPEG, GIF)

### Slow predictions
**Solution**: This is normal for the first prediction (model loading). Subsequent predictions are faster.

## Design Features

- **Minimalist Black & White**: Clean, professional design
- **Responsive Layout**: Works on desktop and mobile
- **Intuitive Navigation**: Easy to find features
- **Fast Loading**: Optimized CSS and minimal images
- **Drag & Drop**: Convenient file upload interface

## Model Training Details

### Architecture
```
Input (224x224x3)
    ↓
Rescaling (1./255)
    ↓
MobileNetV2 Base (frozen)
    ↓
Global Average Pooling
    ↓
Dense (256, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.3)
    ↓
Output (5 classes, Softmax)
```

### Training Features
- **Data Augmentation**: Rotation, zoom, shift, flip
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Reduces learning rate if validation plateaus
- **Model Checkpointing**: Saves best model weights

## Next Steps

To improve accuracy:
1. **Add Real Data**: Replace synthetic data with actual plant images
2. **Increase Classes**: Add more plant types
3. **Fine-tune**: Unfreeze base model layers for transfer learning
4. **Data Augmentation**: Use more aggressive augmentation strategies

## License

This project is created for educational purposes.

## Author

Built with TensorFlow, Keras, and Flask
