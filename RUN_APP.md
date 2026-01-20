# 🌿 Plant Identifier Application - Quick Start Guide

## Overview
This is a complete plant identification web application with the following features:
- ✅ **Photo Upload**: Upload images of plants to identify them
- ✅ **AI Identification**: Uses deep learning (TensorFlow/Keras) to identify plants
- ✅ **Comparison Photos**: Shows reference images from the training dataset
- ✅ **Plant Information**: Displays common name, scientific name, and care instructions
- ✅ **Confidence Score**: Shows how confident the model is about the identification

## Installation & Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model (if not already trained)
If you don't have a trained model yet:
```bash
python train_model.py
```

This will create:
- `models/plant_model.h5` - The trained neural network
- `models/plant_model_metadata.json` - Model metadata

**Training takes approximately 10-15 minutes** depending on your system.

### Step 3: Run the Web Application
```bash
python app_improved.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
```

### Step 4: Open in Browser
Open your web browser and navigate to:
```
http://localhost:5000
```

## Using the Application

### Identify a Plant
1. Click the **"Identify"** link in the navigation menu
2. Click the upload area or drag & drop a plant image
3. Review the preview and click **"Identify Plant"**
4. View results:
   - **Your Photo**: The image you uploaded
   - **Reference Sample**: A comparison photo from the training dataset
   - **Plant Information**: Common name, scientific name, care instructions
   - **Confidence Score**: Percentage confidence in the identification

### Other Features
- **Home**: Overview of the application
- **Database**: Browse all plants in the database
- **Care Guide**: View detailed care instructions for each plant type

## Supported Plant Types
The model can identify the following plants:
- 🌹 **Rose** - Rosa spp.
- 🌻 **Sunflower** - Helianthus annuus
- 🌷 **Tulip** - Tulipa spp.
- 🌵 **Cactus** - Cactaceae
- 🪴 **Orchid** - Orchidaceae

## Troubleshooting

### Model Not Found
If you see "Model not trained yet" error:
```bash
python train_model.py
```

### Port Already in Use
If port 5000 is already used, modify `app_improved.py`:
```python
app.run(debug=True, port=5001)  # Change 5001 to any available port
```

### Image Upload Issues
- Use JPG, PNG, GIF formats
- Maximum file size: 5MB
- Ensure good lighting in the photo for better results

### Low Confidence Results
- Try different angles of the plant
- Ensure the plant is clearly visible
- Use good lighting conditions
- The model requires at least 30% confidence to make a prediction

## File Structure
```
├── app_improved.py          # Main Flask application
├── models/
│   ├── plant_model.h5       # Trained model
│   └── plant_model_metadata.json
├── training_data/           # Training images (used for comparison photos)
│   ├── rose/
│   ├── sunflower/
│   ├── tulip/
│   ├── cactus/
│   └── orchid/
├── uploads/                 # Temporary upload folder
├── static/
│   └── style.css           # Styling
├── templates/
│   ├── identify.html       # Main identification page
│   ├── index.html          # Home page
│   ├── database.html       # Plant database page
│   └── guide.html          # Care guide page
└── requirements.txt        # Python dependencies
```

## Performance Notes
- **First prediction**: May take 3-5 seconds (model loading)
- **Subsequent predictions**: 1-2 seconds
- **GPU acceleration**: If you have CUDA-capable GPU, TensorFlow will automatically use it for faster predictions

## Features in Detail

### Comparison Photos
The app automatically retrieves a random plant image from the training dataset (`training_data/[plant_name]/`) to show as a reference. This helps users verify if the identification looks correct.

### Confidence Score
- **80%+**: Very confident identification
- **50-80%**: Moderate confidence, check the comparison photo
- **30-50%**: Low confidence, result may be incorrect

### Plant Information Provided
For each identified plant:
- Common name (e.g., "Rose")
- Scientific name (e.g., "Rosa spp.")
- Care guide with:
  - Watering schedule
  - Light requirements
  - Difficulty level
  - Description

## Advanced Usage

### Retrain Model with New Data
1. Add new plant images to `training_data/[plant_name]/`
2. Run `train_model.py` again
3. The app will use the updated model

### Customize Plant Information
Edit the `PLANT_DATABASE` dictionary in `app_improved.py` to add/modify plant information.

### Change Confidence Threshold
In `app_improved.py`, modify this line:
```python
if confidence < 0.3:  # Change 0.3 to any value between 0 and 1
```

## API Endpoints (for developers)

```
POST /api/predict              - Upload image and get prediction
GET  /api/plants               - Get all plants
GET  /api/plant/<name>         - Get specific plant info
GET  /api/model/status         - Get model status
GET  /api/categories           - Get plant categories
```

## Support & Issues
For issues or questions, check that:
1. All dependencies are installed: `pip list`
2. Model file exists: `models/plant_model.h5` (run training if missing)
3. Port 5000 is available: `netstat -ano | findstr :5000` (Windows) or `lsof -i :5000` (Mac/Linux)
4. Python version is 3.8 or higher: `python --version`

## License
This project uses:
- TensorFlow (Apache 2.0)
- Flask (BSD)
- Keras (MIT)

Enjoy identifying plants! 🌿
