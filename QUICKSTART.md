# 🌿 Plant Identifier - Quick Start Guide

## 5-Minute Setup

### Step 1: Install Dependencies (2 minutes)
```bash
pip install -r requirements.txt
```

### Step 2: Train Model (5-15 minutes)
```bash
python train_model.py
```

You'll see:
- Creating synthetic training data ✓
- Building model architecture ✓
- Training for 20 epochs (progress bar)
- Evaluating model ✓
- Saving model to `models/plant_model.h5` ✓

### Step 3: Run Application
```bash
python app_improved.py
```

You'll see:
```
* Serving Flask app 'app_improved'
* Debug mode: on
* Running on http://127.0.0.1:5000
```

### Step 4: Open in Browser
Go to: **http://localhost:5000**

## What You Can Do

### 1. Identify Plants 📸
- Click "Identify" tab
- Upload an image
- Get instant plant identification
- See care instructions

### 2. Browse Database 📚
- Click "Database" tab
- See all 5 plant types
- View detailed information
- Check difficulty levels

### 3. Read Care Guide 💧
- Click "Care Guide" tab
- Learn watering, light, temperature
- Get tips for common problems
- Understand propagation methods

## Interface Preview

### Home Page
- Welcome message
- 4 feature cards
- Quick start button

### Identify Page
- Drag-drop upload area
- Image preview
- AI-powered prediction
- Confidence percentage
- Plant care information

### Database
- 5x1 grid of plant cards
- Plant name and difficulty
- Scientific name
- Description and care tips

### Care Guide
- Comprehensive care tips
- Best practices
- Troubleshooting advice
- Organized sections

## Key URLs

- Home: http://localhost:5000/
- Identify: http://localhost:5000/identify
- Database: http://localhost:5000/database
- Care Guide: http://localhost:5000/guide

## API for Developers

### Identify Plant
```bash
curl -X POST -F "file=@image.jpg" \
  http://localhost:5000/api/predict
```

Response:
```json
{
  "success": true,
  "plant": "rose",
  "confidence": 0.95,
  "info": {
    "scientific_name": "Rosa spp.",
    "description": "Beautiful flowering plant",
    "watering": "Water deeply when dry",
    "light": "Full sun (6+ hours)",
    "difficulty": "Intermediate"
  }
}
```

### Get All Plants
```bash
curl http://localhost:5000/api/plants
```

### Model Status
```bash
curl http://localhost:5000/api/model/status
```

## Supported Image Formats

✓ PNG
✓ JPG
✓ JPEG
✓ GIF

Maximum file size: **5 MB**

## System Requirements

- Python 3.8+
- 4GB RAM (minimum)
- 2GB disk space for model + data
- Modern web browser

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "Port 5000 already in use"
Edit `app_improved.py` line 132:
```python
app.run(debug=True, port=5001, host='0.0.0.0')
```

### "Model not found"
Run training first:
```bash
python train_model.py
```

### Slow first prediction
Normal! Model is loading. Subsequent predictions are faster.

## Performance Tips

- Use modern browser (Chrome recommended)
- Upload images <2MB for faster processing
- Keep browser window active during prediction
- Clear browser cache if having issues

## Files Created

After running:

```
✓ models/plant_model.h5          (100 MB - trained model)
✓ training_data/                  (multiple folders with images)
✓ models/training_history.png     (training charts)
```

## Customization

### Change Port
In `app_improved.py`:
```python
app.run(debug=True, port=8000)  # Use 8000 instead
```

### Add More Plants
Edit `PLANT_DATABASE` in `app_improved.py` and add classes to `PLANT_CLASSES` in `train_model.py`

### Change Training Parameters
Edit `CONFIG` in `train_model.py`:
```python
CONFIG = {
    'epochs': 30,          # More epochs for better accuracy
    'batch_size': 16,      # Smaller batch for less memory
    'learning_rate': 0.0005,  # Lower for finer tuning
}
```

## Next Steps

1. ✓ Setup complete
2. ✓ Model trained
3. ✓ App running
4. **Add real plant images** - Replace synthetic data
5. **Deploy to cloud** - Use Heroku, AWS, or Google Cloud
6. **Mobile version** - Create React Native app

## Getting Help

See `README.md` for comprehensive documentation
See `IMPLEMENTATION_SUMMARY.md` for technical details

## Have Fun! 🎉

Your plant identifier is now ready to use!
