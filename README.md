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

---

# PLANT IDENTIFIER - GHID IN LIMBA ROMANA

## Ce a fost construit

O aplicatie web completa pentru identificarea plantelor cu:

- Aplicatie Flask cu interfata web moderna
- Model de invatare automata (TensorFlow + MobileNetV2)
- Baza de date cu 102 specii de flori (Oxford Flowers dataset)
- Interfata utilizator simpla in alb si negru
- Responsive design (mobile, tableta, desktop)
- Pipeline complet de antrenare a modelului

## Pagini disponibile

**Pagina de Start (/)** - Prezentare aplicatie si features principale

**Identificare Plante (/identify)** - Upload imagine prin drag and drop, predictie instantanea, afisare incredere

**Baza de Date (/database)** - Vizualizare 102 specii de flori cu informatii detaliate

**Ghid Ingrijire (/guide)** - Sfaturi pentru ingrijirea plantelor, cerinte de apa si lumina

## Cum se foloseste

### Pasul 1: Instalare dependente
```bash
pip install -r requirements.txt
```

### Pasul 2: Antrenare model
```bash
python train_model.py
```
Aceasta descarca dataset Oxford Flowers 102 si antreneaza modelul (15-30 minute).

### Pasul 3: Pornire aplicatie
```bash
python app_improved.py
```

### Pasul 4: Deschidere in browser
```
http://localhost:5000
```

## Caracteristici principale

- **Identificare Plantelor**: Incarca imagine, obtine predictie instantanee cu scor incredere
- **Baza de Date Flori**: 102 specii cu nume stiintific, descrieri, nivel dificultate
- **Ghid Ingrijire**: Cum sa uda, cat de multa lumina, temperatura, umiditate, probleme frecvente
- **Design**: Alb si negru minimalist, interfata simpla, responsive pe toate dispozitivele

## Detalii tehnice

**Model Arhitectura**:
- Baza: MobileNetV2 (ImageNet weights)
- Straturi custom: Dense 256 + Dense 128 + Dense 102
- Intrare: Imagini 224x224 RGB
- Iesire: 102 clase flori cu incredere

**Antrenare**:
- Transfer learning din ImageNet
- Data augmentation (rotatie, zoom, flip)
- Early stopping pentru a preveni overfitting
- Learning rate scheduling adaptat

**Performanta**:
- Dimensiune model: circa 100 MB
- Timp predictie: sub 1 secunda
- Timp incarcare: 2 secunde la inceput
- RAM necesar: circa 2-4 GB

## Probleme si solutii

**Eroare: Model not found**
- Ruleaza intai: python train_model.py

**Eroare: Port 5000 in use**
- Schimba portul in app_improved.py: app.run(debug=True, port=5001)

**Eroare: Import error**
- Instaleaza din nou: pip install -r requirements.txt

**Predictii lente la inceput**
- Normal - modelul se incarca. Predictiile urmatoare sunt mai rapide.

## Structura fisiere

```
Plant Identifier/
- app_improved.py           Aplicatia Flask
- train_model.py            Script antrenare
- requirements.txt          Dependente Python
- templates/
  - index.html              Pagina start
  - identify.html           Identificare
  - database.html           Baza date
  - guide.html              Ghid ingrijire
- static/
  - style.css               Stiluri
- models/
  - plant_model.h5          Model antrenat
```

## Dataset folosit

Oxford Flowers 102:
- 8189 imagini cu flori
- 102 clase (specii)
- Imagini de inalta calitate
- Bine etichetate
- Descarcate automat la antrenare

## Cerinte sistem

**Minim**:
- Python 3.8+
- 4 GB RAM
- 2 GB spatiu disk
- Browser modern

**Recomandat**:
- Python 3.10+
- 8 GB RAM
- 5 GB spatiu disk
- Chrome sau Edge

## Compatibilitate browser

- Chrome: Suportat complet
- Firefox: Suportat complet
- Safari: Suportat complet
- Edge: Suportat complet
- Mobile browsers: Suportat complet (responsive)

## Personalizare

**Adaugare plante noi**: Editeaza PLANT_DATABASE in app_improved.py

**Schimbare culori**: Editeaza static/style.css

**Schimbare parametri antrenare**: Editeaza CONFIG in train_model.py

**Schimbare port**: Editeaza app.run() in app_improved.py

## Pasii urmatori

**Imediat**:
1. Instaleaza dependente
2. Antreneaza modelul
3. Porneste aplicatia
4. Testeaza in browser

**Termen scurt**:
1. Adauga imagini reale de plante
2. Imbunatateste precizia modelului
3. Adauga mai multe specii
4. Personalizeaza design

**Termen lung**:
1. Deplaseaza pe server cloud
2. Adauga autentificare utilizatori
3. Stocheaza predictii in baza date
4. Creeaza aplicatie mobila

---

**Aplicatia este completa si functionala. Deschide http://localhost:5000 in browser si incearca sa incarci imagini cu flori!**
