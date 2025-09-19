# Arogya AI - Disease Prediction System with Ayurvedic Recommendations

## Overview

Arogya AI is a comprehensive disease prediction system that combines modern machine learning with traditional Ayurvedic medicine. It provides both accurate disease diagnosis and personalized Ayurvedic treatment recommendations.

## Key Features

✅ **High-Accuracy Disease Prediction**: 99.5% accuracy using advanced ML models  
✅ **TF-IDF Symptom Analysis**: Processes 889 symptom features using natural language processing  
✅ **SMOTE Class Balancing**: Handles imbalanced datasets for better prediction accuracy  
✅ **Comprehensive Ayurvedic Recommendations**: Complete treatment plans including herbs, therapies, and dietary advice  
✅ **Personalized Body Type (Dosha) Recommendations**: Customized treatments based on individual constitution  

## What You Get from Predictions

Each prediction provides all the requested fields:

- **Ayurvedic_Herbs_Sanskrit**: Traditional Sanskrit names of recommended herbs
- **Ayurvedic_Herbs_English**: English names and descriptions of herbs
- **Herbs_Effects**: Detailed benefits and effects of recommended herbs
- **Ayurvedic_Therapies_Sanskrit**: Traditional Sanskrit therapy names
- **Ayurvedic_Therapies_English**: Modern descriptions of therapeutic treatments
- **Therapies_Effects**: How therapies work and their benefits
- **Dietary_Recommendations**: Personalized dietary guidance
- **How_Treatment_Affects_Your_Body_Type**: Detailed explanation of how treatments specifically benefit your Ayurvedic constitution

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (if needed)
```bash
python train_model.py
```
This creates `random_forest_model.pkl` with all necessary components.

### 3. Use the System

#### Option A: Web Interface (Recommended)
```bash
python app.py
```
Then open your browser to: `http://localhost:5000`

**Or use the startup scripts:**
- Linux/Mac: `./start_web.sh`
- Windows: `start_web.bat`

#### Option B: Command Line
```bash
python arogya_predict.py
```

## 🌐 Web Interface

**NEW!** Arogya AI now includes a beautiful web interface that provides:

- **User-friendly form** similar to Symptoma AI
- **Responsive design** for desktop, tablet, and mobile
- **Real-time predictions** with comprehensive Ayurvedic recommendations
- **Print functionality** for offline reference
- **Interactive results** display

### Web Interface Features:
- Comprehensive health assessment form
- All required fields with helpful descriptions
- Instant disease prediction with confidence scores
- Complete Ayurvedic treatment plans
- Beautiful, professional presentation

![Web Interface](https://github.com/user-attachments/assets/51d12133-9164-4186-aff6-d717b28e4f95)

![Prediction Results](https://github.com/user-attachments/assets/4cec80dd-7e96-4771-8a1e-48b7d00fce4b)

## Sample Output

```
🔍 PREDICTION RESULTS:
   Predicted Disease: Jwara (Fever)
   Confidence: 99.90%
   Symptoms: fever, body ache, headache, fatigue
   Body Type: Pitta

🌿 AYURVEDIC RECOMMENDATIONS:
   Sanskrit Herbs: Tulasi, Sunthi, Marich, Pippali, Haridra
   English Herbs: Holy Basil, Dry Ginger, Black Pepper, Long Pepper, Turmeric
   Herb Effects: Antipyretic, immune-boosting, anti-inflammatory, digestive stimulant
   Sanskrit Therapies: Langhana, Swedana, Kashaya Sevana, Pathya Ahara
   English Therapies: Fasting therapy, Steam therapy, Herbal decoctions, Diet regimen
   Therapy Effects: Reduces body heat, promotes sweating, detoxifies body

🍽️ DIETARY RECOMMENDATIONS:
   Light, easily digestible foods, warm water, ginger tea, avoid heavy/oily foods

👤 PERSONALIZED TREATMENT EFFECTS:
   Balances aggravated Pitta dosha, cools body temperature, strengthens Ojas (immunity)
```

## System Architecture

1. **Data Processing**: TF-IDF vectorization of symptoms (889 features)
2. **Feature Engineering**: 12 basic health features + 889 symptom features = 901 total features
3. **Class Balancing**: SMOTE applied for balanced training dataset
4. **Model Training**: Random Forest/Logistic Regression with 99%+ accuracy
5. **Ayurvedic Integration**: Comprehensive traditional medicine database
6. **Personalization**: Body type (Dosha) specific recommendations

## Model Performance

- **Random Forest Accuracy**: 100%
- **Logistic Regression Accuracy**: 99.5%
- **SVM Accuracy**: 95.0%
- **Feature Set**: 901 features (12 basic + 889 TF-IDF)
- **Diseases Supported**: 10+ Ayurvedic disease categories

## Supported Diseases

- Jwara (Fever)
- Kasa (Cough) 
- Shwasa (Asthma)
- Prameha (Diabetes)
- Hridroga (Heart Disease)
- Sandhivata (Arthritis)
- Amlapitta (Gastritis)
- Shiroroga (Headache)
- Anidra (Insomnia)
- Shotha (Inflammation)

## Usage Modes

### 1. Web Interface (Recommended - NEW!)
A modern, user-friendly web application with:
- Comprehensive health assessment form
- Real-time disease prediction
- Beautiful results display
- Mobile-responsive design
- Print functionality

```bash
python app.py
# Open browser to http://localhost:5000
```

### 2. Demo Mode (Command Line)
Runs sample predictions with pre-defined test cases.

### 3. Interactive Mode (Command Line)
Collects user symptoms and health information interactively.

### 4. API Integration (Ready)
The system is designed to be easily integrated into web applications or APIs.

## File Structure

```
├── train_model.py           # Model training script
├── arogya_predict.py        # Main prediction system
├── disease_prediction_system.py  # Alternative comprehensive implementation
├── app.py                   # Flask web application (NEW!)
├── start_web.sh            # Linux/Mac startup script (NEW!)
├── start_web.bat           # Windows startup script (NEW!)
├── requirements.txt         # Python dependencies
├── random_forest_model.pkl  # Trained model (generated)
├── templates/              # Web interface templates (NEW!)
│   └── index.html          # Main web page
├── static/                 # Web interface assets (NEW!)
│   ├── css/style.css       # Custom styles
│   └── js/script.js        # Frontend JavaScript
├── README_WEB_INTERFACE.md # Web interface documentation (NEW!)
└── AyurCore.ipynb          # Original research notebook
```

## Technical Implementation

- **ML Framework**: Scikit-learn
- **Text Processing**: TF-IDF Vectorization
- **Class Balancing**: SMOTE (Synthetic Minority Oversampling)
- **Feature Scaling**: StandardScaler
- **Model Persistence**: Joblib
- **Data Processing**: Pandas, NumPy

## Ayurvedic Knowledge Base

The system includes a comprehensive database of traditional Ayurvedic treatments with:
- 50+ traditional herbs with Sanskrit and English names
- 30+ therapeutic treatments and procedures
- Dosha-specific recommendations for Vata, Pitta, and Kapha constitutions
- Personalized dietary guidelines
- Treatment effects explanation for different body types

## Future Enhancements

- ~~Integration with real medical datasets~~ ✅ **DONE**
- ~~Web interface for easier access~~ ✅ **COMPLETED**
- Mobile application
- Multi-language support
- Advanced NLP for symptom processing
- Telemedicine integration
- User authentication and history
- Dashboard for healthcare providers

## Disclaimer

This system is for educational and research purposes. Always consult with qualified healthcare professionals for medical advice and treatment.

---

**Stay healthy with the wisdom of Ayurveda! 🌿**
