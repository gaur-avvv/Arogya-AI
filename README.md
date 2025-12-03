# Arogya AI - Disease Prediction System with Ayurvedic Recommendations

## Overview

Arogya AI is a comprehensive disease prediction system that combines modern machine learning with traditional Ayurvedic medicine. It provides both accurate disease diagnosis and personalized Ayurvedic treatment recommendations.

## Key Features

✅ **High-Accuracy Disease Prediction**: 100% accuracy using Random Forest ML model  
✅ **TF-IDF Symptom Analysis**: Processes 807 symptom features using natural language processing  
✅ **SMOTE Class Balancing**: Handles imbalanced datasets for better prediction accuracy  
✅ **Comprehensive Ayurvedic Recommendations**: Complete treatment plans including herbs, therapies, and dietary advice  
✅ **Personalized Body Type (Dosha) Recommendations**: Customized treatments based on individual constitution  
✅ **Interactive Assessment Mode**: User-friendly symptom and health data collection  
✅ **Integrated ML + LLM System**: Combines machine learning predictions with LLM-powered Ayurvedic analysis  
✅ **Offline Fallback Mode**: Works completely offline when LLM/internet is unavailable  
✅ **Contextual Diagnosis**: Considers all symptoms, lifestyle, and environmental factors for accurate diagnosis  
✅ **Detailed Treatment Plans**: Includes herbs, dietary recommendations, lifestyle advice, and home remedies  
✅ **Privacy-First Design**: Option to run entirely offline with local Ayurvedic database

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

### 3. Run the Model
```bash
python arogya_predict.py
```

### 4. Interactive Mode
For personalized assessment, run the script and choose interactive mode when prompted:
```bash
python arogya_predict.py
```


## Enhanced Features

### 🌿 Comprehensive Dosha Selection
The system now includes a detailed Ayurvedic body type assessment with 6 constitution types:
- **Vata** (Air_Space_Constitution) - Thin/Lean: Naturally thin build, difficulty gaining weight, dry skin, cold hands/feet
- **Pitta** (Fire_Water_Constitution) - Medium: Medium build, good muscle tone, warm body, strong appetite  
- **Kapha** (Earth_Water_Constitution) - Heavy/Large: Naturally larger build, gains weight easily, cool moist skin, steady energy
- **Vata-Pitta** (Air_Fire_Mixed_Constitution) - Thin to Medium: Variable build, creative energy, moderate body temperature
- **Vata-Kapha** (Air_Earth_Mixed_Constitution) - Thin to Heavy: Variable patterns, irregular tendencies, sensitive to changes
- **Pitta-Kapha** (Fire_Earth_Mixed_Constitution) - Medium to Heavy: Strong stable build, good strength, balanced metabolism

### 📊 Calibrated Confidence Scoring
The system now uses sophisticated confidence calibration that considers the gap between the top prediction and second-best prediction to provide more realistic confidence estimates:
- Large gap between predictions: Higher confidence possible (up to 98%)
- Medium gap: Moderate confidence (up to 95%)
- Small gap: Conservative confidence (up to 85%)

### 🤖 Integrated ML + LLM Analysis
The system combines machine learning predictions with advanced LLM-powered contextual analysis. The ML model provides an initial prediction, which the LLM then evaluates against all symptoms, lifestyle factors, and environmental conditions to provide a more accurate, contextualized diagnosis with comprehensive Ayurvedic treatment plans.

## Sample Output

```
============================================================
AROGYA AI - Integrated ML + LLM Prediction System
============================================================
Please provide your details to receive a personalized analysis.
------------------------------------------------------------
Enter your symptoms (comma-separated): joint pain, stiffness, swelling, difficulty walking, morning stiffness
Enter your age: 52
Enter your height (cm): 168
Enter your weight (kg): 78
Enter your gender: Female
Enter your general body type (e.g., Thin, Medium, Heavy): Medium
Enter your food habits (e.g., Vegetarian, Non-Vegetarian, Mixed): Vegetarian
Enter your current medication (if any, otherwise type 'None'): None
Enter any allergies (if any, otherwise type 'None'): None
Enter the current season (e.g., Summer, Monsoon, Winter): Winter
Enter the current weather (e.g., Hot, Humid, Cold): Cold

Analyzing your information...
   => ML Model Prediction: 'Arthritis' (Confidence: 96.50%)

============================================================
🌿 Arogya AI - Personalized Ayurvedic Analysis 🌿
============================================================
💖 Your Ayurvedic Diagnosis

Predicted Disease: Arthritis (Sandhivata) [Confidence Level: 97%]

Based on your profile and symptoms, you are experiencing Arthritis (Sandhivata in Ayurveda), 
primarily caused by Vata dosha aggravation. The combination of joint pain, stiffness, swelling, 
difficulty walking, and morning stiffness are classic symptoms of this condition, especially 
prevalent during cold weather which naturally aggravates Vata.

🌿 Your Personalized Ayurvedic Plan

🩺 Condition Explained
Sandhivata (Arthritis) occurs when Vata dosha accumulates in the joints (Sandhi), causing 
pain, stiffness, and inflammation. The cold, dry qualities of Vata are particularly aggravated 
during winter, leading to reduced flexibility and increased discomfort. Ama (toxins) may also 
accumulate in the joints, further worsening the condition.

Ayurvedic Medicinal Herbs
- Sanskrit: Shallaki, Guggulu, Ashwagandha, Nirgundi
- English: Boswellia, Indian Bdellium, Winter Cherry, Vitex
- Effects: Anti-inflammatory, reduces joint pain, strengthens bones and tissues, improves 
  mobility, reduces Vata aggravation

💆 Ayurvedic Therapies
- Sanskrit: Abhyanga, Pinda Sweda, Janu Basti, Swedana
- English: Warm oil massage, Herbal bolus therapy, Knee pooling therapy, Steam therapy
- Effects: Lubricates joints, reduces stiffness, improves circulation, removes toxins, 
  alleviates pain, nourishes tissues

🥗 Dietary Recommendations

Eat This:
- Warm, cooked, and easily digestible foods
- Ghee, sesame oil, and healthy fats
- Ginger, turmeric, and warming spices
- Cooked vegetables like carrots, sweet potatoes, and squash
- Warm milk with turmeric before bed
- Moong dal and whole grains like rice and wheat

Avoid This:
- Cold, raw, and frozen foods
- Excess sour, salty foods (can increase inflammation)
- Refined sugars and processed foods
- Nightshade vegetables (tomatoes, potatoes, eggplant) which may aggravate inflammation
- Heavy, oily, and deep-fried foods

🏃 Lifestyle Advice
- Practice gentle yoga and stretching exercises daily to maintain flexibility
- Keep joints warm, especially during cold weather
- Apply warm sesame oil massage to affected joints before bathing
- Maintain regular sleep schedule (sleep before 10 PM, wake before 6 AM)
- Stay active but avoid overexertion
- Practice stress management through meditation and pranayama

🌿 Home Remedies & Precautions
- Drink warm water with ginger throughout the day
- Apply warm sesame or castor oil to painful joints
- Use heating pads or warm compresses on affected areas
- Take turmeric milk (1 tsp turmeric in warm milk) before bed
- Gentle massage with warm oils improves circulation
- Epsom salt bath can provide relief

👤 How Treatment Affects Your Body Type
These Ayurvedic treatments specifically address Vata imbalance by providing warmth, 
lubrication, and nourishment to your joints. The warm, oily therapies counteract the 
cold, dry nature of aggravated Vata, helping restore balance and mobility. Regular 
practice will strengthen your tissues, reduce inflammation, and improve overall joint health.

⚠️ Important Note: This is a complementary Ayurvedic approach. For severe arthritis, 
persistent pain, or worsening symptoms, please consult with a qualified healthcare 
professional or rheumatologist for comprehensive medical evaluation and treatment.

---
💡 This analysis combines ML prediction (96.50% confidence) with traditional Ayurvedic 
wisdom to provide personalized recommendations based on your unique constitution and symptoms.
```

## System Architecture

1. **Data Processing**: TF-IDF vectorization of symptoms (807 features)
2. **Feature Engineering**: 12 basic health features + 807 symptom features = 819 total features
3. **Class Balancing**: SMOTE applied for balanced training dataset (4201 → 20748 samples)
4. **Model Training**: Random Forest/Logistic Regression/SVM with ensemble approach
5. **ML Prediction**: Initial disease prediction using trained Random Forest model
6. **LLM Analysis**: Advanced contextual analysis considering all symptoms and lifestyle factors
7. **Ayurvedic Integration**: Comprehensive traditional medicine database with personalized recommendations
8. **Confidence Calibration**: Intelligent assessment of prediction confidence based on symptom patterns

## Model Performance

- **Random Forest Accuracy**: 1.0000 (100%)
- **Logistic Regression Accuracy**: 0.9964 (99.64%)
- **SVM Accuracy**: 0.9417 (94.17%)
- **Feature Set**: 819 features (12 basic + 807 TF-IDF)
- **Training Dataset**: 4,201 samples across 399 diseases
- **SMOTE Augmentation**: 20,748 balanced samples
- **Diseases Supported**: 399 disease categories with full Ayurvedic treatment plans

## Supported Diseases

The system supports diagnosis and Ayurvedic treatment recommendations for **399+ diseases** across multiple medical categories:

### Infectious Diseases
- Common Cold
- Influenza (Flu)
- Pneumonia
- Tuberculosis
- Dengue
- Malaria
- Chikungunya
- Chicken Pox
- Measles
- Mumps
- Typhoid
- Hepatitis A, B, C
- HIV/AIDS
- Urinary Tract Infection
- Skin Infection
- Fungal Infection
- Cholera
- Diarrhoea

### Metabolic & Endocrine Disorders
- Diabetes
- Hypertension
- Thyroid Disorders (Hyper/Hypothyroidism)
- Obesity
- PCOS (Polycystic Ovary Syndrome)
- Anemia

### Respiratory Conditions
- Asthma
- Bronchitis
- Sinusitis
- Cough
- Fever

### Cardiovascular Diseases
- Heart Disease
- Stroke

### Digestive System Disorders
- Gastritis
- Gastroenteritis
- Constipation
- Appendicitis
- Peptic Ulcer
- Jaundice
- Cirrhosis
- Fatty Liver
- Gallstones
- Kidney Stones

### Neurological & Mental Health
- Migraine
- Headache
- Depression
- Anxiety
- Insomnia
- Epilepsy
- Alzheimer Disease
- Parkinson Disease
- Meningitis
- Encephalitis

### Musculoskeletal Disorders
- Arthritis
- Cervical Spondylosis
- Disc Prolapse
- Rheumatoid Arthritis
- Gout
- Osteoporosis
- Paralysis

### Skin & Hair Conditions
- Acne
- Allergy
- Eczema
- Psoriasis
- Dandruff
- Hair Loss
- Vitiligo

### Eye, Ear & Dental Issues
- Conjunctivitis
- Glaucoma
- Cataract
- Hearing Loss
- Tinnitus
- Vertigo
- Toothache
- Gum Disease
- Mouth Ulcer

### Reproductive & Urogenital Health
- Erectile Dysfunction
- Infertility
- Menstrual Disorders
- Endometriosis
- Menopause
- Prostate Enlargement
- Overactive Bladder
- Urinary Incontinence

### Cancer Types
- Breast Cancer
- Various other cancer types

### Other Conditions
- Cancer (various types)
- Sleep Apnea
- Chronic Fatigue Syndrome
- Fibromyalgia
- Autoimmune Disorders
- Lupus
- Multiple Sclerosis
- Carpal Tunnel Syndrome
- Tennis Elbow
- Plantar Fasciitis
- Chronic Pain Syndromes

*The complete system supports 399+ diseases in total, with comprehensive Ayurvedic treatment recommendations for each condition.*

## Usage Modes

### 1. Demo Mode (Default)
Runs sample predictions with pre-defined test cases demonstrating the system capabilities.

### 2. Interactive Mode
Collects user symptoms and health information through an intuitive questionnaire:
- Symptoms input
- Age, height, weight
- Gender and age group
- Enhanced dosha selection with detailed descriptions
- Lifestyle factors (food habits, medication, allergies)
- Environmental factors (season, weather)

### 3. API Integration (Ready)
The system is designed to be easily integrated into web applications or APIs.

### 4. Offline Mode 🔌
**NEW**: The system now works completely offline when internet/LLM is unavailable:
- Automatic detection of LLM availability
- Graceful fallback to local Ayurvedic database
- Same ML prediction accuracy (100%)
- Comprehensive offline recommendations from 4,201+ disease database
- No API key required for basic operation
- Perfect for privacy-sensitive deployments

See [FALLBACK_MECHANISM.md](FALLBACK_MECHANISM.md) for detailed documentation.



## Technical Implementation

- **ML Framework**: Scikit-learn
- **Text Processing**: TF-IDF Vectorization
- **Class Balancing**: SMOTE (Synthetic Minority Oversampling)
- **Feature Scaling**: StandardScaler
- **Model Persistence**: Joblib
- **Data Processing**: Pandas, NumPy
- **Confidence Calibration**: Advanced prediction gap analysis

## File Structure

```
├── train_model.py           # Model training script
├── arogya_predict.py        # Main prediction system with interactive mode
├── disease_prediction_system.py  # Alternative comprehensive implementation
├── demo.py                  # Detailed system demonstration
├── test_fallback.py         # Fallback mechanism testing script
├── requirements.txt         # Python dependencies
├── random_forest_model.pkl  # Trained model (generated)
├── enhanced_ayurvedic_treatment_dataset.csv  # Comprehensive Ayurvedic treatment database (4,201+ diseases)
├── FALLBACK_MECHANISM.md    # Detailed offline mode documentation
└── AyurCore.ipynb          # Original research notebook
```

## Enhanced Ayurvedic Knowledge Base

The system includes an expanded database of traditional Ayurvedic treatments with:
- 50+ traditional herbs with Sanskrit and English names
- 30+ therapeutic treatments and procedures
- Dosha-specific recommendations for Vata, Pitta, and Kapha constitutions
- Personalized dietary guidelines
- Treatment effects explanation for different body types
- Advanced disease-to-treatment mapping with fallback options
- Dynamic treatment personalization based on body constitution

## Future Enhancements

- ~~Integration with real medical datasets~~ ✅ **DONE**
- ~~Enhanced confidence scoring~~ ✅ **DONE**
- ~~Interactive assessment mode~~ ✅ **DONE**
- ~~Top-5 predictions~~ ✅ **DONE**
- ~~Comprehensive dosha selection~~ ✅ **DONE**
- Web interface for easier access
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
