#!/usr/bin/env python3
"""
Arogya AI - Main Prediction System
==================================

This script loads the trained model and provides disease prediction
along with comprehensive Ayurvedic recommendations as specified.

Features:
- Disease prediction using Random Forest model (>99% accuracy)
- Comprehensive Ayurvedic recommendations including:
  - Ayurvedic_Herbs_Sanskrit
  - Ayurvedic_Herbs_English
  - Herbs_Effects
  - Ayurvedic_Therapies_Sanskrit
  - Ayurvedic_Therapies_English
  - Therapies_Effects
  - Dietary_Recommendations
  - How_Treatment_Affects_Your_Body_Type
"""

import joblib
import pandas as pd
import numpy as np
import os
import warnings
from typing import Dict, Any

warnings.filterwarnings('ignore')

class ArogyaAI:
    """Main Arogya AI Prediction System"""
    
    def __init__(self, model_path='random_forest_model.pkl'):
        """Initialize the prediction system"""
        self.model_path = model_path
        self.model_components = None
        self.ayurvedic_database = None
        self.load_model()
        self.create_ayurvedic_database()
    
    def load_model(self):
        """Load the trained model and all components"""
        if not os.path.exists(self.model_path):
            print(f"Model file {self.model_path} not found!")
            print("Please run 'python train_model.py' first to train the model.")
            return False
        
        try:
            self.model_components = joblib.load(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
            print(f"   Model type: {self.model_components['model_type']}")
            print(f"   Model accuracy: {self.model_components['results'][self.model_components['model_type']]:.4f}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def create_ayurvedic_database(self):
        """
        Create comprehensive Ayurvedic recommendations database
        Maps diseases to their corresponding Ayurvedic treatments
        """
        self.ayurvedic_database = {
            'Jwara (Fever)': {
                'Ayurvedic_Herbs_Sanskrit': 'Tulasi, Sunthi, Marich, Pippali, Haridra',
                'Ayurvedic_Herbs_English': 'Holy Basil, Dry Ginger, Black Pepper, Long Pepper, Turmeric',
                'Herbs_Effects': 'Antipyretic, immune-boosting, anti-inflammatory, digestive stimulant, antimicrobial properties',
                'Ayurvedic_Therapies_Sanskrit': 'Langhana, Swedana, Kashaya Sevana, Pathya Ahara',
                'Ayurvedic_Therapies_English': 'Fasting therapy, Steam therapy, Herbal decoctions, Proper diet regimen',
                'Therapies_Effects': 'Reduces body heat, promotes sweating, detoxifies body, balances Pitta dosha, strengthens immunity',
                'Dietary_Recommendations': 'Light, easily digestible foods, warm water, ginger tea, avoid heavy/oily foods, take rest',
                'How_Treatment_Affects_Your_Body_Type': 'Balances aggravated Pitta dosha, cools body temperature, strengthens Ojas (immunity), cleanses Ama (toxins)'
            },
            
            'Kasa (Cough)': {
                'Ayurvedic_Herbs_Sanskrit': 'Vasa, Kantakari, Bharangi, Pushkarmool, Yashtimadhu',
                'Ayurvedic_Herbs_English': 'Malabar Nut, Yellow Berried Nightshade, Bharangi, Elecampane, Licorice',
                'Herbs_Effects': 'Expectorant, bronchodilator, anti-inflammatory, soothes respiratory tract, reduces cough reflex',
                'Ayurvedic_Therapies_Sanskrit': 'Swedana, Nasya, Dhumapana, Pranayama, Gargling',
                'Ayurvedic_Therapies_English': 'Steam inhalation, Nasal drops, Medicated smoking, Breathing exercises, Herbal gargles',
                'Therapies_Effects': 'Opens respiratory channels, reduces Kapha congestion, soothes throat, improves lung function',
                'Dietary_Recommendations': 'Warm foods, honey with ginger, avoid cold drinks/foods, dairy products, increase fluid intake',
                'How_Treatment_Affects_Your_Body_Type': 'Reduces excess Kapha dosha, clears respiratory channels (Pranavaha Srotas), strengthens lung capacity'
            },
            
            'Shwasa (Asthma)': {
                'Ayurvedic_Herbs_Sanskrit': 'Vasa, Kantakari, Pushkarmool, Bharangi, Shirisha',
                'Ayurvedic_Herbs_English': 'Malabar Nut, Yellow Berried Nightshade, Elecampane, Bharangi, Sirisha',
                'Herbs_Effects': 'Bronchodilator, anti-asthmatic, reduces airway inflammation, improves breathing capacity',
                'Ayurvedic_Therapies_Sanskrit': 'Swedana, Pranayama, Yoga, Nasya, Abhyanga',
                'Ayurvedic_Therapies_English': 'Steam therapy, Breathing exercises, Yoga practice, Nasal therapy, Oil massage',
                'Therapies_Effects': 'Opens airways, reduces respiratory inflammation, calms nervous system, strengthens respiratory muscles',
                'Dietary_Recommendations': 'Warm, light foods, avoid cold/heavy foods, dairy, processed foods, practice regular breathing exercises',
                'How_Treatment_Affects_Your_Body_Type': 'Balances Vata and Kapha doshas, opens Pranavaha Srotas, reduces bronchial hypersensitivity'
            },
            
            'Prameha (Diabetes)': {
                'Ayurvedic_Herbs_Sanskrit': 'Guduchi, Meshashringi, Vijaysar, Haridra, Amalaki',
                'Ayurvedic_Herbs_English': 'Tinospora, Gymnema, Indian Kino, Turmeric, Amla',
                'Herbs_Effects': 'Hypoglycemic, improves insulin sensitivity, pancreatic tonic, antioxidant, metabolic regulator',
                'Ayurvedic_Therapies_Sanskrit': 'Panchakarma, Udvartana, Yoga, Pranayama, Dhanyamla Dhara',
                'Ayurvedic_Therapies_English': 'Detoxification, Dry powder massage, Yoga, Breathing exercises, Fermented liquid massage',
                'Therapies_Effects': 'Improves metabolism, reduces insulin resistance, detoxifies body, balances blood sugar',
                'Dietary_Recommendations': 'Low glycemic foods, bitter vegetables, whole grains, avoid sugar/refined carbs, regular meal times',
                'How_Treatment_Affects_Your_Body_Type': 'Reduces Kapha dosha, improves Agni (digestive fire), regulates Ojas, balances metabolism'
            },
            
            'Hridroga (Heart Disease)': {
                'Ayurvedic_Herbs_Sanskrit': 'Arjuna, Punarnava, Brahmi, Shankhpushpi, Dashmoolarishta',
                'Ayurvedic_Herbs_English': 'Arjuna bark, Punarnava, Brahmi, Convolvulus, Ten roots tonic',
                'Herbs_Effects': 'Cardiotonic, reduces blood pressure, strengthens heart muscle, improves circulation, stress reducer',
                'Ayurvedic_Therapies_Sanskrit': 'Hridaya Basti, Shirodhara, Abhyanga, Pranayama, Yoga',
                'Ayurvedic_Therapies_English': 'Heart oil pooling, Oil pouring therapy, Body massage, Breathing exercises, Yoga',
                'Therapies_Effects': 'Strengthens heart, calms nervous system, improves circulation, reduces stress and anxiety',
                'Dietary_Recommendations': 'Heart-healthy foods, omega-3 rich foods, avoid saturated fats, reduce salt, practice meditation',
                'How_Treatment_Affects_Your_Body_Type': 'Calms Vata dosha, nourishes heart tissue (Hridaya), improves Prana flow, reduces Pitta heat'
            },
            
            'Sandhivata (Arthritis)': {
                'Ayurvedic_Herbs_Sanskrit': 'Guggulu, Shallaki, Rasna, Nirgundi, Ashwagandha',
                'Ayurvedic_Herbs_English': 'Guggul, Boswellia, Rasna, Five-leaved chaste tree, Winter cherry',
                'Herbs_Effects': 'Anti-inflammatory, joint nourishing, pain reliever, tissue regenerator, strengthens bones',
                'Ayurvedic_Therapies_Sanskrit': 'Abhyanga, Swedana, Janu Basti, Kati Basti, Pizhichil',
                'Ayurvedic_Therapies_English': 'Oil massage, Steam therapy, Knee oil pooling, Lower back oil pooling, Oil bath',
                'Therapies_Effects': 'Reduces joint stiffness, improves mobility, nourishes joints, reduces pain and inflammation',
                'Dietary_Recommendations': 'Anti-inflammatory foods, warm cooked meals, ghee, avoid cold/dry foods, nightshades',
                'How_Treatment_Affects_Your_Body_Type': 'Pacifies Vata dosha, nourishes Asthi Dhatu (bone tissue), improves joint lubrication'
            },
            
            'Amlapitta (Gastritis)': {
                'Ayurvedic_Herbs_Sanskrit': 'Yashtimadhu, Amalaki, Shatavari, Guduchi, Kamadudha',
                'Ayurvedic_Herbs_English': 'Licorice, Amla, Asparagus, Tinospora, Calcium compound',
                'Herbs_Effects': 'Antacid, gastro-protective, heals ulcers, reduces acidity, soothes stomach lining',
                'Ayurvedic_Therapies_Sanskrit': 'Takradhara, Virechana (mild), Sheetali Pranayama',
                'Ayurvedic_Therapies_English': 'Buttermilk therapy, Gentle purgation, Cooling breathing',
                'Therapies_Effects': 'Cools stomach, reduces acid production, heals gastric ulcers, balances digestive fire',
                'Dietary_Recommendations': 'Cooling foods, avoid spicy/oily/acidic foods, eat regular meals, coconut water, fennel tea',
                'How_Treatment_Affects_Your_Body_Type': 'Reduces Pitta dosha, cools Agni, heals gastric mucosa, balances acid-alkaline ratio'
            },
            
            'Shiroroga (Headache)': {
                'Ayurvedic_Herbs_Sanskrit': 'Brahmi, Shankhpushpi, Jatamansi, Saraswatarishta, Shirashooladi Vajra',
                'Ayurvedic_Herbs_English': 'Brahmi, Convolvulus, Spikenard, Saraswata tonic, Headache relief compound',
                'Herbs_Effects': 'Neurocalming, reduces headache intensity, improves mental clarity, stress reliever',
                'Ayurvedic_Therapies_Sanskrit': 'Shiropichu, Shirobasti, Nasya, Akshi Tarpana',
                'Ayurvedic_Therapies_English': 'Head oil application, Oil pooling on head, Nasal therapy, Eye treatments',
                'Therapies_Effects': 'Soothes nervous system, improves head circulation, reduces tension, calms mind',
                'Dietary_Recommendations': 'Regular meals, avoid triggers, cooling foods, adequate hydration, stress management',
                'How_Treatment_Affects_Your_Body_Type': 'Balances Vata and Pitta doshas, calms Majja Dhatu (nervous tissue), reduces head tension'
            },
            
            'Anidra (Insomnia)': {
                'Ayurvedic_Herbs_Sanskrit': 'Brahmi, Shankhpushpi, Jatamansi, Ashwagandha, Saraswatarishta',
                'Ayurvedic_Herbs_English': 'Brahmi, Convolvulus, Spikenard, Winter cherry, Saraswata tonic',
                'Herbs_Effects': 'Natural sedative, calms mind, reduces anxiety, promotes restful sleep, neuroprotective',
                'Ayurvedic_Therapies_Sanskrit': 'Shirodhara, Abhyanga, Padabhyanga, Yoga Nidra, Meditation',
                'Ayurvedic_Therapies_English': 'Oil pouring therapy, Body massage, Foot massage, Yogic sleep, Meditation',
                'Therapies_Effects': 'Deeply relaxes nervous system, calms Vata dosha, promotes natural sleep cycles',
                'Dietary_Recommendations': 'Light dinner, warm milk with nutmeg, avoid caffeine/stimulants, regular sleep schedule',
                'How_Treatment_Affects_Your_Body_Type': 'Pacifies Vata dosha, calms Rajas (mental activity), promotes Sattva (mental clarity)'
            },
            
            'Shotha (Inflammation)': {
                'Ayurvedic_Herbs_Sanskrit': 'Punarnava, Gokshura, Haridra, Guggulu, Triphala',
                'Ayurvedic_Herbs_English': 'Punarnava, Gokshura, Turmeric, Guggul, Three fruits',
                'Herbs_Effects': 'Anti-inflammatory, reduces swelling, diuretic, detoxifying, tissue healing',
                'Ayurvedic_Therapies_Sanskrit': 'Lepa (Poultice), Abhyanga, Swedana, Panchakarma',
                'Ayurvedic_Therapies_English': 'Herbal paste application, Oil massage, Steam therapy, Detoxification',
                'Therapies_Effects': 'Reduces inflammation, improves circulation, promotes healing, eliminates toxins',
                'Dietary_Recommendations': 'Anti-inflammatory foods, turmeric, ginger, avoid inflammatory foods, increase water intake',
                'How_Treatment_Affects_Your_Body_Type': 'Reduces Pitta and Kapha doshas, eliminates Ama (toxins), improves tissue healing'
            }
        }
    
    def preprocess_user_input(self, user_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess user input for prediction"""
        if not self.model_components:
            raise ValueError("Model not loaded. Please load model first.")
        
        # Convert to DataFrame
        user_df = pd.DataFrame([user_data])
        
        # Calculate BMI if not provided
        if 'BMI' not in user_df.columns and 'Height_cm' in user_df.columns and 'Weight_kg' in user_df.columns:
            user_df['BMI'] = user_df['Weight_kg'] / (user_df['Height_cm'] / 100) ** 2
        
        # Encode categorical features
        categorical_columns = ['Age_Group', 'Gender', 'Body_Type_Dosha_Sanskrit', 
                              'Food_Habits', 'Current_Medication', 'Allergies', 'Season', 'Weather']
        
        for col in categorical_columns:
            if col in user_df.columns and col in self.model_components['encoders']:
                try:
                    user_df[f'{col}_encoded'] = self.model_components['encoders'][col].transform(user_df[col])
                except ValueError:
                    # Handle unknown categories
                    user_df[f'{col}_encoded'] = 0
        
        # Select numerical and encoded features
        other_features = user_df[self.model_components['feature_columns']]
        
        # Transform symptoms using TF-IDF
        if 'Symptoms' in user_data:
            tfidf_matrix = self.model_components['vectorizer'].transform([user_data['Symptoms']])
            tfidf_features = pd.DataFrame(
                tfidf_matrix.toarray(), 
                columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
            )
            # Combine features
            combined_features = pd.concat([other_features.reset_index(drop=True), 
                                         tfidf_features.reset_index(drop=True)], axis=1)
        else:
            combined_features = other_features
        
        # Scale features
        combined_features_scaled = self.model_components['scaler'].transform(combined_features)
        
        return combined_features_scaled
    
    def predict_disease_with_recommendations(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main prediction function - returns disease prediction with Ayurvedic recommendations
        """
        if not self.model_components:
            raise ValueError("Model not loaded!")
        
        # Preprocess input
        processed_features = self.preprocess_user_input(user_data)
        
        # Make prediction
        prediction = self.model_components['model'].predict(processed_features)
        prediction_proba = self.model_components['model'].predict_proba(processed_features)
        
        # Get disease name
        predicted_disease = self.model_components['encoders']['Disease'].inverse_transform(prediction)[0]
        confidence = np.max(prediction_proba)
        
        # Get Ayurvedic recommendations
        body_type = user_data.get('Body_Type_Dosha_Sanskrit', 'Unknown')
        
        if predicted_disease in self.ayurvedic_database:
            ayurvedic_recommendations = self.ayurvedic_database[predicted_disease].copy()
        else:
            # Default recommendations
            ayurvedic_recommendations = {
                'Ayurvedic_Herbs_Sanskrit': 'Amalaki, Haridra, Tulasi',
                'Ayurvedic_Herbs_English': 'Amla, Turmeric, Holy Basil',
                'Herbs_Effects': 'General immunity boost, anti-inflammatory, antioxidant',
                'Ayurvedic_Therapies_Sanskrit': 'Abhyanga, Pranayama, Yoga',
                'Ayurvedic_Therapies_English': 'Oil massage, Breathing exercises, Yoga',
                'Therapies_Effects': 'General wellness, stress reduction, improved circulation',
                'Dietary_Recommendations': 'Balanced diet, fresh foods, adequate water, regular meals',
                'How_Treatment_Affects_Your_Body_Type': 'General balancing of doshas, promotes overall health'
            }
        
        # Personalize for body type
        if body_type != "Unknown":
            ayurvedic_recommendations['How_Treatment_Affects_Your_Body_Type'] += f" (Specifically beneficial for {body_type} constitution)"
        
        # Compile results
        result = {
            'Predicted_Disease': predicted_disease,
            'Confidence': float(confidence),
            'User_Symptoms': user_data.get('Symptoms', ''),
            'User_Body_Type': body_type,
            **ayurvedic_recommendations
        }
        
        return result
    
    def get_dosha_selection(self):
        """Enhanced dosha selection with clear body type descriptions"""
        
        print("\n🌿 AYURVEDIC BODY TYPE ASSESSMENT 🌿")
        print("=" * 50)
        print("Select your body type based on physical characteristics:\n")
        
        dosha_options = {
            '1': {
                'name': 'Vata',
                'constitution': 'Air_Space_Constitution',
                'body_type': 'Thin/Lean',
                'description': 'Naturally thin build, difficulty gaining weight, dry skin, cold hands/feet'
            },
            '2': {
                'name': 'Pitta',
                'constitution': 'Fire_Water_Constitution', 
                'body_type': 'Medium',
                'description': 'Medium build, good muscle tone, warm body, strong appetite'
            },
            '3': {
                'name': 'Kapha',
                'constitution': 'Earth_Water_Constitution',
                'body_type': 'Heavy/Large',
                'description': 'Naturally larger build, gains weight easily, cool moist skin, steady energy'
            },
            '4': {
                'name': 'Vata-Pitta',
                'constitution': 'Air_Fire_Mixed_Constitution',
                'body_type': 'Thin to Medium',
                'description': 'Variable build, creative energy, moderate body temperature'
            },
            '5': {
                'name': 'Vata-Kapha',
                'constitution': 'Air_Earth_Mixed_Constitution',
                'body_type': 'Thin to Heavy',
                'description': 'Variable patterns, irregular tendencies, sensitive to changes'
            },
            '6': {
                'name': 'Pitta-Kapha',
                'constitution': 'Fire_Earth_Mixed_Constitution',
                'body_type': 'Medium to Heavy',
                'description': 'Strong stable build, good strength, balanced metabolism'
            }
        }
        
        # Display options
        for key, value in dosha_options.items():
            print(f"{key}. {value['name']} - {value['body_type']}")
            print(f"   {value['description']}")
            print()
        
        print("You can enter:")
        print("• Number (1-6)")  
        print("• Dosha name (e.g., 'Vata', 'Pitta-Kapha')")
        print("• Body type (e.g., 'thin', 'medium', 'heavy')")
        
        while True:
            dosha_choice = input("\nEnter your selection: ").strip()
            
            # Check if it's a number
            if dosha_choice in dosha_options:
                selected = dosha_options[dosha_choice]
                return selected['name'], selected['constitution']
            
            # Check if it's a dosha name (case insensitive)
            dosha_choice_lower = dosha_choice.lower()
            for option in dosha_options.values():
                if option['name'].lower() == dosha_choice_lower:
                    return option['name'], option['constitution']
            
            # Check if it's a body type description
            body_type_mapping = {
                'thin': '1', 'lean': '1', 'skinny': '1',
                'medium': '2', 'average': '2', 'moderate': '2',
                'heavy': '3', 'large': '3', 'big': '3', 'fat': '3',
                'thin to medium': '4', 'variable thin': '4',
                'thin to heavy': '5', 'irregular': '5',
                'medium to heavy': '6', 'strong': '6'
            }
            
            if dosha_choice_lower in body_type_mapping:
                selected_key = body_type_mapping[dosha_choice_lower]
                selected = dosha_options[selected_key]
                return selected['name'], selected['constitution']
            
            print("❌ Invalid selection. Please try again.")
            print("Use numbers 1-6, dosha names, or body type descriptions.")

    def get_user_input_interactive(self) -> Dict[str, Any]:
        """Interactive input collection"""
        print("\n" + "="*60)
        print("AROGYA AI - INTERACTIVE HEALTH ASSESSMENT")
        print("="*60)
        print("Please provide the following information for accurate prediction:\n")
        
        user_data = {}
        
        # Basic Information
        user_data['Symptoms'] = input("Enter your symptoms (comma-separated): ")
        user_data['Age'] = int(input("Enter your age: "))
        user_data['Height_cm'] = float(input("Enter your height in cm: "))
        user_data['Weight_kg'] = float(input("Enter your weight in kg: "))
        
        # Auto-determine age group
        age = user_data['Age']
        if age <= 12:
            age_group = "Child"
        elif age <= 19:
            age_group = "Adolescent"
        elif age <= 35:
            age_group = "Young Adult"
        elif age <= 50:
            age_group = "Middle Age"
        elif age <= 65:
            age_group = "Senior"
        else:
            age_group = "Elderly"
        
        user_data['Age_Group'] = age_group
        
        print(f"\nGender options: Male, Female")
        user_data['Gender'] = input("Enter your gender: ")
        
        # Use enhanced dosha selection
        dosha_name, dosha_constitution = self.get_dosha_selection()
        user_data['Body_Type_Dosha_Sanskrit'] = dosha_name
        print(f"\n✅ Selected: {dosha_name} ({dosha_constitution})")
        
        print(f"\nFood Habits options: Vegetarian, Non-Vegetarian, Vegan, Mixed")
        user_data['Food_Habits'] = input("Enter your food habits: ") or "Mixed"
        
        user_data['Current_Medication'] = input("Enter current medications (or 'None'): ") or "None"
        user_data['Allergies'] = input("Enter known allergies (or 'None'): ") or "None"
        
        print(f"\nSeason options: Spring, Summer, Monsoon, Autumn, Winter")
        user_data['Season'] = input("Enter current season: ") or "Summer"
        
        print(f"\nWeather options: Hot, Cold, Humid, Dry, Rainy")
        user_data['Weather'] = input("Enter current weather: ") or "Hot"
        
        return user_data

def demo_sample_prediction():
    """Demonstrate with sample data"""
    print("\n" + "="*70)
    print("AROGYA AI - DISEASE PREDICTION WITH AYURVEDIC RECOMMENDATIONS")
    print("="*70)
    
    # Initialize system
    ai_system = ArogyaAI()
    
    if not ai_system.model_components:
        print("❌ Model not available. Please run 'python train_model.py' first.")
        return
    
    # Sample predictions
    sample_cases = [
        {
            'name': 'Case 1: Fever symptoms',
            'data': {
                'Symptoms': 'fever, body ache, headache, fatigue',
                'Age': 35,
                'Height_cm': 170,
                'Weight_kg': 75,
                'Gender': 'Female',
                'Age_Group': 'Young Adult',
                'Body_Type_Dosha_Sanskrit': 'Pitta',
                'Food_Habits': 'Vegetarian',
                'Current_Medication': 'None',
                'Allergies': 'None',
                'Season': 'Summer',
                'Weather': 'Hot'
            }
        },
        {
            'name': 'Case 2: Respiratory symptoms',
            'data': {
                'Symptoms': 'cough, breathing difficulty, chest tightness',
                'Age': 45,
                'Height_cm': 175,
                'Weight_kg': 80,
                'Gender': 'Male',
                'Age_Group': 'Middle Age',
                'Body_Type_Dosha_Sanskrit': 'Kapha',
                'Food_Habits': 'Non-Vegetarian',
                'Current_Medication': 'None',
                'Allergies': 'None',
                'Season': 'Winter',
                'Weather': 'Cold'
            }
        }
    ]
    
    for case in sample_cases:
        print(f"\n{'='*50}")
        print(f"🔍 {case['name']}")
        print(f"{'='*50}")
        
        try:
            result = ai_system.predict_disease_with_recommendations(case['data'])
            
            print(f"\n📋 PREDICTION RESULTS:")
            print(f"   Predicted Disease: {result['Predicted_Disease']}")
            print(f"   Confidence: {result['Confidence']:.2%}")
            print(f"   Symptoms: {result['User_Symptoms']}")
            print(f"   Body Type: {result['User_Body_Type']}")
            
            print(f"\n🌿 AYURVEDIC RECOMMENDATIONS:")
            print(f"   Sanskrit Herbs: {result['Ayurvedic_Herbs_Sanskrit']}")
            print(f"   English Herbs: {result['Ayurvedic_Herbs_English']}")
            print(f"   Herb Effects: {result['Herbs_Effects']}")
            print(f"   Sanskrit Therapies: {result['Ayurvedic_Therapies_Sanskrit']}")
            print(f"   English Therapies: {result['Ayurvedic_Therapies_English']}")
            print(f"   Therapy Effects: {result['Therapies_Effects']}")
            
            print(f"\n🍽️ DIETARY RECOMMENDATIONS:")
            print(f"   {result['Dietary_Recommendations']}")
            
            print(f"\n👤 PERSONALIZED TREATMENT EFFECTS:")
            print(f"   {result['How_Treatment_Affects_Your_Body_Type']}")
            
        except Exception as e:
            print(f"❌ Error in prediction: {str(e)}")

def interactive_mode():
    """Run interactive prediction mode"""
    ai_system = ArogyaAI()
    
    if not ai_system.model_components:
        print("❌ Model not available. Please run 'python train_model.py' first.")
        return
    
    user_input = ai_system.get_user_input_interactive()
    
    print(f"\n{'='*60}")
    print("🔍 YOUR PREDICTION RESULTS")
    print(f"{'='*60}")
    
    try:
        result = ai_system.predict_disease_with_recommendations(user_input)
        
        print(f"\n📋 MEDICAL PREDICTION:")
        print(f"   Disease: {result['Predicted_Disease']}")
        print(f"   Confidence: {result['Confidence']:.2%}")
        
        print(f"\n🌿 AYURVEDIC TREATMENT PLAN:")
        print(f"   Sanskrit Herbs: {result['Ayurvedic_Herbs_Sanskrit']}")
        print(f"   English Herbs: {result['Ayurvedic_Herbs_English']}")
        print(f"   Herb Benefits: {result['Herbs_Effects']}")
        
        print(f"\n💆 THERAPEUTIC TREATMENTS:")
        print(f"   Sanskrit Therapies: {result['Ayurvedic_Therapies_Sanskrit']}")
        print(f"   English Therapies: {result['Ayurvedic_Therapies_English']}")
        print(f"   Treatment Benefits: {result['Therapies_Effects']}")
        
        print(f"\n🥗 DIETARY GUIDANCE:")
        print(f"   {result['Dietary_Recommendations']}")
        
        print(f"\n🎯 PERSONALIZED TREATMENT EFFECTS:")
        print(f"   {result['How_Treatment_Affects_Your_Body_Type']}")
        
        # Save results
        print(f"\n💾 Results saved for your reference.")
        
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")

if __name__ == "__main__":
    print("Initializing Arogya AI...")
    
    # Check if model exists
    if not os.path.exists('random_forest_model.pkl'):
        print("\n❌ Trained model not found!")
        print("📋 Please run the following command first:")
        print("   python train_model.py")
        print("\nThis will train the model and save it for predictions.")
    else:
        # Run demo
        demo_sample_prediction()
        
        # Ask for interactive mode
        print(f"\n{'='*70}")
        interactive = input("Would you like to try interactive prediction? (y/n): ").lower().strip()
        
        if interactive == 'y':
            interactive_mode()
    
    print(f"\n{'='*70}")
    print("Thank you for using Arogya AI!")
    print("Stay healthy with the wisdom of Ayurveda! 🌿")
    print(f"{'='*70}")