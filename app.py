#!/usr/bin/env python3
"""
Arogya AI - Web Interface
=========================

A Flask web application providing a user-friendly interface 
for disease prediction with Ayurvedic recommendations.
"""

from flask import Flask, render_template, request, jsonify
from arogya_predict import ArogyaAI
import json
import traceback

app = Flask(__name__)

# Initialize the AI system
ai_system = None

def init_ai_system():
    """Initialize the AI system once"""
    global ai_system
    if ai_system is None:
        try:
            ai_system = ArogyaAI()
            if not ai_system.model_components:
                print("❌ Model not loaded. Please run 'python train_model.py' first.")
                return False
            print("✅ Arogya AI system initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Error initializing AI system: {str(e)}")
            return False
    return True

@app.route('/')
def index():
    """Main page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if not init_ai_system():
            return jsonify({
                'error': 'AI system not available. Please contact administrator.'
            }), 500
        
        # Get form data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['symptoms', 'age', 'height', 'weight', 'gender']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Prepare user data for prediction
        user_data = {
            'Symptoms': data['symptoms'],
            'Age': int(data['age']),
            'Height_cm': float(data['height']),
            'Weight_kg': float(data['weight']),
            'Gender': data['gender'],
            'Body_Type_Dosha_Sanskrit': data.get('dosha', 'Vata'),
            'Food_Habits': data.get('food_habits', 'Mixed'),
            'Current_Medication': data.get('current_medication', 'None'),
            'Allergies': data.get('allergies', 'None'),
            'Season': data.get('season', 'Summer'),
            'Weather': data.get('weather', 'Hot')
        }
        
        # Auto-determine age group
        age = user_data['Age']
        if age <= 12:
            user_data['Age_Group'] = "Child"
        elif age <= 19:
            user_data['Age_Group'] = "Adolescent"
        elif age <= 35:
            user_data['Age_Group'] = "Young Adult"
        elif age <= 60:
            user_data['Age_Group'] = "Middle Age"
        elif age <= 75:
            user_data['Age_Group'] = "Senior"
        else:
            user_data['Age_Group'] = "Elderly"
        
        # Get prediction
        result = ai_system.predict_disease_with_recommendations(user_data)
        
        # Format response
        response = {
            'success': True,
            'prediction': {
                'disease': result['Predicted_Disease'],
                'confidence': f"{result['Confidence']:.1%}",
                'user_body_type': result.get('User_Body_Type', user_data['Body_Type_Dosha_Sanskrit'])
            },
            'ayurvedic_recommendations': {
                'herbs_sanskrit': result['Ayurvedic_Herbs_Sanskrit'],
                'herbs_english': result['Ayurvedic_Herbs_English'],
                'herbs_effects': result['Herbs_Effects'],
                'therapies_sanskrit': result['Ayurvedic_Therapies_Sanskrit'],
                'therapies_english': result['Ayurvedic_Therapies_English'],
                'therapies_effects': result['Therapies_Effects'],
                'dietary_recommendations': result['Dietary_Recommendations'],
                'body_type_effects': result['How_Treatment_Affects_Your_Body_Type']
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    if init_ai_system():
        return jsonify({'status': 'healthy', 'model_loaded': True})
    else:
        return jsonify({'status': 'error', 'model_loaded': False}), 500

if __name__ == '__main__':
    # Initialize AI system on startup
    init_ai_system()
    app.run(debug=True, host='0.0.0.0', port=5000)