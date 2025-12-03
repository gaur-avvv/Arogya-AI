import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import google.generativeai as genai

# --- Step 1: Configuration ---
# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API client
genai.configure(api_key=GEMINI_API_KEY)

# --- Step 2: Load the Trained ML Model ---
# Load the pre-trained Random Forest model and its components
try:
    model_path = "random_forest_model.pkl"
    model_components = joblib.load(model_path)
    model = model_components['model']
    scaler = model_components['scaler']
    vectorizer = model_components['vectorizer']
    encoders = model_components['encoders']
    # The feature columns used during training
    training_feature_columns = model_components['feature_columns']
    print("✅ ML Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at '{model_path}'.")
    print("Please run 'python train_model.py' to train and save the model first.")
    exit()
except KeyError:
    print(f"❌ Error: The model file '{model_path}' is missing required components like 'feature_columns'.")
    print("Please re-run 'python train_model.py' to ensure the model is saved correctly.")
    exit()


def preprocess_input(user_data):
    """
    Preprocesses raw user input into a format the ML model can understand.
    """
    # Create a DataFrame from the user data
    user_df = pd.DataFrame([user_data])

    # Calculate BMI if not provided
    if 'BMI' not in user_df.columns:
        user_df['BMI'] = user_df['Weight_kg'] / (user_df['Height_cm'] / 100) ** 2

    # Encode categorical features using the loaded encoders
    categorical_columns = ['Age_Group', 'Gender', 'Body_Type_Dosha_Sanskrit', 
                           'Food_Habits', 'Current_Medication', 'Allergies', 'Season', 'Weather']
    
    for col in categorical_columns:
        # Use a default value (0) if a category is new or unseen
        encoded_values = []
        for item in user_df[col]:
            try:
                # The encoder expects a list or array
                encoded_values.append(encoders[col].transform([item])[0])
            except ValueError:
                # Handle unseen labels by assigning a default value, e.g., 0
                encoded_values.append(0) 
        user_df[f'{col}_encoded'] = encoded_values

    # Vectorize symptoms using the loaded TF-IDF vectorizer
    tfidf_features = vectorizer.transform(user_df['Symptoms']).toarray()
    
    # **FIX**: Create TF-IDF column names that match the training process ('tfidf_0', 'tfidf_1', etc.)
    tfidf_cols = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf_cols)

    # Combine all features
    base_feature_df = user_df[training_feature_columns].reset_index(drop=True)
    final_df = pd.concat([base_feature_df, tfidf_df], axis=1)
    
    # **FIX**: Ensure the final DataFrame columns exactly match what the scaler was fitted on.
    # This handles any discrepancy in the number of TF-IDF features.
    # Get the feature names from the scaler object itself
    scaler_feature_names = scaler.get_feature_names_out()
    final_df = final_df.reindex(columns=scaler_feature_names, fill_value=0)

    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(final_df)
    return scaled_features

def get_llm_validation_and_explanation(user_data, ml_prediction, confidence):
    """
    Uses the Gemini LLM to validate the ML prediction and provide a detailed,
    personalized Ayurvedic explanation.
    """
    # --- Step 3: Craft a Detailed Prompt for the LLM ---
    # This prompt guides the LLM to perform the validation and explanation task.
    prompt = f"""
    You are an expert Ayurvedic health assistant. Your task is to analyze a user's health data and an initial model prediction, then provide a final, trustworthy, and personalized Ayurvedic diagnosis and plan.

    **User's Health Profile:**
    - **Symptoms:** {user_data['Symptoms']}
    - **Age:** {user_data['Age']}
    - **Gender:** {user_data['Gender']}
    - **Body Type (Dosha):** {user_data['Body_Type_Dosha_Sanskrit']}
    - **Food Habits:** {user_data['Food_Habits']}
    - **Season:** {user_data['Season']}
    - **Weather:** {user_data['Weather']}
    - **Height:** {user_data['Height_cm']} cm
    - **Weight:** {user_data['Weight_kg']} kg

    **Initial Analysis (Internal Use Only):**
    - **Predicted Condition:** {ml_prediction}
    - **Initial Confidence:** {confidence:.2f}

    **Your Instructions:**

    1.  **Analyze and Diagnose:** You are an expert Ayurvedic health assistant and have extensive knowledge of Ayurvedic principles and practices. Based on the user's profile and your Ayurvedic knowledge, determine the final diagnosis. You can agree with the initial prediction or correct it with a brief reason.

    2.  **Generate Response:** Structure your entire response *exactly* like the example below. Use the same headings, emojis, and simple, clean formatting. Do not use asterisks or markdown bolding. Keep explanations concise and easy to read and don't use technical jargon.

    **--- RESPONSE TEMPLATE ---**

    💖 Your Ayurvedic Diagnosis

    Predicted Disease: [Your Final Diagnosis Here] [Confidence Level: in %]

    Based on your profile, it seems you are experiencing [Your Final Diagnosis Here].
    [Provide a brief, simple explanation of why, connecting symptoms, body type, current weather and season. And how Ayurveda views this condition along with ayurveda terms use english also in ().]

    🌿 Your Personalized Ayurvedic Plan

    🩺 Condition Explained
    [Explain the condition in simple Ayurvedic terms with English understandable terms in (). Keep it short and relatable and understandable.]

    Ayurvedic Medicinal Herbs
    - [List 3-4 specific Ayurvedic herbs or formulations known to help with the condition.]
    - [Example: Ashwagandha, Turmeric, Triphala]

    🥗 Dietary Recommendations
    Focus on foods that help you feel better.
    (Search online from trusted, verified and official sources and websites including books, articles, pdf, text, data, json, everywhere for common Ayurvedic recommendations for the diagnosed disease)

    Eat This:
    - [List 5-6 specific food items or types]
    - [Example: Warm soups and well-cooked vegetables]

    Avoid This:
    - [List 3-4 specific food items or types to avoid]
    - [Example: Cold drinks and heavy, oily foods]

    🏃 Lifestyle Advice
    - [Provide 3-4 simple, actionable lifestyle tips.]
    - [Example: Ensure you get plenty of rest and keep warm.]

    🌿 Home Remedies & Precautions
    - [List 3-4 simple and safe home remedies.]
    - [Example: Sip on warm ginger tea throughout the day.]

    ⚠️ Important Note: If your symptoms worsen or (related to diagnosis details), please consult a medical doctor. This plan is for gentle support.
    **--- END OF TEMPLATE ---**
    """

    # --- Step 4: Query the Gemini Model ---
    try:
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with the LLM: {e}"

def main():
    """
    Main function to run the prediction workflow.
    """
    print("="*60)
    print("AROGYA AI - Integrated ML + LLM Prediction System")
    print("="*60)
    print("Please provide your details to receive a personalized analysis.")
    print("-" * 60)


    # --- User Input Section ---
    # Get user input with more user-friendly prompts
    symptoms = input("Enter your symptoms (comma-separated): ")
    age = int(input("Enter your age: "))
    height_cm = int(input("Enter your height (cm): "))
    weight_kg = int(input("Enter your weight (kg): "))
    gender = input("Enter your gender: ")
    
    # Ask for body type in simple English
    body_type_english = input("Enter your general body type (e.g., Thin, Medium, Heavy): ").strip().lower()
    
    # Map English input to Sanskrit Dosha terms
    dosha_map = {
        "thin": "Vata",
        "medium": "Pitta",
        "heavy": "Kapha"
    }
    # Default to 'Vata' if the input is not recognized
    body_type_sanskrit = dosha_map.get(body_type_english, "Vata")

    food_habits = input("Enter your food habits (e.g., Vegetarian, Non-Vegetarian, Mixed): ")
    current_medication = input("Enter your current medication (if any, otherwise type 'None'): ")
    allergies = input("Enter any allergies (if any, otherwise type 'None'): ")
    season = input("Enter the current season (e.g., Summer, Monsoon, Winter): ")
    weather = input("Enter the current weather (e.g., Hot, Humid, Cold): ")

    # Automatically determine Age_Group from age
    if age <= 12:
        age_group = "Child"
    elif 13 <= age <= 19:
        age_group = "Adolescent"
    elif 20 <= age <= 39:
        age_group = "Young Adult"
    elif 40 <= age <= 59:
        age_group = "Middle-Aged Adult"
    else:
        age_group = "Senior"

    # Assemble the user_data dictionary for processing
    user_data = {
        "Symptoms": symptoms,
        "Age": age,
        "Height_cm": height_cm,
        "Weight_kg": weight_kg,
        "Gender": gender,
        "Age_Group": age_group,
        "Body_Type_Dosha_Sanskrit": body_type_sanskrit,
        "Food_Habits": food_habits,
        "Current_Medication": current_medication,
        "Allergies": allergies,
        "Season": season,
        "Weather": weather
    }

    # --- ML Model Prediction (Internal) ---
    print("\nAnalyzing your information...")
    try:
        # Preprocess the user's input data
        features = preprocess_input(user_data)

        # Get prediction and confidence score from the ML model
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)
        predicted_disease = encoders['Disease'].inverse_transform(prediction)[0]
        confidence = np.max(prediction_proba)

        print(f"   => ML Model Prediction: '{predicted_disease}' (Confidence: {confidence:.2%})")

    except Exception as e:
        print(f"   => Error during analysis: {e}")
        return

    # --- LLM Validation and Explanation ---
    llm_explanation = get_llm_validation_and_explanation(user_data, predicted_disease, confidence)
    
    print("\n" + "="*60)
    print("🌿 Arogya AI - Personalized Ayurvedic Analysis 🌿")
    print("="*60)
    print(llm_explanation)


if __name__ == "__main__":
    main()
