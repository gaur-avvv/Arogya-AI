import os
import sys
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for consoles (prevents UnicodeEncodeError on Windows)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --- Step 1: Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai_client = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        genai_client = genai
    except Exception as e:
        print(f"Warning: Failed to initialize Google Generative AI: {e}")

# --- Step 2: Load the Trained ML Model ---
try:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_forest_model.pkl")
    model_components = joblib.load(model_path)
    model = model_components['model']
    scaler = model_components['scaler']
    vectorizer = model_components['vectorizer']
    encoders = model_components['encoders']
    training_feature_columns = model_components['feature_columns']
except FileNotFoundError:
    print(f"Error: Model file not found at '{model_path}'. Please run 'python train_model.py' first.")
    model = None
    scaler = None
    vectorizer = None
    encoders = {}
    training_feature_columns = []
except Exception as e:
    print(f"Error loading model file '{model_path}': {e}")
    model = None
    scaler = None
    vectorizer = None
    encoders = {}
    training_feature_columns = []

# --- Step 3: Ayurvedic Knowledge Base & Offline Fallback ---
_AYURVEDIC_CACHE = None

BUILTIN_AYURVEDIC_DATABASE = {
    'Common Cold': {
        'Ayurvedic_Herbs_Sanskrit': 'Tulasi, Sunthi, Haridra',
        'Ayurvedic_Herbs_English': 'Holy Basil, Ginger, Turmeric',
        'Herbs_Effects': 'Boosts immunity, reduces inflammation, clears respiratory passages',
        'Ayurvedic_Therapies_Sanskrit': 'Nasya, Swedana, Kashaya',
        'Ayurvedic_Therapies_English': 'Nasal therapy, Steam therapy, Herbal decoctions',
        'Therapies_Effects': 'Clears nasal passages, promotes sweating, balances Kapha dosha',
        'Dietary_Recommendations': 'Warm foods, ginger tea, avoid cold and heavy foods, increase warming spices',
        'Diet_Eat': 'Warm soups, steamed vegetables, ginger tea, turmeric milk, light khichdi',
        'Diet_Avoid': 'Cold drinks, ice cream, yogurt at night, deep-fried snacks',
        'How_Treatment_Affects_Your_Body_Type': 'Reduces Kapha, warms the body, improves circulation and metabolic Agni',
        'Condition_Explanation': 'Common cold (Pratishyaya) is predominantly a Kapha-Vata imbalance causing mucus buildup in the respiratory channels.'
    },
    'Diabetes': {
        'Ayurvedic_Herbs_Sanskrit': 'Guduchi, Meshashringi, Vijaysar, Karela',
        'Ayurvedic_Herbs_English': 'Tinospora, Gymnema, Indian Kino, Bitter Gourd',
        'Herbs_Effects': 'Regulates blood sugar, improves insulin sensitivity, supports pancreatic function',
        'Ayurvedic_Therapies_Sanskrit': 'Panchakarma, Udvartana, Yoga Pranayama',
        'Ayurvedic_Therapies_English': 'Detoxification, Dry herbal powder massage, Yogic breathing',
        'Therapies_Effects': 'Detoxifies tissues, improves circulation, reduces stress, balances metabolism',
        'Dietary_Recommendations': 'Low glycemic foods, bitter vegetables, avoid sugar/refined carbs, regular meal timings',
        'Diet_Eat': 'Barley, bitter gourd (karela), fenugreek seeds, amla, green leafy vegetables',
        'Diet_Avoid': 'Refined sugar, white flour, sweetened beverages, heavy oily sweets',
        'How_Treatment_Affects_Your_Body_Type': 'Balances Kapha dosha, reduces tissue inflammation, enhances digestive fire (Agni)',
        'Condition_Explanation': 'Diabetes (Prameha) is described as a disorder of Medas (fat tissue) and Kapha leading to compromised metabolic balance.'
    },
    'Arthritis': {
        'Ayurvedic_Herbs_Sanskrit': 'Shallaki, Guggulu, Ashwagandha, Nirgundi',
        'Ayurvedic_Herbs_English': 'Boswellia, Indian Bdellium, Winter Cherry, Vitex',
        'Herbs_Effects': 'Anti-inflammatory, reduces joint pain, strengthens musculoskeletal tissues, lubricates joints',
        'Ayurvedic_Therapies_Sanskrit': 'Abhyanga, Pinda Sweda, Janu Basti, Swedana',
        'Ayurvedic_Therapies_English': 'Warm oil massage, Herbal bolus fomentation, Knee oil pooling, Steam therapy',
        'Therapies_Effects': 'Lubricates joints, eliminates Ama (toxins), relieves morning stiffness, pacifies Vata',
        'Dietary_Recommendations': 'Warm, cooked, easily digestible foods; ghee, ginger, turmeric; avoid nightshades',
        'Diet_Eat': 'Warm vegetable soups, moong dal, ginger tea, warm milk with turmeric, soaked almonds',
        'Diet_Avoid': 'Cold foods, raw salads, excess tomatoes/eggplant/potatoes, carbonated drinks',
        'How_Treatment_Affects_Your_Body_Type': 'Subdues dry, cold Vata dosha, restores joint synovial lubrication, reduces chronic inflammation',
        'Condition_Explanation': 'Arthritis (Sandhivata / Amavata) occurs when aggravated Vata and metabolic toxins (Ama) lodge in the joints, leading to pain and stiffness.'
    },
    'Gastritis': {
        'Ayurvedic_Herbs_Sanskrit': 'Yashtimadhu, Amalaki, Shatavari, Guduchi',
        'Ayurvedic_Herbs_English': 'Licorice, Amla, Asparagus racemosus, Tinospora',
        'Herbs_Effects': 'Soothes gastric mucosa, balances stomach acid, supports mucosal healing',
        'Ayurvedic_Therapies_Sanskrit': 'Takradhara, Virechana, Pachana Karma',
        'Ayurvedic_Therapies_English': 'Medicated buttermilk pouring, Therapeutic purgation, Digestive regulation',
        'Therapies_Effects': 'Cools gastric heat, reduces excess Pitta, harmonizes digestive secretions',
        'Dietary_Recommendations': 'Cooling foods, fresh coconut water, avoid sour/spicy/fried foods, eat on time',
        'Diet_Eat': 'Cooked rice, fresh buttermilk, coconut water, sweet fruits (melons, pears), soaked raisins',
        'Diet_Avoid': 'Spicy curries, vinegar, deep-fried foods, caffeine, alcohol, sour citrus fruits',
        'How_Treatment_Affects_Your_Body_Type': 'Reduces sharp Pitta heat, soothes the gut lining, enhances digestive calm',
        'Condition_Explanation': 'Gastritis (Amlapitta) is caused by aggravated Pitta dosha producing excess heat and acidity in the digestive tract.'
    },
    'Hypertension': {
        'Ayurvedic_Herbs_Sanskrit': 'Arjuna, Punarnava, Brahmi, Shankhpushpi',
        'Ayurvedic_Herbs_English': 'Arjuna bark, Boerhavia, Brahmi, Convolvulus',
        'Herbs_Effects': 'Strengthens cardiac muscles, normalizes blood pressure, calms the nervous system',
        'Ayurvedic_Therapies_Sanskrit': 'Shirodhara, Abhyanga, Hrid Basti, Pranayama',
        'Ayurvedic_Therapies_English': 'Warm oil forehead therapy, Gentle body massage, Chest oil pooling, Alternate nostril breathing',
        'Therapies_Effects': 'Relieves mental stress, induces vascular relaxation, balances Vata-Pitta circulation',
        'Dietary_Recommendations': 'Low sodium diet, fresh seasonal vegetables, adequate hydration, avoid stimulants',
        'Diet_Eat': 'Pomegranate, watermelon, cucumber, whole grains, coriander-cumin water',
        'Diet_Avoid': 'Excess salt, processed snacks, pickles, red meat, caffeine',
        'How_Treatment_Affects_Your_Body_Type': 'Calms hyperactive Vata and Pitta, stabilizes circulation, protects heart tissue',
        'Condition_Explanation': 'Hypertension (Rakta Gata Vata) involves disturbed vascular Vata flow compounded by Pitta-induced stress.'
    },
    'Asthma': {
        'Ayurvedic_Herbs_Sanskrit': 'Vasa, Kantakari, Bharangi, Pushkarmool',
        'Ayurvedic_Herbs_English': 'Malabar nut, Yellow-berried nightshade, Bharangi, Elecampane',
        'Herbs_Effects': 'Bronchodilator, expectorant, reduces bronchial inflammation, improves lung capacity',
        'Ayurvedic_Therapies_Sanskrit': 'Swedana, Nasya, Dhumapana, Pranayama',
        'Ayurvedic_Therapies_English': 'Herbal steam, Medicated nasal drops, Herbal inhalation, Yogic breathing',
        'Therapies_Effects': 'Liquefies trapped mucus, clears respiratory channels (Pranavaha Srotas), enhances vital capacity',
        'Dietary_Recommendations': 'Warm, light meals; avoid dairy, chilled foods, and bananas during active congestion',
        'Diet_Eat': 'Warm soups, ginger water, honey with black pepper, warm herbal teas',
        'Diet_Avoid': 'Ice water, cheese, heavy curds, cold milk, fried foods',
        'How_Treatment_Affects_Your_Body_Type': 'Pacifies obstructed Vata and heavy Kapha in the chest channels, aiding clear respiration',
        'Condition_Explanation': 'Asthma (Tamaka Shwasa) arises when Vata pushed by Kapha obstructs the air pathways in the lungs.'
    },
    'Insomnia': {
        'Ayurvedic_Herbs_Sanskrit': 'Brahmi, Shankhpushpi, Ashwagandha, Jatamansi',
        'Ayurvedic_Herbs_English': 'Brahmi, Convolvulus, Winter Cherry, Indian Spikenard',
        'Herbs_Effects': 'Nervine tonic, calms restless thoughts, promotes natural deep sleep, reduces cortisol',
        'Ayurvedic_Therapies_Sanskrit': 'Shirodhara, Padabhyanga, Abhyanga, Yoga Nidra',
        'Ayurvedic_Therapies_English': 'Forehead oil stream, Foot massage with warm oil, Full body massage, Guided relaxation',
        'Therapies_Effects': 'Soothes cranial nerves, relaxes muscle tension, grounds unstable Vata energy',
        'Dietary_Recommendations': 'Warm milk with nutmeg or cardamom before bed; avoid late screen time and heavy dinners',
        'Diet_Eat': 'Warm whole milk, oatmeal, cooked vegetables, soaked almonds, chamomile or licorice infusion',
        'Diet_Avoid': 'Coffee, dark chocolate, energy drinks, late heavy meals',
        'How_Treatment_Affects_Your_Body_Type': 'Grounds nervous Vata dosha, cools mental Pitta, facilitates natural restorative sleep',
        'Condition_Explanation': 'Insomnia (Anidra) is primarily a manifestation of agitated Tarpaka Kapha and elevated Prana Vata disrupting mental tranquility.'
    },
    'Fever': {
        'Ayurvedic_Herbs_Sanskrit': 'Tulasi, Sunthi, Maricha, Pippali, Haridra',
        'Ayurvedic_Herbs_English': 'Holy Basil, Dry Ginger, Black Pepper, Long Pepper, Turmeric',
        'Herbs_Effects': 'Antipyretic, digestive stimulant, antimicrobial, breaks sweat and reduces body heat',
        'Ayurvedic_Therapies_Sanskrit': 'Langhana, Swedana, Kashaya Sevana',
        'Ayurvedic_Therapies_English': 'Light therapeutic fasting, Gentle steam, Herbal decoction intake',
        'Therapies_Effects': 'Kindles digestive fire (Agni), burns metabolic toxins (Ama), lowers core body temperature',
        'Dietary_Recommendations': 'Very light liquid diet, warm water, avoid heavy solid foods until temperature normalizes',
        'Diet_Eat': 'Moong dal soup, rice gruel (kanji), boiled lukewarm water, herbal tea',
        'Diet_Avoid': 'Solid heavy grains, dairy, cold foods, oily preparations',
        'How_Treatment_Affects_Your_Body_Type': 'Eliminates trapped Pitta and Ama, restores normal metabolic thermoregulation',
        'Condition_Explanation': 'Fever (Jwara) is considered the king of diseases in Ayurveda, caused by impaired Agni and Ama spreading heat through the circulation.'
    }
}

def load_ayurvedic_database():
    """
    Loads and caches the Ayurvedic recommendations database from the enhanced CSV,
    falling back to the comprehensive built-in dictionary if the CSV is unavailable.
    """
    global _AYURVEDIC_CACHE
    if _AYURVEDIC_CACHE is not None:
        return _AYURVEDIC_CACHE

    db = {}
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enhanced_ayurvedic_treatment_dataset.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            required_cols = [
                'Disease', 'Ayurvedic_Herbs_Sanskrit', 'Ayurvedic_Herbs_English',
                'Herbs_Effects', 'Ayurvedic_Therapies_Sanskrit', 'Ayurvedic_Therapies_English',
                'Therapies_Effects', 'Dietary_Recommendations', 'How_Treatment_Affects_Your_Body_Type'
            ]
            if all(col in df.columns for col in required_cols):
                for _, row in df.iterrows():
                    disease_name = str(row['Disease']).strip()
                    if disease_name and disease_name not in db:
                        db[disease_name] = {
                            'Ayurvedic_Herbs_Sanskrit': str(row['Ayurvedic_Herbs_Sanskrit']),
                            'Ayurvedic_Herbs_English': str(row['Ayurvedic_Herbs_English']),
                            'Herbs_Effects': str(row['Herbs_Effects']),
                            'Ayurvedic_Therapies_Sanskrit': str(row['Ayurvedic_Therapies_Sanskrit']),
                            'Ayurvedic_Therapies_English': str(row['Ayurvedic_Therapies_English']),
                            'Therapies_Effects': str(row['Therapies_Effects']),
                            'Dietary_Recommendations': str(row['Dietary_Recommendations']),
                            'How_Treatment_Affects_Your_Body_Type': str(row['How_Treatment_Affects_Your_Body_Type']),
                        }
        except Exception as e:
            print(f"Warning: Could not parse CSV dataset: {e}")

    # Merge built-in database entries
    for k, v in BUILTIN_AYURVEDIC_DATABASE.items():
        if k not in db:
            db[k] = v

    _AYURVEDIC_CACHE = db
    return _AYURVEDIC_CACHE

def get_ayurvedic_record(disease_name: str, dosha: str = "Vata") -> dict:
    """
    Finds the most suitable Ayurvedic treatment record for a predicted disease.
    Matches exact names, parentheses content, and key stems.
    """
    db = load_ayurvedic_database()
    disease_str = str(disease_name).strip()

    # 1. Exact match
    if disease_str in db:
        return db[disease_str]

    # 2. Case-insensitive exact match
    disease_lower = disease_str.lower()
    for name, record in db.items():
        if name.lower() == disease_lower:
            return record

    # 3. Extract text inside parentheses, e.g., 'Sandhivata (Arthritis)' -> 'Arthritis'
    if '(' in disease_str and ')' in disease_str:
        inner = disease_str.split('(')[1].split(')')[0].strip()
        outer = disease_str.split('(')[0].strip()
        for candidate in [inner, outer]:
            for name, record in db.items():
                if candidate.lower() in name.lower() or name.lower() in candidate.lower():
                    return record

    # 4. Keyword fuzzy match
    keywords = [w for w in disease_lower.replace('(', ' ').replace(')', ' ').split() if len(w) > 3]
    for kw in keywords:
        for name, record in db.items():
            if kw in name.lower():
                return record

    # 5. Default fallback based on Dosha
    return {
        'Ayurvedic_Herbs_Sanskrit': 'Amalaki, Haridra, Tulasi, Ashwagandha',
        'Ayurvedic_Herbs_English': 'Indian Gooseberry, Turmeric, Holy Basil, Winter Cherry',
        'Herbs_Effects': 'Enhances immunity (Ojas), anti-inflammatory, balances metabolic doshas',
        'Ayurvedic_Therapies_Sanskrit': 'Abhyanga, Pranayama, Swedana',
        'Ayurvedic_Therapies_English': 'Therapeutic oil massage, Breathwork, Gentle steam therapy',
        'Therapies_Effects': 'Harmonizes vital energy, detoxifies channels, restores systemic vitality',
        'Dietary_Recommendations': 'Warm, fresh, cooked meals; light soups; avoid processed, heavy foods',
        'Diet_Eat': 'Fresh seasonal vegetables, whole grains, herbal teas, warm soups',
        'Diet_Avoid': 'Cold drinks, stale or deep-fried foods, artificial preservatives',
        'How_Treatment_Affects_Your_Body_Type': f'Calms and balances {dosha} constitution, restoring natural equilibrium.',
        'Condition_Explanation': f'A disturbance in bodily equilibrium requiring gentle herbal rejuvenation and lifestyle adjustments.'
    }

def get_fallback_recommendations(user_data: dict, predicted_disease: str, confidence: float) -> str:
    """
    Generates a structured, compassionate Ayurvedic plan using the local database
    when Google Gemini API is unavailable or offline.
    """
    dosha = user_data.get('Body_Type_Dosha_Sanskrit', 'Vata')
    season = user_data.get('Season', 'Current Season')
    weather = user_data.get('Weather', 'Current Weather')
    
    rec = get_ayurvedic_record(predicted_disease, dosha)
    
    herbs_sanskrit = rec.get('Ayurvedic_Herbs_Sanskrit', 'Tulasi, Ginger, Haridra')
    herbs_english = rec.get('Ayurvedic_Herbs_English', 'Holy Basil, Ginger, Turmeric')
    herbs_effects = rec.get('Herbs_Effects', 'Strengthens vitality, reduces inflammation')
    
    therapies_sanskrit = rec.get('Ayurvedic_Therapies_Sanskrit', 'Abhyanga, Swedana, Pranayama')
    therapies_english = rec.get('Ayurvedic_Therapies_English', 'Warm oil massage, Herbal steam, Yogic breathing')
    therapies_effects = rec.get('Therapies_Effects', 'Detoxifies channels, relieves tension')
    
    diet_eat = rec.get('Diet_Eat', 'Warm cooked soups, seasonal vegetables, herbal teas, light grains')
    diet_avoid = rec.get('Diet_Avoid', 'Chilled beverages, excessively greasy or deep-fried foods, heavy processed snacks')
    
    explanation = rec.get('Condition_Explanation', f'{predicted_disease} indicates an imbalance influenced by constitutional and environmental elements.')
    body_effect = rec.get('How_Treatment_Affects_Your_Body_Type', f'These therapeutic measures specifically pacify aggravated {dosha} dosha.')

    output = f"""💖 Your Ayurvedic Diagnosis

Predicted Disease: {predicted_disease} [Confidence Level: {confidence:.0f}%] [Offline Mode]

Based on your profile and symptoms, you are experiencing {predicted_disease}.
From an Ayurvedic viewpoint, this reflects an aggravated {dosha} dosha influenced by the {season} season and {weather} weather. Restoring your body's Agni (digestive fire) and clearing Srotas (micro-channels) is key to recovery.

🌿 Your Personalized Ayurvedic Plan

🩺 Condition Explained
{explanation}

Ayurvedic Medicinal Herbs
- Sanskrit: {herbs_sanskrit}
- English: {herbs_english}
- Effects: {herbs_effects}

💆 Ayurvedic Therapies
- Sanskrit: {therapies_sanskrit}
- English: {therapies_english}
- Effects: {therapies_effects}

🥗 Dietary Recommendations
Focus on nourishing foods that pacify your current imbalance:

Eat This:
- {diet_eat}
- Warm, light, and easily digestible home-cooked meals
- Healthy fats like pure cow ghee and cold-pressed sesame oil in moderation

Avoid This:
- {diet_avoid}
- Cold, raw, iced foods and carbonated drinks
- Irregular meal timings and excessive fasting

🏃 Lifestyle Advice
- Practice gentle daily stretching, yoga asanas, or walking
- Maintain a consistent sleep schedule (sleep before 10 PM, wake before dawn)
- Practice alternate nostril breathing (Nadi Shodhana Pranayama) for 10 minutes daily
- Keep yourself comfortably warm and avoid cold drafts

🌿 Home Remedies & Precautions
- Sip warm ginger or tulsi infusion throughout the day
- Practice gentle self-massage (Abhyanga) with warm sesame or coconut oil before bathing
- Avoid heavy mental stress or late-night screen exposure

👤 How Treatment Affects Your Body Type
{body_effect}

⚠️ Important Note: This is an offline Ayurvedic recommendation provided for complementary wellness support. If your symptoms worsen or persist, please consult a qualified healthcare professional or physician.

🔌 Note: Running in offline mode (LLM unavailable). Recommendations generated from the verified Ayurvedic clinical database."""

    return output

# --- Step 4: Preprocessing & XAI Support ---
def preprocess_input(user_data):
    """
    Preprocesses raw user input into a scaled feature array ready for model inference.
    """
    user_df = pd.DataFrame([user_data])

    # Categorize Age into Age_Group if missing
    if 'Age_Group' not in user_df.columns and 'Age' in user_df.columns:
        age_val = user_df['Age'].iloc[0]
        if age_val < 13:
            user_df['Age_Group'] = "Child"
        elif 13 <= age_val < 20:
            user_df['Age_Group'] = "Adolescent"
        elif 20 <= age_val < 60:
            user_df['Age_Group'] = "Adult"
        else:
            user_df['Age_Group'] = "Senior"
    elif 'Age_Group' not in user_df.columns:
        user_df['Age_Group'] = "Adult"

    # Calculate BMI if not provided
    if 'BMI' not in user_df.columns:
        h = float(user_df['Height_cm'].iloc[0]) if 'Height_cm' in user_df.columns else 170.0
        w = float(user_df['Weight_kg'].iloc[0]) if 'Weight_kg' in user_df.columns else 70.0
        user_df['BMI'] = w / ((h / 100.0) ** 2)

    # Encode categorical features
    categorical_columns = [
        'Age_Group', 'Gender', 'Body_Type_Dosha_Sanskrit', 
        'Food_Habits', 'Current_Medication', 'Allergies', 'Season', 'Weather'
    ]
    
    for col in categorical_columns:
        encoded_values = []
        val = user_df[col].iloc[0] if col in user_df.columns else "Unknown"
        if col in encoders:
            try:
                encoded_values.append(encoders[col].transform([val])[0])
            except (ValueError, KeyError):
                encoded_values.append(0)
        else:
            encoded_values.append(0)
        user_df[f'{col}_encoded'] = encoded_values

    # Vectorize symptoms using TF-IDF
    symptoms_text = str(user_df['Symptoms'].iloc[0]) if 'Symptoms' in user_df.columns else ""
    if vectorizer is not None:
        tfidf_features = vectorizer.transform([symptoms_text]).toarray()
    else:
        tfidf_features = np.zeros((1, 10))

    tfidf_cols = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf_cols)

    # Combine features
    base_cols = [c for c in training_feature_columns if c in user_df.columns]
    base_feature_df = user_df[base_cols].reset_index(drop=True)
    final_df = pd.concat([base_feature_df, tfidf_df], axis=1)

    if scaler is not None:
        scaler_feature_names = scaler.get_feature_names_out()
        final_df = final_df.reindex(columns=scaler_feature_names, fill_value=0)
        scaled_features = scaler.transform(final_df)
    else:
        scaled_features = final_df.values

    return scaled_features

def get_symptom_weights(symptoms_text: str):
    """
    Computes explainable AI (XAI) feature contribution weights for input symptoms
    using the fitted TF-IDF vectorizer. Returns list of {symptom, weight}.
    """
    if not symptoms_text or vectorizer is None:
        return []

    try:
        tfidf_matrix = vectorizer.transform([symptoms_text])
        feature_names = vectorizer.get_feature_names_out()
        nonzero_indices = tfidf_matrix.nonzero()[1]
        
        breakdown = []
        for idx in nonzero_indices:
            score = float(tfidf_matrix[0, idx])
            breakdown.append({
                "symptom": str(feature_names[idx]),
                "weight": round(score * 100, 1)
            })

        breakdown.sort(key=lambda x: x["weight"], reverse=True)

        if not breakdown:
            raw_tokens = [s.strip() for s in symptoms_text.split(",") if s.strip()]
            for tok in raw_tokens[:4]:
                breakdown.append({"symptom": tok, "weight": 50.0})

        return breakdown[:6]
    except Exception:
        return []

# --- Step 5: Generative AI Reasoning ---
def get_llm_validation_and_explanation(user_data, ml_prediction, confidence):
    """
    Uses the Gemini LLM to validate the ML prediction and provide a detailed,
    personalized Ayurvedic explanation. Automatically falls back to local clinical
    database if LLM or network is unavailable.
    """
    if not GEMINI_API_KEY or genai_client is None:
        return get_fallback_recommendations(user_data, ml_prediction, confidence)

    prompt = f"""You are an expert Ayurvedic health assistant. Your task is to analyze a user's health data and an initial model prediction, then provide a final, trustworthy, and personalized Ayurvedic diagnosis and plan.

**User's Health Profile:**
- **Symptoms:** {user_data.get('Symptoms', 'N/A')}
- **Age:** {user_data.get('Age', 'N/A')}
- **Gender:** {user_data.get('Gender', 'N/A')}
- **Body Type (Dosha):** {user_data.get('Body_Type_Dosha_Sanskrit', 'N/A')}
- **Food Habits:** {user_data.get('Food_Habits', 'N/A')}
- **Season:** {user_data.get('Season', 'N/A')}
- **Weather:** {user_data.get('Weather', 'N/A')}
- **Height:** {user_data.get('Height_cm', 'N/A')} cm
- **Weight:** {user_data.get('Weight_kg', 'N/A')} kg

**Initial Analysis (Internal Use Only):**
- **Predicted Condition:** {ml_prediction}
- **Initial Confidence:** {confidence:.2f}%

**Your Instructions:**
1. Analyze and Diagnose: Determine the final diagnosis based on Ayurvedic principles and symptom presentation.
2. Generate Response: Follow this clean structure without markdown asterisks or bolding:

💖 Your Ayurvedic Diagnosis

Predicted Disease: [Your Final Diagnosis Here] [Confidence Level: in %]

Based on your profile, it seems you are experiencing [Your Final Diagnosis Here].
[Provide a brief, compassionate explanation connecting symptoms, body type, weather, and season.]

🌿 Your Personalized Ayurvedic Plan

🩺 Condition Explained
[Explain the condition in simple Ayurvedic terms with English terminology in parentheses.]

Ayurvedic Medicinal Herbs
- [3-4 specific herbs or formulations with Sanskrit and English names]

💆 Ayurvedic Therapies
- [2-3 traditional procedures with English equivalents]

🥗 Dietary Recommendations
Eat This:
- [5-6 specific beneficial food items]

Avoid This:
- [3-4 specific food items to avoid]

🏃 Lifestyle Advice
- [3-4 actionable daily lifestyle habits]

🌿 Home Remedies & Precautions
- [3-4 safe home remedies]

👤 How Treatment Affects Your Body Type
[Explanation of how the treatment restores balance to the user's specific constitution]

⚠️ Important Note: This plan is for gentle complementary support. If symptoms worsen, please consult a qualified healthcare professional."""

    try:
        gemini_model = genai_client.GenerativeModel('gemini-2.5-flash')
        response = gemini_model.generate_content(prompt)
        if response and response.text:
            return response.text
        return get_fallback_recommendations(user_data, ml_prediction, confidence)
    except Exception as e:
        print(f"Notice: Gemini API unavailable ({e}). Gracefully switching to offline Ayurvedic database.")
        return get_fallback_recommendations(user_data, ml_prediction, confidence)

# --- Step 6: Unified ArogyaAI Class Interface ---
class ArogyaAI:
    """
    Unified predictor and recommendation engine for Arogya AI.
    Provides an object-oriented interface compatible with demo.py and external clients.
    """
    def __init__(self):
        self.model_components = model_components
        self.model = model
        self.scaler = scaler
        self.vectorizer = vectorizer
        self.encoders = encoders
        self.training_feature_columns = training_feature_columns

    def predict(self, user_data: dict):
        if self.model is None:
            raise ValueError("Model is not loaded. Please train or provide random_forest_model.pkl.")
        scaled_features = preprocess_input(user_data)
        raw_pred = self.model.predict(scaled_features)[0]
        probabilities = self.model.predict_proba(scaled_features)[0]
        confidence = float(np.max(probabilities))
        
        if 'Disease' in self.encoders:
            predicted_disease = self.encoders['Disease'].inverse_transform([raw_pred])[0]
        else:
            predicted_disease = str(raw_pred)
            
        return predicted_disease, confidence, probabilities

    def predict_disease_with_recommendations(self, user_data: dict):
        predicted_disease, confidence, _ = self.predict(user_data)
        dosha = user_data.get('Body_Type_Dosha_Sanskrit', 'Vata')
        
        rec = get_ayurvedic_record(predicted_disease, dosha)
        formatted_text = get_llm_validation_and_explanation(user_data, predicted_disease, confidence * 100)
        
        return {
            'Predicted_Disease': predicted_disease,
            'Confidence': float(confidence),
            'Ayurvedic_Herbs_Sanskrit': rec.get('Ayurvedic_Herbs_Sanskrit', 'Tulasi, Ginger, Haridra'),
            'Ayurvedic_Herbs_English': rec.get('Ayurvedic_Herbs_English', 'Holy Basil, Ginger, Turmeric'),
            'Herbs_Effects': rec.get('Herbs_Effects', 'Boosts immunity, reduces inflammation'),
            'Ayurvedic_Therapies_Sanskrit': rec.get('Ayurvedic_Therapies_Sanskrit', 'Abhyanga, Swedana'),
            'Ayurvedic_Therapies_English': rec.get('Ayurvedic_Therapies_English', 'Warm oil massage, Steam therapy'),
            'Therapies_Effects': rec.get('Therapies_Effects', 'Detoxifies channels, relieves stiffness'),
            'Dietary_Recommendations': rec.get('Dietary_Recommendations', 'Warm, light, cooked meals'),
            'How_Treatment_Affects_Your_Body_Type': rec.get('How_Treatment_Affects_Your_Body_Type', f'Nourishes and pacifies {dosha} dosha.'),
            'recommendation': formatted_text
        }

# --- Step 7: CLI Assessment Interface ---
def get_safe_number_input(prompt, converter=int):
    """
    Safely gets numeric input from the user, handling ValueErrors.
    """
    while True:
        try:
            return converter(input(prompt))
        except ValueError:
            print(f"Invalid input. Please enter a valid {'number' if converter == float else 'integer'}.")

def main():
    print("=" * 60)
    print("AROGYA AI - Integrated ML + LLM Prediction System")
    print("=" * 60)
    print("Please provide your details to receive a personalized analysis.")
    print("-" * 60)

    symptoms = input("Enter your symptoms (comma-separated): ")
    age = get_safe_number_input("Enter your age: ")
    height_cm = get_safe_number_input("Enter your height (cm): ")
    weight_kg = get_safe_number_input("Enter your weight (kg): ")
    gender = input("Enter your gender: ")

    body_type_english = input("Enter your general body type (e.g., Thin, Medium, Heavy): ").strip().lower()
    dosha_map = {
        "thin": "Vata",
        "medium": "Pitta",
        "heavy": "Kapha"
    }
    body_type_sanskrit = dosha_map.get(body_type_english, "Vata")

    food_habits = input("Enter your food habits (e.g., Vegetarian, Non-Vegetarian, Mixed): ")
    current_medication = input("Enter your current medication (if any, otherwise type 'None'): ")
    allergies = input("Enter any allergies (if any, otherwise type 'None'): ")
    season = input("Enter the current season (e.g., Summer, Monsoon, Winter): ")
    weather = input("Enter the current weather (e.g., Hot, Humid, Cold): ")

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

    print("\nAnalyzing your information...")
    try:
        engine = ArogyaAI()
        predicted_disease, confidence, _ = engine.predict(user_data)
        print(f"   => ML Model Prediction: '{predicted_disease}' (Confidence: {confidence:.2%})")
    except Exception as e:
        print(f"   => Error during analysis: {e}")
        return

    llm_explanation = get_llm_validation_and_explanation(user_data, predicted_disease, confidence * 100)
    print("\n" + "=" * 60)
    print("Arogya AI - Personalized Ayurvedic Analysis")
    print("=" * 60)
    print(llm_explanation)

if __name__ == "__main__":
    main()
