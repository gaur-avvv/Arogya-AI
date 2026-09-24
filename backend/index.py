from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add the root directory to sys.path so we can import from arogya_predict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arogya_predict import (
    preprocess_input,
    get_llm_validation_and_explanation,
    get_symptom_weights,
    get_ayurvedic_record,
    model,
    encoders
)
from hdi_engine import analyze_herb_drug_interactions

app = FastAPI(
    title="ArogyaAI API",
    description="Clinical Decision Support System combining deterministic ML disease prediction with Ayurvedic intelligence & Pharmacovigilance.",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    Symptoms: str
    Age: int
    Height_cm: int
    Weight_kg: int
    Gender: str
    Body_Type_Dosha_Sanskrit: str
    Food_Habits: str = "Mixed"
    Current_Medication: str = "None"
    Allergies: str = "None"
    Season: str = "Spring"
    Weather: str = "Clear"

class HDICheckRequest(BaseModel):
    medications: str
    herbs: str
    allergies: str = "None"

@app.post("/api/predict")
def predict_disease(data: PredictRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Machine learning model is not loaded. Please verify random_forest_model.pkl exists on the server."
        )

    try:
        user_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        
        # Preprocess features using trained scaler and vectorizer
        scaled_features = preprocess_input(user_dict)
        
        # Deterministic ML Prediction
        raw_prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        confidence = float(max(probabilities) * 100)
        
        # Decode the categorical encoded integer into human-readable disease name
        if 'Disease' in encoders:
            predicted_disease = str(encoders['Disease'].inverse_transform([raw_prediction])[0])
        else:
            predicted_disease = str(raw_prediction)

        # Compute symptom weight breakdown for Explainable AI (AI X-Ray)
        xai_breakdown = get_symptom_weights(data.Symptoms)
        
        # Perform Herb-Drug Interaction (HDI) Safety Screening
        rec = get_ayurvedic_record(predicted_disease, data.Body_Type_Dosha_Sanskrit)
        herbs_to_check = f"{rec.get('Ayurvedic_Herbs_Sanskrit', '')}, {rec.get('Ayurvedic_Herbs_English', '')}"
        hdi_safety_alerts = analyze_herb_drug_interactions(
            current_medications=data.Current_Medication,
            recommended_herbs=herbs_to_check,
            allergies=data.Allergies
        )

        # Clinical Guardrail: Check confidence threshold
        if confidence < 35.0:
            return {
                "prediction": "Inconclusive Data",
                "confidence": round(confidence, 1),
                "recommendation": "The AI confidence is too low based on your provided symptoms. Please consult a doctor immediately.",
                "ml_prediction": predicted_disease,
                "xai_breakdown": xai_breakdown,
                "hdi_safety_alerts": hdi_safety_alerts
            }
        
        # Get generative validation or offline Ayurvedic database plan
        llm_response = get_llm_validation_and_explanation(user_dict, predicted_disease, confidence)
        
        return {
            "prediction": predicted_disease,
            "confidence": round(confidence, 1),
            "recommendation": llm_response,
            "ml_prediction": predicted_disease,
            "xai_breakdown": xai_breakdown,
            "hdi_safety_alerts": hdi_safety_alerts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/check-hdi")
def check_hdi(data: HDICheckRequest):
    """
    Dedicated endpoint for practitioners and pharmacists to screen arbitrary
    allopathic medications against Ayurvedic herbs and botanical allergens.
    """
    try:
        return analyze_herb_drug_interactions(
            current_medications=data.medications,
            recommended_herbs=data.herbs,
            allergies=data.allergies
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {
        "message": "Welcome to ArogyaAI API. Visit /docs for Swagger UI documentation.",
        "health_check": "/api/health",
        "hdi_screening": "/api/check-hdi"
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ArogyaAI Backend",
        "model_loaded": model is not None,
        "hdi_engine_active": True
    }
