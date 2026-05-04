from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add the root directory to sys.path so we can import from arogya_predict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arogya_predict import preprocess_input, get_llm_validation_and_explanation, model

app = FastAPI(title="ArogyaAI API")

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

@app.post("/api/predict")
def predict_disease(data: PredictRequest):
    try:
        user_dict = data.dict()
        
        # Preprocess using the ML model
        scaled_features = preprocess_input(user_dict)
        
        # Predict
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        confidence = float(max(probabilities) * 100)
        
        if confidence < 35.0:
            return {
                "prediction": "Inconclusive Data",
                "confidence": confidence,
                "recommendation": "The AI confidence is too low based on your provided symptoms. Please consult a doctor immediately.",
                "ml_prediction": prediction
            }
        
        # Get LLM generated response
        llm_response = get_llm_validation_and_explanation(user_dict, prediction, confidence)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "recommendation": llm_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    return {"status": "ArogyaAI Backend is healthy!"}
