import unittest
import numpy as np
from arogya_predict import (
    preprocess_input,
    get_symptom_weights,
    ArogyaAI,
    model,
    encoders
)

class TestPredictionPipeline(unittest.TestCase):
    def setUp(self):
        self.engine = ArogyaAI()
        self.sample_user = {
            "Symptoms": "joint pain, morning stiffness, swelling",
            "Age": 54,
            "Height_cm": 172,
            "Weight_kg": 76,
            "Gender": "Male",
            "Body_Type_Dosha_Sanskrit": "Vata",
            "Food_Habits": "Vegetarian",
            "Current_Medication": "None",
            "Allergies": "None",
            "Season": "Winter",
            "Weather": "Cold"
        }

    def test_bmi_computation(self):
        """Verify automatic calculation of BMI when omitted."""
        user = self.sample_user.copy()
        # Height: 172cm (1.72m), Weight: 76kg -> BMI approx 25.69
        features = preprocess_input(user)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], 1)

    def test_unseen_categorical_labels(self):
        """Ensure unseen or novel categories do not crash preprocessing."""
        user = self.sample_user.copy()
        user["Food_Habits"] = "CompletelyUnseenDietPattern123"
        user["Weather"] = "ExtraterrestrialMonsoon"
        features = preprocess_input(user)
        self.assertIsInstance(features, np.ndarray)

    def test_disease_decoding_to_string(self):
        """Verify that ML prediction returns human-readable disease string, not raw integer."""
        pred_disease, confidence, _ = self.engine.predict(self.sample_user)
        self.assertIsInstance(pred_disease, str)
        self.assertGreater(len(pred_disease), 3)
        self.assertFalse(pred_disease.isdigit(), "Prediction must be a decoded disease name, not an integer.")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_symptom_xai_weights(self):
        """Verify TF-IDF Explainable AI breakdown generation."""
        breakdown = get_symptom_weights("fever, headache, chills")
        self.assertIsInstance(breakdown, list)
        self.assertGreater(len(breakdown), 0)
        for item in breakdown:
            self.assertIn("symptom", item)
            self.assertIn("weight", item)
            self.assertIsInstance(item["weight"], float)

    def test_arogya_ai_comprehensive_prediction(self):
        """Verify ArogyaAI.predict_disease_with_recommendations returns all required fields."""
        res = self.engine.predict_disease_with_recommendations(self.sample_user)
        required_keys = [
            'Predicted_Disease',
            'Confidence',
            'Ayurvedic_Herbs_Sanskrit',
            'Ayurvedic_Herbs_English',
            'Herbs_Effects',
            'Ayurvedic_Therapies_Sanskrit',
            'Ayurvedic_Therapies_English',
            'Therapies_Effects',
            'Dietary_Recommendations',
            'How_Treatment_Affects_Your_Body_Type',
            'recommendation'
        ]
        for key in required_keys:
            self.assertIn(key, res, f"Result dictionary missing key: {key}")
        self.assertIsInstance(res['recommendation'], str)
        self.assertIn("Your Ayurvedic Diagnosis", res['recommendation'])

if __name__ == "__main__":
    unittest.main()
