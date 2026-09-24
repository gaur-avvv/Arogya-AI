import unittest
from fastapi.testclient import TestClient
from backend.index import app

class TestBackendAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test the root welcome endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("health_check", data)

    def test_health_check_endpoint(self):
        """Test health check and model loading state."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_predict_disease_endpoint_success(self):
        """Test POST /api/predict with standard clinical input."""
        payload = {
            "Symptoms": "joint pain, morning stiffness, swelling",
            "Age": 55,
            "Height_cm": 170,
            "Weight_kg": 75,
            "Gender": "Male",
            "Body_Type_Dosha_Sanskrit": "Vata",
            "Food_Habits": "Vegetarian",
            "Current_Medication": "None",
            "Allergies": "None",
            "Season": "Winter",
            "Weather": "Cold"
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("prediction", data)
        self.assertIsInstance(data["prediction"], str)
        self.assertFalse(data["prediction"].isdigit(), "Prediction must be a human-readable disease name.")
        
        self.assertIn("confidence", data)
        self.assertIsInstance(data["confidence"], (int, float))
        
        self.assertIn("recommendation", data)
        self.assertIsInstance(data["recommendation"], str)
        
        self.assertIn("xai_breakdown", data)
        self.assertIsInstance(data["xai_breakdown"], list)

    def test_predict_endpoint_validation_error(self):
        """Test that malformed inputs trigger FastAPI validation error (HTTP 422)."""
        malformed_payload = {
            "Symptoms": "fever",
            "Age": "invalid_not_an_int"
        }
        response = self.client.post("/api/predict", json=malformed_payload)
        self.assertEqual(response.status_code, 422)

if __name__ == "__main__":
    unittest.main()
