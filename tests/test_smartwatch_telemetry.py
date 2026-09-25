import unittest
from fastapi.testclient import TestClient
from backend.index import app
from telemetry_nadi_engine import (
    SmartwatchTelemetry,
    analyze_smartwatch_nadi,
    get_sample_telemetry
)

class TestSmartwatchTelemetry(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_vata_sarpa_gati_classification(self):
        """Test low HRV and elevated stress flags Vata Nadi (Sarpa Gati)."""
        data = SmartwatchTelemetry(
            device_name="Apple Watch",
            resting_heart_rate=78.0,
            hrv_rmssd=20.0,
            skin_temp_celsius=36.2,
            stress_index=75.0,
            sleep_efficiency_pct=65.0
        )
        report = analyze_smartwatch_nadi(data)
        self.assertEqual(report["digital_nadi"]["dominant_dosha"], "Vata")
        self.assertIn("Sarpa Gati", report["digital_nadi"]["gati_classification"])
        self.assertGreater(report["dosha_drift_matrix"]["vata_percentage"], 40.0)

    def test_pitta_manduka_gati_classification(self):
        """Test elevated resting heart rate and temperature flags Pitta Nadi (Manduka Gati)."""
        data = SmartwatchTelemetry(
            device_name="Garmin",
            resting_heart_rate=88.0,
            hrv_rmssd=40.0,
            skin_temp_celsius=37.5,
            stress_index=45.0,
            sleep_efficiency_pct=80.0
        )
        report = analyze_smartwatch_nadi(data)
        self.assertEqual(report["digital_nadi"]["dominant_dosha"], "Pitta")
        self.assertIn("Manduka Gati", report["digital_nadi"]["gati_classification"])
        self.assertGreater(report["dosha_drift_matrix"]["pitta_percentage"], 38.0)

    def test_kapha_hamsa_gati_classification(self):
        """Test resting bradycardia and high vagal HRV flags Kapha Nadi (Hamsa Gati)."""
        data = SmartwatchTelemetry(
            device_name="Fitbit",
            resting_heart_rate=54.0,
            hrv_rmssd=85.0,
            skin_temp_celsius=36.4,
            stress_index=15.0,
            sleep_efficiency_pct=95.0
        )
        report = analyze_smartwatch_nadi(data)
        self.assertEqual(report["digital_nadi"]["dominant_dosha"], "Kapha")
        self.assertIn("Hamsa Gati", report["digital_nadi"]["gati_classification"])
        self.assertGreater(report["dosha_drift_matrix"]["kapha_percentage"], 38.0)

    def test_ojas_vitality_index(self):
        """Verify Ojas Vitality score computation."""
        data = SmartwatchTelemetry(
            resting_heart_rate=60.0,
            hrv_rmssd=75.0,
            sleep_efficiency_pct=90.0,
            stress_index=20.0
        )
        report = analyze_smartwatch_nadi(data)
        ojas = report["ojas_vitality_index"]
        self.assertIn("score", ojas)
        self.assertGreaterEqual(ojas["score"], 70.0)
        self.assertIn("Optimal", ojas["status"])

    def test_api_telemetry_analyze_endpoint(self):
        """Test POST /api/telemetry/analyze endpoint."""
        sample = get_sample_telemetry("pitta_heat")
        response = self.client.post("/api/telemetry/analyze", json=sample)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("digital_nadi", data)
        self.assertIn("dosha_drift_matrix", data)
        self.assertIn("ojas_vitality_index", data)

    def test_api_telemetry_sample_endpoint(self):
        """Test GET /api/telemetry/sample endpoint."""
        response = self.client.get("/api/telemetry/sample?profile=kapha_calm")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("resting_heart_rate", data)
        self.assertEqual(data["resting_heart_rate"], 56.0)

    def test_api_predict_with_embedded_telemetry(self):
        """Test POST /api/predict fuses smartwatch telemetry with disease prediction."""
        payload = {
            "Symptoms": "headache, eye irritation, acid reflux",
            "Age": 32,
            "Height_cm": 175,
            "Weight_kg": 72,
            "Gender": "Male",
            "Body_Type_Dosha_Sanskrit": "Pitta",
            "Food_Habits": "Mixed",
            "Current_Medication": "None",
            "Allergies": "None",
            "Season": "Summer",
            "Weather": "Hot",
            "telemetry": get_sample_telemetry("pitta_heat")
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("digital_nadi_telemetry", data)
        self.assertIsNotNone(data["digital_nadi_telemetry"])
        self.assertEqual(data["digital_nadi_telemetry"]["digital_nadi"]["dominant_dosha"], "Pitta")

    def test_telemetry_physiological_bounds_validation(self):
        """Test that invalid physiological readings (e.g. HR = 15 bpm) fail schema validation (HTTP 422)."""
        invalid_payload = {
            "resting_heart_rate": 15.0, # Below 35 bpm physiological limit
            "hrv_rmssd": 50.0
        }
        response = self.client.post("/api/telemetry/analyze", json=invalid_payload)
        self.assertEqual(response.status_code, 422)

if __name__ == "__main__":
    unittest.main()
