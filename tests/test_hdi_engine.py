import unittest
from fastapi.testclient import TestClient
from backend.index import app
from hdi_engine import analyze_herb_drug_interactions

class TestHDIEngine(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_no_medications_safe(self):
        """Verify that reporting 'None' or empty medications returns SAFE status."""
        result = analyze_herb_drug_interactions(
            current_medications="None",
            recommended_herbs=["Ashwagandha", "Tulasi"],
            allergies="None"
        )
        self.assertFalse(result["has_alerts"])
        self.assertEqual(result["max_severity"], "SAFE")
        self.assertEqual(result["total_alerts"], 0)

    def test_anticoagulant_guggulu_high_risk(self):
        """Test Warfarin + Guggulu yields HIGH severity bleeding risk alert."""
        result = analyze_herb_drug_interactions(
            current_medications="Warfarin 5mg daily",
            recommended_herbs=["Guggulu", "Shallaki"],
            allergies="None"
        )
        self.assertTrue(result["has_alerts"])
        self.assertEqual(result["max_severity"], "HIGH")
        self.assertGreaterEqual(result["total_alerts"], 1)
        alert_mechanisms = [a["mechanism"] for a in result["alerts"]]
        self.assertTrue(any("platelet" in m.lower() or "hemorrhage" in m.lower() for m in alert_mechanisms))

    def test_antidiabetic_meshashringi_moderate_risk(self):
        """Test Metformin + Gymnema/Meshashringi yields MODERATE hypoglycemia alert."""
        result = analyze_herb_drug_interactions(
            current_medications="Metformin 500mg, Glimepiride 1mg",
            recommended_herbs=["Meshashringi", "Vijaysar", "Karela"],
            allergies="None"
        )
        self.assertTrue(result["has_alerts"])
        self.assertIn(result["max_severity"], ["HIGH", "MODERATE"])
        self.assertTrue(any("hypoglycemia" in a["mechanism"].lower() or "hypoglycemic" in a["mechanism"].lower() for a in result["alerts"]))

    def test_antihypertensive_yashtimadhu_contraindication(self):
        """Test Amlodipine + Licorice/Yashtimadhu yields HIGH pseudoaldosteronism alert."""
        result = analyze_herb_drug_interactions(
            current_medications="Amlodipine 5mg",
            recommended_herbs=["Yashtimadhu", "Amalaki"],
            allergies="None"
        )
        self.assertTrue(result["has_alerts"])
        self.assertEqual(result["max_severity"], "HIGH")
        self.assertTrue(any("potassium" in a["mechanism"].lower() or "retention" in a["mechanism"].lower() for a in result["alerts"]))

    def test_sedative_ashwagandha_additive_effect(self):
        """Test Alprazolam + Ashwagandha yields MODERATE CNS depression warning."""
        result = analyze_herb_drug_interactions(
            current_medications="Alprazolam 0.25mg at bedtime",
            recommended_herbs=["Ashwagandha", "Jatamansi"],
            allergies="None"
        )
        self.assertTrue(result["has_alerts"])
        self.assertEqual(result["max_severity"], "MODERATE")
        self.assertTrue(any("gabaergic" in a["mechanism"].lower() or "somnolence" in a["mechanism"].lower() for a in result["alerts"]))

    def test_allergy_cross_reactivity(self):
        """Test Aspirin allergy against Shallaki / Boswellia yields allergy alert."""
        result = analyze_herb_drug_interactions(
            current_medications="None",
            recommended_herbs=["Shallaki", "Guggulu"],
            allergies="Aspirin allergy"
        )
        self.assertTrue(result["has_alerts"])
        self.assertEqual(result["max_severity"], "HIGH")
        self.assertTrue(any(a["type"] == "ALLERGY_CONTRAINDICATION" for a in result["alerts"]))

    def test_api_check_hdi_endpoint(self):
        """Test standalone POST /api/check-hdi endpoint."""
        payload = {
            "medications": "Warfarin, Metformin",
            "herbs": "Guggulu, Karela, Turmeric",
            "allergies": "None"
        }
        response = self.client.post("/api/check-hdi", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["has_alerts"])
        self.assertEqual(data["max_severity"], "HIGH")
        self.assertGreaterEqual(data["total_alerts"], 2)

    def test_api_predict_includes_hdi_alerts(self):
        """Test POST /api/predict contains hdi_safety_alerts."""
        payload = {
            "Symptoms": "joint pain, morning stiffness",
            "Age": 55,
            "Height_cm": 170,
            "Weight_kg": 75,
            "Gender": "Male",
            "Body_Type_Dosha_Sanskrit": "Vata",
            "Food_Habits": "Vegetarian",
            "Current_Medication": "Warfarin",
            "Allergies": "None",
            "Season": "Winter",
            "Weather": "Cold"
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("hdi_safety_alerts", data)
        self.assertTrue(data["hdi_safety_alerts"]["has_alerts"])
        self.assertEqual(data["hdi_safety_alerts"]["max_severity"], "HIGH")

if __name__ == "__main__":
    unittest.main()
