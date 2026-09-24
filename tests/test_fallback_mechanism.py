import unittest
from unittest.mock import patch
from arogya_predict import (
    load_ayurvedic_database,
    get_ayurvedic_record,
    get_fallback_recommendations,
    get_llm_validation_and_explanation
)

class TestFallbackMechanism(unittest.TestCase):
    def setUp(self):
        self.sample_user = {
            "Symptoms": "stomach pain, acidity, burning sensation",
            "Age": 30,
            "Gender": "Female",
            "Body_Type_Dosha_Sanskrit": "Pitta",
            "Food_Habits": "Vegetarian",
            "Season": "Summer",
            "Weather": "Hot"
        }

    def test_ayurvedic_database_loading(self):
        """Verify the Ayurvedic treatment database loads and has valid entries."""
        db = load_ayurvedic_database()
        self.assertIsInstance(db, dict)
        self.assertGreater(len(db), 0)

    def test_ayurvedic_record_matching(self):
        """Test exact, fuzzy, and parenthesized matching of disease names."""
        rec1 = get_ayurvedic_record("Sandhivata (Arthritis)")
        self.assertIn("Ayurvedic_Herbs_Sanskrit", rec1)

        rec2 = get_ayurvedic_record("Diabetes")
        self.assertIn("Ayurvedic_Herbs_English", rec2)

        rec3 = get_ayurvedic_record("Unknown Rare Syndrome", dosha="Kapha")
        self.assertIn("How_Treatment_Affects_Your_Body_Type", rec3)

    def test_fallback_recommendations_format(self):
        """Verify the formatted fallback response contains all necessary Ayurvedic sections."""
        rec_text = get_fallback_recommendations(self.sample_user, "Amlapitta (Gastritis)", 98.5)
        self.assertIn("Your Ayurvedic Diagnosis", rec_text)
        self.assertIn("Predicted Disease: Amlapitta (Gastritis)", rec_text)
        self.assertIn("[Offline Mode]", rec_text)
        self.assertIn("Ayurvedic Medicinal Herbs", rec_text)
        self.assertIn("Ayurvedic Therapies", rec_text)
        self.assertIn("Dietary Recommendations", rec_text)
        self.assertIn("Eat This:", rec_text)
        self.assertIn("Avoid This:", rec_text)
        self.assertIn("Lifestyle Advice", rec_text)
        self.assertIn("Home Remedies & Precautions", rec_text)

    def test_automatic_fallback_on_llm_error(self):
        """Ensure get_llm_validation_and_explanation falls back seamlessly if LLM fails."""
        with patch("arogya_predict.GEMINI_API_KEY", "dummy_key"):
            with patch("arogya_predict.genai_client") as mock_client:
                mock_model = mock_client.GenerativeModel.return_value
                mock_model.generate_content.side_effect = Exception("API connection timed out")
                
                result = get_llm_validation_and_explanation(
                    self.sample_user, "Amlapitta (Gastritis)", 95.0
                )
                self.assertIn("[Offline Mode]", result)
                self.assertIn("Amlapitta (Gastritis)", result)

if __name__ == "__main__":
    unittest.main()
