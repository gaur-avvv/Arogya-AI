import sys
from unittest.mock import MagicMock

# Mocking dependencies before importing the module under test
mock_pd = MagicMock()
mock_np = MagicMock()
mock_joblib = MagicMock()

sys.modules["pandas"] = mock_pd
sys.modules["numpy"] = mock_np
sys.modules["joblib"] = mock_joblib
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.ensemble"] = MagicMock()
sys.modules["sklearn.preprocessing"] = MagicMock()
sys.modules["sklearn.feature_extraction"] = MagicMock()
sys.modules["sklearn.feature_extraction.text"] = MagicMock()
sys.modules["sklearn.model_selection"] = MagicMock()
sys.modules["sklearn.metrics"] = MagicMock()
sys.modules["imblearn"] = MagicMock()
sys.modules["imblearn.over_sampling"] = MagicMock()

import unittest
from disease_prediction_system import ArogyaAIPredictor

class TestDiseasePredictionSystem(unittest.TestCase):
    def test_predict_without_training(self):
        """Test that calling predict_disease_with_recommendations before training raises ValueError."""
        predictor = ArogyaAIPredictor()
        user_data = {"Symptoms": "fever", "Age": 25}

        with self.assertRaises(ValueError) as cm:
            predictor.predict_disease_with_recommendations(user_data)

        self.assertEqual(str(cm.exception), "Model not trained. Please train the model first.")

if __name__ == "__main__":
    unittest.main()
