"""Unit tests for the Watson NLP emotion detector."""

import os
import unittest
from unittest.mock import Mock, patch

from emotion_detection import EMOTIONS, MODEL_ID, emotion_detector
from server import app


class EmotionDetectorTests(unittest.TestCase):
    """Verify response formatting and service error handling."""

    def setUp(self) -> None:
        self.environment = patch.dict(os.environ, {"WATSON_NLP_URL": "https://example.test"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    @patch("emotion_detection.requests.post")
    def test_returns_all_scores_and_dominant_emotion(self, post: Mock) -> None:
        scores = {
            "anger": 0.1,
            "disgust": 0.2,
            "fear": 0.3,
            "joy": 0.9,
            "sadness": 0.4,
        }
        post.return_value.status_code = 200
        post.return_value.json.return_value = {
            "emotionPredictions": [{"emotion": scores}]
        }

        result = emotion_detector("I feel delighted.")

        self.assertEqual(
            result,
            {**scores, "dominant_emotion": "joy"},
        )
        post.assert_called_once_with(
            "https://example.test",
            json={"raw_document": {"text": "I feel delighted."}},
            headers={"grpc-metadata-mm-model-id": MODEL_ID},
            timeout=30,
        )

    @patch("emotion_detection.requests.post")
    def test_returns_null_scores_for_bad_request(self, post: Mock) -> None:
        post.return_value.status_code = 400

        result = emotion_detector("")

        self.assertEqual(
            result,
            {**dict.fromkeys(EMOTIONS), "dominant_emotion": None},
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_requires_watson_endpoint(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "WATSON_NLP_URL"):
            emotion_detector("This is a test.")

    @patch("server.emotion_detector")
    def test_blank_form_input_displays_error(self, detector: Mock) -> None:
        response = app.test_client().post("/", data={"textToAnalyze": "  "})

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Please enter text to analyze.", response.data)
        detector.assert_not_called()

    @patch("server.emotion_detector")
    def test_form_displays_emotion_result(self, detector: Mock) -> None:
        detector.return_value = {
            "anger": 0.1,
            "disgust": 0.2,
            "fear": 0.3,
            "joy": 0.9,
            "sadness": 0.4,
            "dominant_emotion": "joy",
        }

        response = app.test_client().post(
            "/",
            data={"textToAnalyze": "I feel delighted."},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Dominant emotion:", response.data)
        self.assertIn(b"joy", response.data)
        detector.assert_called_once_with("I feel delighted.")


if __name__ == "__main__":
    unittest.main()
