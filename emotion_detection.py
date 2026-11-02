"""Client for the Watson NLP emotion detection model."""

import os
from typing import Any

import requests


EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
REQUEST_TIMEOUT_SECONDS = 30


def _empty_result() -> dict[str, float | None]:
    """Return the standard result shape when the service rejects a request."""
    result: dict[str, float | None] = {emotion: None for emotion in EMOTIONS}
    result["dominant_emotion"] = None
    return result


def emotion_detector(text_to_analyze: str) -> dict[str, float | str | None]:
    """Analyze text and return its five emotion scores and dominant emotion."""
    endpoint = os.environ.get("WATSON_NLP_URL")
    if not endpoint:
        raise RuntimeError("Set WATSON_NLP_URL to your Watson NLP endpoint.")

    response = requests.post(
        endpoint,
        json={"raw_document": {"text": text_to_analyze}},
        headers={"grpc-metadata-mm-model-id": MODEL_ID},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        return _empty_result()

    payload: Any = response.json()
    predictions = payload.get("emotionPredictions")
    if predictions:
        scores = predictions[0]["emotion"]
    else:
        scores = payload.get("emotion")
    if not isinstance(scores, dict):
        raise ValueError("Watson NLP response did not contain emotion scores.")

    result: dict[str, float | str | None] = {
        emotion: scores[emotion] for emotion in EMOTIONS
    }
    result["dominant_emotion"] = max(EMOTIONS, key=lambda emotion: result[emotion])
    return result
