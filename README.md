# oaqjp-final-project-emb-ai

Final Project

Emotion detection web application using the Watson NLP emotion model.

## Setup

1. Install the dependencies with `python -m pip install -r requirements.txt`.
2. Set `WATSON_NLP_URL` to the complete URL of your Watson NLP emotion-analysis endpoint.
3. Run the application with `python server.py` and open `http://127.0.0.1:5000`.

The endpoint should accept a JSON body containing `raw_document.text` and the
`grpc-metadata-mm-model-id` header for
`emotion_aggregated-workflow_lang_en_stock`.

Run unit tests with `python -m unittest -v test_emotion_detection.py`.
Run static analysis with `python -m pylint server.py`.
