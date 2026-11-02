"""Flask web interface for Watson NLP emotion detection."""

from flask import Flask, render_template, request

from emotion_detection import emotion_detector


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the form and analyze non-empty submitted text."""
    result = None
    error = None
    text_to_analyze = ""

    if request.method == "POST":
        text_to_analyze = request.form.get("textToAnalyze", "").strip()
        if not text_to_analyze:
            error = "Please enter text to analyze."
        else:
            result = emotion_detector(text_to_analyze)
            if result["dominant_emotion"] is None:
                error = "The emotion service could not process this text."

    return render_template(
        "index.html",
        text_to_analyze=text_to_analyze,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run()
