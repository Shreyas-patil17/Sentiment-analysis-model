import os
import re
from string import punctuation
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request, flash
from typing import List


# --- Flask App Initialization ---
app = Flask(__name__)
# Required for flashing messages (optional but good for user feedback)
app.secret_key = os.urandom(24) # Or set a fixed secret key

# --- Load Model and Vectorizer ---
# Load these once when the app starts for efficiency
MODEL_PATH = 'sentiment_model.joblib'
VECTORIZER_PATH = 'vectorizer.joblib'
model = None
vectorizer = None
model_load_error = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer file not found at {VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and Vectorizer loaded successfully.")
except Exception as e:
    model_load_error = f"FATAL ERROR: Failed to load model/vectorizer: {str(e)}"
    print(model_load_error) # Print error to console on startup

# --- Text Preprocessing Function (from your API code) ---
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))
def preprocess_text(text_data: List[str]) -> List[str]:
    """
    Preprocess text by removing punctuation, lowercasing, and removing stopwords
    """
    preprocessed_text = []
    stop_words = set(stopwords.words('english'))
    
    for sentence in text_data:
        # Remove punctuation
        sentence = re.sub(r'[^\w\s]', '', sentence)
        # Tokenize, lowercase, and remove stopwords
        tokens = word_tokenize(sentence)
        filtered_sentence = ' '.join([word.lower() for word in tokens if word.lower() not in stop_words])
        preprocessed_text.append(filtered_sentence)
    
    return preprocessed_text

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # Check if model loading failed during startup
    if model_load_error:
        flash(model_load_error, 'error') # Show error to user
        return render_template('index.html', prediction=None, probabilities=None, submitted_text=None)

    prediction_result = None
    probabilities_result = None
    error_message = None
    submitted_text = None

    if request.method == 'POST':
        submitted_text = request.form.get('news_text', '').strip()

        if not submitted_text:
            error_message = "Please enter some text to analyse."
        elif not model or not vectorizer:
             # This check is redundant if model_load_error is handled, but good safety
            error_message = "Model or Vectorizer not loaded properly. Check server logs."
        else:
            try:
                # 1. Preprocess
                cleaned_text_list = preprocess_text([submitted_text])
                cleaned_text = cleaned_text_list[0] # Get the single processed string

                # 2. Vectorize
                vectorized_text = vectorizer.transform([cleaned_text]).toarray()

                # 3. Predict
                prediction_code = model.predict(vectorized_text)[0]
                sentiments = {0: "Negative", 1: "Positive"} # Assuming 0=Neg, 1=Pos
                prediction_result = sentiments.get(prediction_code, "Unknown")

                # 4. Get Probabilities (Optional)
                try:
                    probabilities = model.predict_proba(vectorized_text)[0]
                    probabilities_result = {
                        "Negative": float(probabilities[0]),
                        "Positive": float(probabilities[1]) if len(probabilities) > 1 else None
                    }
                except AttributeError:
                    probabilities_result = None # Model doesn't support predict_proba

            except Exception as e:
                error_message = f"An error occurred during analysis: {str(e)}"
                print(f"Analysis Error: {e}") # Log the full error

        # Flash error message if one occurred
        if error_message:
            flash(error_message, 'error')

    # Render the template, passing results (or None)
    return render_template(
        'index.html',
        prediction=prediction_result,
        probabilities=probabilities_result,
        submitted_text=submitted_text # Pass submitted text back to keep it in textarea
    )

if __name__ == "__main__":
    # Use host='0.0.0.0' to make it accessible on your network
    # debug=True automatically reloads when code changes (DO NOT use in production)
    app.run(host="0.0.0.0", port=8000, debug=False)