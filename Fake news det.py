from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import os # Import os for better file handling checks

# --- 1. INITIALIZATION ---
app = Flask(__name__)

# Check for required files (Added for robustness)
if not os.path.exists('model.pkl') or not os.path.exists('news.csv'):
    print("ERROR: Required files 'model.pkl' or 'news.csv' not found. Ensure they are in the same directory.")
    # Consider raising an error or exiting gracefully

# Load dataframe and split
try:
    dataframe = pd.read_csv('news.csv')
    x = dataframe['text']
    y = dataframe['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
except Exception as e:
    print(f"ERROR loading/splitting data: {e}")
    # Handle data loading error

# Load model
try:
    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
except Exception as e:
    print(f"ERROR loading model.pkl: {e}")
    # Handle model loading error

# --- 2. CORRECT TFIDF VECTORIZER HANDLING ---
# Initialize the vectorizer globally
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)

# **FIT THE VECTORIZER ONCE** on the training data when the app starts.
# This ensures a consistent vocabulary for all future predictions.
try:
    tfvect.fit(x_train) # We only need .fit() to learn the vocabulary
except Exception as e:
    print(f"ERROR fitting TfidfVectorizer: {e}")
    # Handle fitting error

# You can optionally keep the transform for training/testing but it's not needed for the web app logic.
# tfid_x_train = tfvect.transform(x_train)
# tfid_x_test = tfvect.transform(x_test)

# --- 3. PREDICTION FUNCTION (CLEANED) ---
def fake_news_det(news):
    """
    Predicts if a news article is fake or real using the globally fitted vectorizer
    and the globally loaded model.
    """
    # The vectorizer is already fitted, so we only use .transform() on the input
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    
    # Return a more user-friendly label
    if prediction[0] == 'FAKE':
        return "The News is **FAKE** ❌"
    else:
        return "The News is **REAL** ✅"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            message = request.form['message']
            pred = fake_news_det(message)
            print(f"Prediction made: {pred}")
            return render_template('index.html', prediction=pred)
        except Exception as e:
            # Catch errors during prediction (e.g., if model/vectorizer failed to load earlier)
            print(f"Prediction failed due to exception: {e}")
            return render_template('index.html', prediction="Prediction failed: An internal error occurred.")
    else:
        # This branch is rarely hit for a POST route but included for completeness
        return render_template('index.html', prediction="Something went wrong (Not POST request)")

if __name__ == '__main__':
    app.run(debug=True)
