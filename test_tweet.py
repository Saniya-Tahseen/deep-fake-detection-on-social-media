"""
Terminal CLI Tool for Real-Time Deepfake Tweet Detection
Run:
    python test_tweet.py
or
    python test_tweet.py "Paste your tweet here"
"""

import sys
import os
import pickle
import string
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF verbose logging

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from keras.models import load_model

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
table = str.maketrans('', '', string.punctuation)

def cleanText(doc):
    tokens = str(doc).split()
    tokens = [w.translate(table) for w in tokens]
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w.lower() not in stop_words]
    tokens = [w for w in tokens if len(w) > 1]
    tokens = [ps.stem(w) for w in tokens]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

# Load assets
base_dir = os.path.dirname(os.path.abspath(__file__))
tfidf_path = os.path.join(base_dir, "model", "tfidf.pckl")
scaler_path = os.path.join(base_dir, "model", "scaler.pckl")
model_path = os.path.join(base_dir, "model", "attention_bilstm.keras")

if not os.path.exists(tfidf_path) or not os.path.exists(model_path):
    print("Error: Model assets not found. Please ensure model/ contains trained weights.")
    sys.exit(1)

with open(tfidf_path, "rb") as f:
    tfidf = pickle.load(f)

with open(scaler_path, "rb") as f:
    sc = pickle.load(f)

model = load_model(model_path)

def predict_tweet(tweet_text):
    cleaned = cleanText(tweet_text.strip().lower())
    vec = tfidf.transform([cleaned]).toarray()
    scaled = sc.transform(vec)
    reshaped = np.reshape(scaled, (1, 30, 30))
    probs = model.predict(reshaped, verbose=0)[0]
    pred_class = np.argmax(probs)
    
    label = "Bot Fake (Machine-Generated)" if pred_class == 1 else "Normal (Human-Generated)"
    confidence = float(np.max(probs) * 100)
    
    print("\n" + "="*65)
    print("🔍 DEEPFAKE TWEET DETECTION ANALYSIS RESULT")
    print("="*65)
    print(f"📝 Raw Tweet       : {tweet_text}")
    print(f"🧹 Cleaned Root    : {cleaned}")
    print(f"🏆 Model Used      : Proposed CNN-BiLSTM + Attention (96.25% Acc)")
    print(f"🎯 Prediction      : {label}")
    print(f"📊 Confidence      : {confidence:.2f}%")
    print(f"📈 Raw Probabilities: Human: {probs[0]*100:.2f}% | Bot: {probs[1]*100:.2f}%")
    print("="*65 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        predict_tweet(input_text)
    else:
        print("\n" + "="*65)
        print("  FastText + CNN-BiLSTM Social Media Deepfake Detector (CLI)")
        print("="*65)
        print("Type a tweet below (or type 'exit' / 'q' to quit):\n")
        while True:
            try:
                user_input = input("Enter Tweet > ").strip()
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Exiting detector. Goodbye!")
                    break
                if not user_input:
                    continue
                predict_tweet(user_input)
            except (KeyboardInterrupt, EOFError):
                break
