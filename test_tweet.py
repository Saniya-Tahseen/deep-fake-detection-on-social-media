"""
Terminal CLI Tool for Real-Time Deepfake Tweet Detection
Powered by SOTA Stacking Ensemble (BERTweet + Stylometry + CatBoost)
Run:
    python test_tweet.py
or
    python test_tweet.py "Paste your tweet here"
"""

import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF verbose logging

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from DeepApp.stacking_model import predict_stacking_ensemble

def predict_tweet(tweet_text):
    res = predict_stacking_ensemble(tweet_text, base_dir=base_dir)
    if not res:
        print("Error: Empty tweet provided.")
        return

    label = "Bot Fake (Machine-Generated)" if res['prediction_class'] == 1 else "Normal (Human-Generated)"
    base = res['base_learners']
    style = base['stylometry']

    print("\n" + "="*68)
    print("  DEEPFAKE TWEET DETECTION ANALYSIS RESULT (SOTA ENSEMBLE)")
    print("="*68)
    print(f" Raw Tweet           : {tweet_text}")
    print(f" Model Used          : {res['algorithm']}")
    print(f" Benchmark Accuracy  : {res['meta_accuracy']}")
    print(f" Final Prediction    : {label}")
    print(f" Meta Confidence     : {res['confidence']}")
    print(f" Class Probabilities : Human: {res['probabilities']['Normal']} | Bot: {res['probabilities']['Bot Fake']}")
    print("-"*68)
    print(" MULTI-MODEL SIGNALS & STYLOMETRIC FOOTPRINT:")
    print(f"  * BERTweet NLP     : {base['bertweet']['label']} (Bot Likelihood: {base['bertweet']['bot_probability']})")
    print(f"  * CNN-BiLSTM Attn  : {base['cnn_bilstm']['label']} (Bot Likelihood: {base['cnn_bilstm']['bot_probability']})")
    print(f"  * Lexical Diversity: {style['lexical_diversity_ttr']} TTR")
    print(f"  * Burstiness Cadence: {style['burstiness']}")
    print(f"  * Uppercase / Punct: {style['uppercase_ratio']} / {style['punctuation_ratio']}")
    print(f"  * Social Footprint : {style['hashtags']} Hashtags | {style['urls']} URLs")
    print("="*68 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        predict_tweet(input_text)
    else:
        print("\n" + "="*68)
        print("  BERTweet + Stylometry + CatBoost Social Media Deepfake Detector")
        print("="*68)
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
