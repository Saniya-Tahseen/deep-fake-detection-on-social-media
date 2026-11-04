"""
Terminal CLI Tool for Real-Time Deepfake Tweet Detection
Powered by Enhanced Proposed CNN-BiLSTM + Attention Deep Model (#1 SOTA Champion)
Run:
    python test_tweet.py
or
    python test_tweet.py "Paste your tweet here"
"""

import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('KERAS_HOME', os.path.join(base_dir, '.keras'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(base_dir, '.matplotlib'))
os.environ.setdefault('HF_HOME', os.path.join(base_dir, '.cache', 'huggingface'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

    print("\n" + "="*70)
    print("  DEEPFAKE TWEET DETECTION ANALYSIS RESULT (CHAMPION MODEL)")
    print("="*70)
    print(f" Raw Tweet           : {tweet_text}")
    print(f" Model Used          : {res['algorithm']}")
    print(f" Benchmark Accuracy  : {res['meta_accuracy']}")
    print(f" Final Prediction    : {label}")
    print(f" Confidence          : {res['confidence']}")
    print(f" Class Probabilities : Human: {res['probabilities']['Normal']} | Bot: {res['probabilities']['Bot Fake']}")
    print("-"*70)
    print(" DEEP MODEL SIGNALS & STYLOMETRIC ATTENTION FOOTPRINT:")
    print(f"  * CNN-BiLSTM Attn  : {base['cnn_bilstm']['label']} (Bot Likelihood: {base['cnn_bilstm']['bot_probability']})")
    print(f"  * BERTweet NLP     : {base['bertweet']['label']} (Bot Likelihood: {base['bertweet']['bot_probability']})")
    print(f"  * Lexical Diversity: {style['lexical_diversity_ttr']} TTR")
    print(f"  * Burstiness Cadence: {style['burstiness']}")
    print(f"  * Uppercase / Punct: {style['uppercase_ratio']} / {style['punctuation_ratio']}")
    print(f"  * Social Footprint : {style['hashtags']} Hashtags | {style['urls']} URLs")
    print("="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        raw_tweet = " ".join(sys.argv[1:])
        predict_tweet(raw_tweet)
    else:
        # Benchmark sample tweet
        sample = "ALERT: Claim your 5000 FREE Crypto Airdrop Tokens immediately by visiting http://bit.ly/crypto-scam #giveaway #free"
        print("No custom tweet provided. Running on benchmark sample tweet...\n")
        predict_tweet(sample)
