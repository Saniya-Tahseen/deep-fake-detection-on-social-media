import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
_stop_words = set(stopwords.words('english'))
_punctuation = set(string.punctuation)

class StylometricExtractor:
    """
    Extracts 16 linguistic, stylistic, and statistical features
    characterizing human vs. machine-generated (bot) social media tweets.
    """
    FEATURE_NAMES = [
        'char_count',
        'word_count',
        'avg_word_len',
        'uppercase_ratio',
        'digits_ratio',
        'punctuation_ratio',
        'exclamation_count',
        'question_count',
        'hashtag_count',
        'mention_count',
        'url_count',
        'type_token_ratio',
        'stopword_ratio',
        'sentence_burstiness',
        'char_repetition_ratio',
        'all_caps_words_count'
    ]

    @staticmethod
    def extract_vector(text: str) -> np.ndarray:
        raw = str(text or '')
        chars = len(raw)
        words = raw.split()
        num_words = len(words)
        
        # Avoid division by zero
        safe_chars = max(1, chars)
        safe_words = max(1, num_words)

        # 1. Length & Size
        char_count = float(chars)
        word_count = float(num_words)
        word_lens = [len(w) for w in words] if words else [0]
        avg_word_len = float(np.mean(word_lens)) if words else 0.0

        # 2. Character Distributions
        uppercase_count = sum(1 for c in raw if c.isupper())
        uppercase_ratio = float(uppercase_count / safe_chars)

        digits_count = sum(1 for c in raw if c.isdigit())
        digits_ratio = float(digits_count / safe_chars)

        punct_count = sum(1 for c in raw if c in _punctuation)
        punctuation_ratio = float(punct_count / safe_chars)

        # 3. Expressive punctuation
        exclamation_count = float(raw.count('!'))
        question_count = float(raw.count('?'))

        # 4. Social Media Specifics (Hashtags, Mentions, URLs)
        hashtag_count = float(len(re.findall(r'#\w+', raw)))
        mention_count = float(len(re.findall(r'@\w+', raw)))
        url_count = float(len(re.findall(r'https?://\S+|www\.\S+|t\.co/\S+', raw)))

        # 5. Lexical Diversity (Type-Token Ratio)
        clean_words = [w.strip(string.punctuation).lower() for w in words if w.strip(string.punctuation)]
        num_clean = max(1, len(clean_words))
        type_token_ratio = float(len(set(clean_words)) / num_clean)

        # 6. Stopwords ratio
        stopword_count = sum(1 for w in clean_words if w in _stop_words)
        stopword_ratio = float(stopword_count / num_clean)

        # 7. Burstiness (standard deviation of word lengths, indicating cadence variance)
        burstiness = float(np.std(word_lens)) if len(word_lens) > 1 else 0.0

        # 8. Character Bigram Diversity (Detects repetitive gibberish/patterned loops)
        if chars >= 2:
            bigrams = [raw[i:i+2] for i in range(chars - 1)]
            char_repetition_ratio = float(len(set(bigrams)) / len(bigrams))
        else:
            char_repetition_ratio = 1.0

        # 9. All-caps shouting words (e.g. BUY, FREE, WIN)
        all_caps_words_count = float(sum(1 for w in words if len(w) > 1 and w.isupper() and w.isalpha()))

        return np.array([
            char_count,
            word_count,
            avg_word_len,
            uppercase_ratio,
            digits_ratio,
            punctuation_ratio,
            exclamation_count,
            question_count,
            hashtag_count,
            mention_count,
            url_count,
            type_token_ratio,
            stopword_ratio,
            burstiness,
            char_repetition_ratio,
            all_caps_words_count
        ], dtype=np.float32)

    @classmethod
    def extract_dict(cls, text: str) -> dict:
        vec = cls.extract_vector(text)
        return {name: round(float(val), 4) for name, val in zip(cls.FEATURE_NAMES, vec)}
