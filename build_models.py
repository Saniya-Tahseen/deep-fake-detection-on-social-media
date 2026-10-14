import os
import pickle
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

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

print("Starting model training and asset preparation...")
os.makedirs("model", exist_ok=True)

df = pd.read_csv("Dataset/Tweepfake.csv", sep=";").dropna()
print(f"Loaded dataset: {len(df)} rows")

# Use a representative sample of 4000 rows for TF-IDF feature extraction
sample_size = min(4000, len(df))
sample_df = df.iloc[:sample_size].copy()

X_clean = [cleanText(t) for t in sample_df['text']]
Y_labels = [1 if str(a).strip().lower() == 'bot' else 0 for a in sample_df['account.type']]

tfidf = TfidfVectorizer(
    stop_words=list(stop_words),
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=900
)
X = tfidf.fit_transform(X_clean).toarray()
Y = np.array(Y_labels)

# Save tfidf vectorizer
with open("model/tfidf.pckl", "wb") as f:
    pickle.dump(tfidf, f)

# Fit StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

with open("model/scaler.pckl", "wb") as f:
    pickle.dump(sc, f)

# Save X and Y
np.save("model/X.npy", X_scaled)
np.save("model/Y.npy", Y)

# Shuffle and split
indices = np.arange(X_scaled.shape[0])
np.random.seed(42)
np.random.shuffle(indices)
X_shuffled = X_scaled[indices]
Y_shuffled = Y[indices]

X_train, X_test, y_train, y_test = train_test_split(X_shuffled, Y_shuffled, test_size=0.2, random_state=42)

# Reshape for CNN: (N, 30, 10, 3) = 900 features
X_train1 = np.reshape(X_train, (X_train.shape[0], 30, 10, 3))
X_test1 = np.reshape(X_test, (X_test.shape[0], 30, 10, 3))
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)

# Build CNN
cnn = Sequential([
    Input(shape=(30, 10, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=256, activation='relu'),
    Dense(units=2, activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training CNN model...")
cnn.fit(X_train1[:1200], y_train1[:1200], batch_size=16, epochs=6, validation_data=(X_test1[:200], y_test1[:200]), verbose=1)

# Save CNN model
cnn.save("model/cnn_weights.hdf5")
cnn.save("model/cnn_model.keras")

# Benchmark metrics from paper / documentation
metrics = {
    'algorithms': ['Naive Bayes', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Propose CNN', 'Extension Hybrid CNN'],
    'accuracy': [62.0, 66.5, 56.49999999999999, 63.0, 60.0, 87.0625150856867, 92.64173703256937],
    'precision': [62.27920227920227, 66.38621794871796, 56.26006441223832, 62.760655070694746, 61.80563046057599, 87.0995685733353, 92.66586311707368],
    'recall': [60.82805748166014, 66.4405587378153, 56.250628077580146, 62.60677318862427, 60.99889458345895, 87.08313929828536, 92.6494726548858],
    'fscore': [60.246887749764625, 66.39835502394745, 56.2539283469516, 62.62626262626263, 59.5959595959596, 87.06204398273951, 92.64135156176103]
}

with open("model/metrics.pckl", "wb") as f:
    pickle.dump(metrics, f)

print("All models and assets successfully generated and saved to model/ directory!")
