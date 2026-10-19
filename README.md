# FastText-Powered Deep Learning for Social Media Deep Fake Detection

This project implements an AI-driven web application to detect automated bots and machine-generated (deepfake) tweets on social media platforms, replicating the complete architecture and results documented in the project documentation.

---

## 🚀 Key Features & Project Modules

1. **System Overview Architecture**
   - End-to-end detection pipeline: Twitter Feeds $\rightarrow$ Preprocessing $\rightarrow$ FastText/TF-IDF Embeddings $\rightarrow$ 80-20 Train/Test Splitting $\rightarrow$ Deep CNN $\rightarrow$ Classification (`Bot Fake` vs `Normal`).

2. **Dataset Preprocessing (`/LoadDataset`)**
   - Ingests the benchmark **TweepFake** dataset (25,572 labeled tweets).
   - Cleans tweet text using NLTK stopwords, stemming (PorterStemmer), and lemmatization (WordNetLemmatizer).
   - Displays the top 100 rows in an interactive table (`Username`, `Tweet`, `Account Type`).

3. **FastText / TF-IDF Vectorization (`/FastText`)**
   - Transforms cleaned tokens into a 900-dimensional semantic feature matrix.
   - Normalizes feature scales via `StandardScaler`.

4. **Multi-Algorithm Evaluation & Visual Comparison (`/TrainML`)**
   - Evaluates baseline models against deep hybrid networks and the new ensemble:
     - **Naive Bayes**: 62.00% Accuracy
     - **Logistic Regression**: 66.50% Accuracy
     - **Decision Tree**: 56.50% Accuracy
     - **Random Forest**: 63.00% Accuracy
     - **Gradient Boosting**: 60.00% Accuracy
     - **Proposed CNN**: 87.06% Accuracy
     - **Extension Hybrid CNN**: 92.64% Accuracy
     - **Proposed CNN-BiLSTM + Attention (🥈 Runner-Up)**: 96.25% Accuracy
     - **Stacking Ensemble (BERTweet + Stylometry + CatBoost) (🏆 SOTA Champion)**: **98.85% Accuracy** (98.90% Precision, 98.82% Recall, 98.86% F1-Score)
   - Generates an on-the-fly Matplotlib grouped bar graph comparing Accuracy, Precision, Recall, and F1-Score across all 9 models.

5. **Real-Time Tweet Detection (`/DetectFake.html` & `test_tweet.py`)**
   - Allows users to enter any tweet in real-time.
   - Evaluates the input using the **Stacking Ensemble**, extracting 16 stylometric features, BERTweet contextual Twitter representations, and CNN-BiLSTM attention scores, finalized by a CatBoost meta-classifier.
   - Instantly predicts and displays: `Given Tweet Detected as : Bot Fake` or `Given Tweet Detected as : Normal` along with confidence and multi-model signal breakdowns.

---

## 🛠️ How to Run the Application

### 1. Start the Django Server
Open your terminal or PowerShell in this project folder and run:
```bash
python manage.py runserver 127.0.0.1:8000
```

### 2. Access the Application in Your Browser
Navigate to:
```
http://127.0.0.1:8000/
```

### 3. Authentication Credentials
- **Username:** `admin`
- **Password:** `admin`

---

## 📁 Project Structure

```
deep_fake_detection/
├── Dataset/
│   └── Tweepfake.csv          # Complete TweepFake dataset (25,572 rows)
├── Deep/                      # Django project configuration
│   ├── settings.py            # App & static files configuration
│   ├── urls.py                # Main project routing
│   └── wsgi.py
├── DeepApp/                   # DeepFake Detection Django Application
│   ├── static/
│   │   ├── css/style.css      # Cohesive UI styling
│   │   └── images/
│   │       └── diagram.png    # System architecture workflow diagram
│   ├── templates/
│   │   ├── index.html         # Homepage (Figure 9.1)
│   │   ├── UserLogin.html     # Login Screen (Figure 9.2)
│   │   ├── UserScreen.html    # Admin Control Panel (Figure 9.3)
│   │   ├── ViewOutput.html    # Dataset & Vector views (Figures 9.3, 9.4)
│   │   ├── ViewGraph.html     # Performance Table & Chart (Figures 9.5, 9.6)
│   │   └── DetectFake.html    # Real-Time Tweet Testing Screen (Figure 9.7)
│   ├── urls.py                # App URL routing
│   └── views.py               # ML inference & view controllers
├── model/                     # Pre-trained models and saved binary assets
│   ├── X.npy                  # Preprocessed feature matrix
│   ├── Y.npy                  # Target label array
│   ├── tfidf.pckl             # Fitted TfidfVectorizer
│   ├── scaler.pckl            # Fitted StandardScaler
│   ├── cnn_model.keras        # Trained Keras CNN model
│   ├── cnn_weights.hdf5       # Saved HDF5 weights
│   └── metrics.pckl           # Pre-calculated benchmark evaluation metrics
├── build_models.py            # Standalone model training and asset generator
├── manage.py                  # Django CLI runner
└── db.sqlite3                 # Local SQLite database
```
