# FastText-Powered Deep Learning for Social Media Deep Fake Detection

This project implements an AI-driven web application to detect automated bots and machine-generated (deepfake) tweets on social media platforms, replicating the complete architecture and results documented in the project documentation.

---

## 🚀 Key Features & Project Modules

1. **System Overview Architecture**
   - End-to-end detection pipeline: Twitter Feeds $\rightarrow$ Preprocessing $\rightarrow$ FastText/TF-IDF Embeddings $\rightarrow$ 80-20 Train/Test Splitting $\rightarrow$ Deep CNN-BiLSTM + Attention $\rightarrow$ Classification (`Bot Fake` vs `Normal`).

2. **Dataset Preprocessing (`/LoadDataset`)**
   - Ingests the benchmark **TweepFake** dataset (25,572 labeled tweets).
   - Cleans tweet text using NLTK stopwords, stemming (PorterStemmer), and lemmatization (WordNetLemmatizer).
   - Displays the top 100 rows in an interactive table (`Username`, `Tweet`, `Account Type`).

3. **FastText / TF-IDF Vectorization (`/FastText`)**
   - Transforms cleaned tokens into a 900-dimensional semantic feature matrix.
   - Normalizes feature scales via `StandardScaler`.

4. **Multi-Algorithm Evaluation & Visual Comparison (`/TrainML`)**
   - Evaluates baseline models against deep hybrid networks, stacking ensemble, and the enhanced proposed architecture:
     - **Naive Bayes**: 62.00% Accuracy
     - **Logistic Regression**: 66.50% Accuracy
     - **Decision Tree**: 56.50% Accuracy
     - **Random Forest**: 63.00% Accuracy
     - **Gradient Boosting**: 60.00% Accuracy
     - **Proposed CNN**: 87.06% Accuracy
     - **Extension Hybrid CNN**: 92.64% Accuracy
     - **Stacking Ensemble (BERTweet + Stylometry + CatBoost) (🥈 Runner-Up)**: 98.85% Accuracy (98.90% Precision, 98.82% Recall, 98.86% F1-Score)
     - **Enhanced Proposed CNN-BiLSTM + Attention (🏆 #1 SOTA Champion)**: **99.42% Accuracy** (99.48% Precision, 99.36% Recall, 99.42% F1-Score)
   - Generates an on-the-fly Matplotlib grouped bar graph comparing Accuracy, Precision, Recall, and F1-Score across all 9 models, highlighting the Proposed CNN-BiLSTM + Attention model as the top performer.

5. **Enhanced Architecture Details (Proposed CNN-BiLSTM + Attention)**
   - **Multi-Scale Parallel Convolutions**: Parallel Conv1D branches (kernel sizes 3, 5, 7) extracting diverse n-gram linguistic spans, stabilized by Batch Normalization and Spatial Dropout.
   - **Stacked Bidirectional LSTMs**: 2-layer stacked BiLSTMs capturing bidirectional sequential and temporal context across Twitter subword tokens.
   - **Additive Self-Attention & Hybrid Pooling**: Bahdanau-style self-attention scoring that assigns dynamic weights to key salient tokens, combined with Global Max Pooling and Global Average Pooling.
   - **Regularized Classification Head**: Deep Dense layers with Batch Normalization and Dropout for robust generalization.

6. **Real-Time Tweet Detection (`/DetectFake.html` & `test_tweet.py`)**
   - Allows users to enter any tweet in real-time.
   - Evaluates input using the champion **Proposed CNN-BiLSTM + Attention** model with multi-scale feature representations and stylometric footprinting.
   - Instantly predicts and displays: `Given Tweet Detected as : Bot Fake` or `Given Tweet Detected as : Normal` along with confidence, accuracy badge, and class probability breakdowns.

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

### 4. Standalone CLI Test
Run real-time inference directly in terminal:
```bash
python test_tweet.py "Enter custom tweet here"
```

---

## 📁 Project Structure

```
deep_fake_detection/
├── Dataset/
│   └── Tweepfake.csv               # Complete TweepFake dataset (25,572 rows)
├── Deep/                           # Django project configuration
│   ├── settings.py                 # App & static files configuration
│   ├── urls.py                     # Main project routing
│   └── wsgi.py
├── DeepApp/                        # DeepFake Detection Django Application
│   ├── static/
│   │   ├── css/style.css           # Cohesive UI styling
│   │   └── images/
│   │       └── diagram.png         # System architecture workflow diagram
│   ├── templates/
│   │   ├── index.html              # Homepage
│   │   ├── UserLogin.html          # Login Screen
│   │   ├── UserScreen.html         # Admin Control Panel
│   │   ├── ViewOutput.html         # Dataset & Vector views
│   │   ├── ViewGraph.html          # Performance Table & Chart
│   │   └── DetectFake.html         # Real-Time Tweet Testing Screen
│   ├── urls.py                     # App URL routing
│   ├── views.py                    # ML inference & view controllers
│   ├── stacking_model.py           # Multi-model inference engine
│   └── stylometry.py               # 16-dimensional stylometric feature extractor
├── model/                          # Pre-trained models and saved binary assets
│   ├── X.npy                       # Preprocessed feature matrix
│   ├── Y.npy                       # Target label array
│   ├── tfidf.pckl                  # Fitted TfidfVectorizer
│   ├── scaler.pckl                 # Fitted StandardScaler
│   ├── cnn_model.keras             # Trained Keras CNN model
│   ├── cnn_weights.hdf5            # Saved HDF5 weights
│   ├── attention_bilstm.keras      # Enhanced Proposed CNN-BiLSTM + Attention model
│   └── metrics.pckl                # Benchmark evaluation metrics (Proposed: 99.42%)
├── build_models.py                 # Standalone model training & asset generator
├── train_improved_attention.py     # Enhanced CNN-BiLSTM + Attention training pipeline
├── test_tweet.py                   # Terminal CLI prediction tool
├── manage.py                       # Django CLI runner
└── db.sqlite3                      # Local SQLite database
```
