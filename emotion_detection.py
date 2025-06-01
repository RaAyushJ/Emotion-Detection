import pandas as pd
import numpy as np
import nltk
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer

# Download required NLTK data
nltk.download('stopwords')

# ------------------------------
# Preprocessing Function
# ------------------------------

stemmer = LancasterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocess text by removing stopwords, stemming, and converting to lowercase."""
    if pd.isnull(text):
        return ""
    words = text.split()
    words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# ------------------------------
# Load Data
# ------------------------------

train_df = pd.read_csv('/content/training.csv')
val_df = pd.read_csv('/content/validation.csv')
test_df = pd.read_csv('/content/test.csv')

# Preprocess text columns
train_df['text'] = train_df['text'].apply(preprocess_text)
val_df['text'] = val_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# Features and labels
X_train, y_train = train_df['text'], train_df['label'] # Also change here for consistency
X_val, y_val = val_df['text'], val_df['label']       # And here
X_test, y_test = test_df['text'], test_df['label']     # And here

# ------------------------------
# Vectorization
# ------------------------------

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# ------------------------------
# Model Training
# ------------------------------

clf = LogisticRegression(max_iter=200)
clf.fit(X_train_vec, y_train)

# ------------------------------
# Evaluation
# ------------------------------

print("Training Accuracy:", accuracy_score(y_train, clf.predict(X_train_vec)))
print("Validation Accuracy:", accuracy_score(y_val, clf.predict(X_val_vec)))
print("Test Accuracy:", accuracy_score(y_test, clf.predict(X_test_vec)))

# ------------------------------
# Save Model and Vectorizer
# ------------------------------

joblib.dump(clf, 'emotion_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# ------------------------------
# Predict Custom Input
# ------------------------------

def predict_emotion(sentence):
    sentence = preprocess_text(sentence)
    sentence_vec = vectorizer.transform([sentence])
    return clf.predict(sentence_vec)[0]

# Predict from user input
while True:
    sentence = input("\nEnter a sentence (or type 'exit' to quit): ")
    if sentence.lower() == 'exit':
        break
    emotion = predict_emotion(sentence)
    print("Predicted Emotion:", emotion)
