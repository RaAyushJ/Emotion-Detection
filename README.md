# Emotion Detection from Tweets

This project implements a machine learning pipeline for detecting emotions in tweets using text classification techniques. The system uses Natural Language Processing (NLP) to preprocess text data and Logistic Regression as the classifier.

## Project Structure

├── train_emotion_model.py # Python script for training, testing, and predicting emotions
├── dataset.csv # Labeled dataset for training
├── validation.csv # Dataset for validation during training
├── test.csv # Dataset for evaluating final model performance
├── emotion_model.pkl # Trained Logistic Regression model saved using joblib
├── tfidf_vectorizer.pkl # TF-IDF vectorizer saved for consistent text processing
├── Output_EmotionDetection.docx # Report/Output documentation
└── README.md # This file


## Requirements

Install the required Python packages using pip:

pip install pandas numpy scikit-learn nltk joblib


Also download NLTK stopwords if not already available:

'''python

    import nltk
    nltk.download('stopwords')

How to Run

    Make sure the files dataset.csv, validation.csv, and test.csv are present in the same directory as train_emotion_model.py.

    Run the training script:

python train_emotion_model.py

This will:

    Load and preprocess the data

    Train a Logistic Regression model

    Evaluate the model on training, validation, and test sets

    Save the trained model and vectorizer

    Allow real-time emotion prediction from user input

Example Usage

Enter a sentence (or type 'exit' to quit): I am feeling so happy today!
Predicted Emotion: joy

Model Details

    Preprocessing:

        Lowercasing

        Stopword removal using NLTK

        Stemming using LancasterStemmer

    Feature Extraction:

        TF-IDF Vectorization (TfidfVectorizer)

    Model:

        Logistic Regression (sklearn.linear_model.LogisticRegression)

Using the Trained Model

To load and use the model in another Python script:

import joblib

# Load model and vectorizer
clf = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocess your text as done during training
# Example preprocessing function must be defined identically

processed = preprocess_text("I am excited about the future.")
vector = vectorizer.transform([processed])
prediction = clf.predict(vector)[0]

print("Predicted Emotion:", prediction)
