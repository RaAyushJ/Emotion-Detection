import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the dataset
dataset = pd.read_csv('dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset['tweet'], dataset['label'], test_size=0.25, random_state=42)

# Preprocess the data
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer

stemmer = LancasterStemmer()

def preprocess_text(text):
  """Preprocesses the given text."""

  # Remove stop words
  text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])

  # Stem the words
  text = ' '.join([stemmer.stem(word) for word in text.split()])

  # Convert the text to lowercase
  text = text.lower()

  return text

# Preprocess the training and testing data
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the training and testing data into TF-IDF vectors
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Create a logistic regression classifier
clf = LogisticRegression()

# Train the classifier on the training data
clf.fit(X_train_vec, y_train)

# Evaluate the classifier on the testing data
y_pred = clf.predict(X_test_vec)
accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)

# To use the trained model to predict the emotion of a new sentence, you can use the following code:

def predict_emotion(sentence):
  """Predicts the emotion of the given sentence."""

  # Preprocess the sentence
  sentence = preprocess_text(sentence)

  # Transform the sentence into a TF-IDF vector
  sentence_vec = vectorizer.transform([sentence])

  # Predict the emotion of the sentence
  emotion = clf.predict(sentence_vec)[0]

  # Return the emotion
  return emotion

# Example usage:

# Take input from the user
sentence = input('Enter a sentence: ')

# Predict the emotion of the sentence
emotion = predict_emotion(sentence)

# Print the predicted emotion
print('Predicted emotion:', emotion)

