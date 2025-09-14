# ============================================
# ✈️ Airline Sentiment Analysis
# Author: Dharani Manne
# Description: Analyze airline tweets and predict sentiment (Positive, Negative, Neutral)
# ============================================

# ==============================
# Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
import re
import random

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ==============================
# Load & Clean Data
# ==============================
df = pd.read_csv("Tweets.csv")

# Drop irrelevant columns
df.drop(columns=['tweet_id', 'airline_sentiment_gold', 'negativereason_gold', 'tweet_coord'], inplace=True)

# Fill missing values
df['negativereason'] = df['negativereason'].fillna('Unknown')
df['negativereason_confidence'] = df['negativereason_confidence'].fillna(df['negativereason_confidence'].mean())
df['tweet_location'] = df['tweet_location'].fillna(df['tweet_location'].mode()[0])
df['user_timezone'] = df['user_timezone'].fillna(df['user_timezone'].mode()[0])

# Remove empty tweets
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].str.len() > 0]

# Subset for modeling
df_model = df[['text','airline_sentiment']].copy()
df_model['text_length'] = df_model['text'].apply(len)

# ==============================
# Exploratory Data Analysis
# ==============================
# Tweet length distribution
plt.figure(figsize=(10,5))
plt.hist(df_model['text_length'], bins=50, edgecolor='black')
plt.title("Tweet Length Distribution")
plt.xlabel("Tweet Length (characters)")
plt.ylabel("Number of Tweets")
plt.grid(True)
plt.show()

# Tweet length by sentiment
plt.figure(figsize=(10,6))
sns.boxplot(data=df_model, x='airline_sentiment', y='text_length', palette='pastel')
plt.title("Tweet Length per Sentiment Category")
plt.show()

# Sentiment distribution
sentiment_counts = df_model['airline_sentiment'].value_counts()
plt.figure(figsize=(6,4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='Set2')
plt.title("Sentiment Distribution")
plt.show()

# ==============================
# Text Preprocessing
# ==============================
stop_words = set(stopwords.words('english') + list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|&\w+;|@", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

df_model['clean_text'] = df_model['text'].apply(clean_text)

# ==============================
# Feature Extraction
# ==============================
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df_model['clean_text'])

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df_model['airline_sentiment'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# ==============================
# Machine Learning Models
# ==============================
# Logistic Regression
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_))

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

print("Naive Bayes Classification Report")
print(classification_report(y_test, y_pred_nb, target_names=label_encoder.classes_))

# ==============================
# Deep Learning Models
# ==============================
# Tokenize text for LSTM/RNN
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df_model['clean_text'])
sequences = tokenizer.texts_to_sequences(df_model['clean_text'])
padded_sequences = pad_sequences(sequences, maxlen=50)

# One-hot encode labels
y_categorical = to_categorical(y_encoded)

# Train-test split
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(padded_sequences, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded)

# LSTM Model
lstm_model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=50),
    LSTM(64),
    Dense(3, activation='softmax')
])
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_train_dl, y_train_dl, epochs=3, batch_size=64, validation_split=0.2)

y_pred_lstm = np.argmax(lstm_model.predict(X_test_dl), axis=1)
y_true_lstm = np.argmax(y_test_dl, axis=1)

print("LSTM Classification Report")
print(classification_report(y_true_lstm, y_pred_lstm, target_names=label_encoder.classes_))

# RNN Model
rnn_model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=50),
    SimpleRNN(64),
    Dense(3, activation='softmax')
])
rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
rnn_model.fit(X_train_dl, y_train_dl, epochs=3, batch_size=64, validation_split=0.2)

y_pred_rnn = np.argmax(rnn_model.predict(X_test_dl), axis=1)
y_true_rnn = np.argmax(y_test_dl, axis=1)

print("RNN Classification Report")
print(classification_report(y_true_rnn, y_pred_rnn, target_names=label_encoder.classes_))

# ==============================
# Evaluation Function
# ==============================
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Metrics:")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall   :", recall_score(y_true, y_pred, average='weighted'))
    print("F1-Score :", f1_score(y_true, y_pred, average='weighted'))

evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("Naive Bayes", y_test, y_pred_nb)
evaluate_model("LSTM", y_true_lstm, y_pred_lstm)
evaluate_model("RNN", y_true_rnn, y_pred_rnn)

# ==============================
# Confusion Matrices
# ==============================
def plot_confusion(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

plot_confusion("Logistic Regression", y_test, y_pred_lr)
plot_confusion("Naive Bayes", y_test, y_pred_nb)
plot_confusion("LSTM", y_true_lstm, y_pred_lstm)
plot_confusion("RNN", y_true_rnn, y_pred_rnn)
