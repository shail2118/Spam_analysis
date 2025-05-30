
import streamlit as st
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Sample data


df = pd.read_csv('spam.csv', encoding='latin-1')
df['email'] = df['email'].apply(preprocess_text)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size=0.2, random_state=42)
model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
model.fit(X_train, y_train)

# Streamlit UI
st.title("Spam Email Prediction using NLP")
user_input = st.text_area("Enter email text:")

if st.button("Predict"):
    processed_input = preprocess_text(user_input)
    prediction = model.predict([processed_input])[0]
    st.write("Prediction:", "Spam" if prediction == 1 else "Not Spam")
