import pickle
import streamlit as st

# Load the model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Use them for predictions
text = st.text_input("Enter message:")
if text:
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    st.write("SPAM" if prediction == 1 else "HAM")