import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore

# Load the trained model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit App
def main():
    st.title("Spam Detection App")
    st.write("Enter a message to check if it's spam or not.")
    
    # User Input
    user_input = st.text_area("Enter your message here:")
    
    if st.button("Predict"):
        if user_input.strip():
            # Transform input using the TF-IDF vectorizer
            input_vectorized = vectorizer.transform([user_input])
            
            # Predict
            prediction = model.predict(input_vectorized)
            result = "Spam" if prediction[0] == 1 else "Not Spam"
            
            st.write(f"Prediction: **{result}**")
        else:
            st.write("Please enter a message to classify.")
    
if __name__ == "__main__":
    main()
