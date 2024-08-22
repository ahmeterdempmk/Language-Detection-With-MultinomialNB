import streamlit as st
from Language_Detection_With_MultinomialNB import model, cv

st.title("Language Detection App")
st.write("This app detects the language of the text you input.")

user_input = st.text_area("Enter your text here:")

if st.button("Detect Language"):
    if user_input:
        data = cv.transform([user_input]).toarray()
        output = model.predict(data)
        st.write(f"Detected language: {output[0]}")
    else:
        st.write("Please enter some text to detect the language.")

st.write("This application uses the MultinomialNB model to detect the language of the text.")