import re
import pickle
import streamlit as st
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk


nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

lb = pickle.load(open('lb.pkl', 'rb'))
lg = pickle.load(open('lg.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))


def preprocess_text(text):
    stemmer = PorterStemmer()
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = text.lower().split()
    words = [stemmer.stem(word) for word in text if word not in stopwords]
    return ' '.join(words)


def predict_emotion(text):
    processed_text = preprocess_text(text)
    vectorized_text = cv.transform([processed_text])
    prediction = lg.predict(vectorized_text)[0]
    emotion = lb.inverse_transform([prediction])[0]
    return emotion


def get_emotion_color(emotion):
    colors = {
        "joy": "#fdeb4c",
        "sadness": "#0c9be1",
        "anger": "#e72222",
        "fear": "#a38cb3",
        "love": "#ff009e",
        "surprise": "#00cee8"
    }
    return colors.get(emotion, "black")


def main():
    st.set_page_config(page_title="Emotion Classifier")
    st.title("Text Emotion Classification")

    user_input = st.text_area("Enter your text here:")

    if st.button("Predict Emotion"):
        if user_input:
            result = predict_emotion(user_input)
            color = get_emotion_color(result)
            st.markdown(f'<p style="font-size:24px;color:{color}"><strong>{result}</strong></p>', unsafe_allow_html=True)
        else:
            st.write("Please enter some text to analyze.")


if __name__ == '__main__':
    main()