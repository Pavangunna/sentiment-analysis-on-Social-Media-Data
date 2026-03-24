import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("sentiment_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

label_map = {
    0: "Negative 😞",
    1: "Positive 😊",
    2: "Neutral 😐"
}

st.title("💬 Sentiment Analysis (Deep Learning)")

text = st.text_area("Enter text:")

if st.button("Analyze"):
    if text.strip():
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=50)

        pred = model.predict(padded)
        result = pred.argmax()

        st.success(label_map[result])
    else:
        st.warning("Enter text")