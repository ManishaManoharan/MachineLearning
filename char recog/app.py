import streamlit as st
import numpy as np
from PIL import Image
import pickle

model = pickle.load(open("mlp_model.pkl","rb"))

st.title("Handwritten Character Recognition")

uploaded = st.file_uploader("Upload an image")

if uploaded:

    image = Image.open(uploaded).convert("L")
    image = image.resize((28,28))

    img = np.array(image)
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1,-1)

    prediction = model.predict(img)

    st.image(image, width=150)
    st.write("Predicted Character:", prediction[0])