# import streamlit as st
# from PIL import Image

# uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
# for uploaded_file in uploaded_files:
#     st.write("filename:", uploaded_file.name)
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
import requests
import os

model_path = r"D:\Program Files\Code repositories\RAG\RAG\model_pokemon.h5"
directory = r"D:\Program Files\Code repositories\PokemonData"
labels = os.listdir(directory)
nb = len(labels)

model = load_model(model_path)

def predict_pokemon(image):
    img = cv2.resize(image, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    label_idx = np.argmax(pred, axis=1)[0]
    label = labels[label_idx]
    return label

st.title("Image Classifier")
st.write("Upload an image and the model will predict its type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    image = np.array(image)
    
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    with st.spinner('Classifying...'):
        label = predict_pokemon(image)
        st.success(f'This is a {label}!')
        
        pokemon = label.lower()
        url = f'https://pokeapi.co/api/v2/pokemon/{pokemon}'
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            st.write("Name:", data['name'])
            st.write("Base Experience:", data['base_experience'])
            st.write("Height:", data['height'], 'm')
            st.write("Weight:", data['weight'], 'kg')
        else:
            st.write("Could not fetch details for this Pokemon.")
