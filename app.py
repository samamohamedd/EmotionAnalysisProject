import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

model = load_model('emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Emotion Analysis from Facial Expressions")
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img_resized = image.resize((48, 48))
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)
    
    prediction = model.predict(img_array)
    emotion = emotion_labels[np.argmax(prediction)]
    
    st.image(uploaded_file, caption=f"Predicted Emotion: {emotion}")
