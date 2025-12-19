import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np

classes = ['Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel', 'Fish', 'Jelly Fish', 'Lobster', 'Nudibranchs', 'Octopus', 'Otter', 'Penguin', 'Puffers', 'Sea Rays', 'Sea Urchins', 'Seahorse', 'Seal', 'Sharks', 'Shrimp', 'Squid', 'Starfish', 'Turtle_Tortoise', 'Whale']


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras")
    return model

model = load_model()

st.title("Image Classification App")
st.write("Upload an image and the model will predict its class.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

IMG_SIZE = 224

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.convert('RGB')  # Ensure RGB format (3 channels)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension: (1, 224, 224, 3)
    return img_array

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img = preprocess_image(uploaded_file)
    prediction = model.predict(img)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("Prediction:")
    st.write(f"Class: **{classes[class_index]}**")
    st.write(f"Confidence: **{confidence*100:.2f}**")