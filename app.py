# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import matplotlib.pyplot as plt

# Load the trained model

# Load model
model = tf.keras.models.load_model('pneumonia.h5')

# Set up Streamlit app
st.title("X-Ray Image Classification App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an X-Ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the uploaded image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(img_array)

    # Display the result
    st.write("\n\nPrediction:")
    if prediction[0][0] > 0.5:
        result = "PNEUMONIA"
    else:
        result = "NORMAL"

    st.write(result)

    # Display the probability
    confidence = f"Confidence: {prediction[0][0]:.2%}"
    st.write(confidence)

    # Display accuracy based on the result
    if result == "PNEUMONIA":
        st.warning("Warning: Pneumonia detected. Seek medical attention.")
    else:
        st.success("No Pneumonia detected. Stay healthy!")

# Add a link to the original Colab notebook
st.markdown("[Link to Original Colab Notebook](https://colab.research.google.com/drive/14hKuzIlMPuNuQcMt5eC9juUoOHqc6CF7)")


    