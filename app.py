# Import libraries
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


# Title of the app
st.title("AI Image Classifier")

st.write("Upload an image and the AI will try to guess what it is.")


# Load pretrained model
model = MobileNetV2(weights="imagenet")


# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])


# If image is uploaded
if uploaded_file:

    image = Image.open(uploaded_file)

    # Show the uploaded image
    st.image(image, caption="Your Image", use_column_width=True)

    # Resize image for the model
    image = image.resize((224,224))

    # Convert image to array
    img = np.array(image)

    # Prepare image for prediction
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make prediction
    prediction = model.predict(img)

    # Get label
    label = decode_predictions(prediction)[0][0]

    # Show result
    st.write("Prediction:", label[1])
    st.write("Confidence:", round(label[2] * 100, 2), "%")