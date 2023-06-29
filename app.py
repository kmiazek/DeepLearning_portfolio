import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('my_model.h5')  # replace with your saved model

st.title('Music Genre Prediction from Album Covers')

def preprocess_image(image):
    image = image.resize((150, 150)) # adjust size if necessary
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

uploaded_file = st.file_uploader("Choose an album cover image...", type=['png', 'jpg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    image = preprocess_image(image)
    
    # Make the prediction
    st.write("Predicting...")
    prediction = model.predict(image)

    # Use argmax to find the index of the most likely prediction
    prediction = np.argmax(prediction, axis=1)

    categories = ['disco', 'electro', 'folk', 'rap', 'rock'] # update if your categories are different
    st.write(f"Predicted genre: {categories[prediction[0]]}")
