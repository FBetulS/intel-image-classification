import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Set page title and favicon
st.set_page_config(
    page_title="Intel Image Classifier",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add title and description
st.title("Intel Image Classification")
st.markdown("""
This app classifies images into 6 categories from the Intel Image Classification dataset:
- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street
""")

# Function to load the model
@st.cache_resource
def load_classification_model():
    model = load_model("intel_image_classifier.h5")
    return model

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the class
def predict(model, img_array):
    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    return class_names[predicted_class], confidence, predictions[0]

# Main function
def main():
    # Load model
    try:
        model = load_classification_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_loaded = False
    
    # Image upload widget
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    # Camera input option
    camera_input = st.camera_input("Or take a photo")
    
    input_image = uploaded_file if uploaded_file is not None else camera_input
    
    if input_image is not None and model_loaded:
        # Display the image
        image = Image.open(input_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Add a button to perform prediction
        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                # Preprocess the image
                processed_image = preprocess_image(image)
                
                # Make prediction
                class_name, confidence, raw_preds = predict(model, processed_image)
                
                # Display results
                st.success(f"Prediction: {class_name.capitalize()}")
                st.info(f"Confidence: {confidence:.2f}%")
                
                # Display prediction probabilities as a bar chart
                st.subheader("Prediction Probabilities")
                class_names = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']
                probs_df = {"Class": class_names, "Probability": raw_preds * 100}
                st.bar_chart(probs_df, x="Class", y="Probability")

# Run the app
if __name__ == "__main__":
    main()
    