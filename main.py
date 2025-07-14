import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# File details
model_path = "flower_model.h5"
drive_file_id = "16e-cT4hByKhSaj-jrmp2UPymacDZK9QW"
gdown_url = f"https://drive.google.com/uc?id={drive_file_id}"

# Download if not exists
if not os.path.exists(model_path):
    with st.spinner("📥 Downloading model from Google Drive..."):
        gdown.download(gdown_url, model_path, quiet=False)
        if not os.path.exists(model_path):
            st.error("❌ Failed to download model. Check Google Drive link.")
            st.stop()

# Load the .h5 model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()

class_names = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

# UI
st.markdown("""
<h1 style='text-align: center; color: #9c27b0;'>🌼 Flower Classifier (.h5)</h1>
<p style='text-align: center;'>Upload a flower image and get prediction</p>
<hr>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    img = image.resize((180, 180))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)[0]
    pred_index = np.argmax(prediction)
    pred_class = class_names[pred_index]
    confidence = prediction[pred_index] * 100

    st.markdown(f"<h3 style='text-align:center;'>🔍 {pred_class.capitalize()} ({confidence:.2f}%)</h3>", unsafe_allow_html=True)

    st.subheader("📊 Prediction Confidence")
    fig, ax = plt.subplots(figsize=(7, 2))
    bars = ax.bar(class_names, prediction, color='#7e57c2')
    ax.set_ylim(0, 1)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}", ha='center', va='bottom')
    st.pyplot(fig)

# Footer
st.markdown("<hr><div style='text-align:center;'><small>Created by Kartik | TensorFlow + Streamlit</small></div>", unsafe_allow_html=True)
