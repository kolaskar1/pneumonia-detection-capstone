import io
import os
import cv2
import numpy as np
import pydicom
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Pneumonia Detection — Chest X-Ray Classifier", page_icon="🩺", layout="centered")
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_pneumonia_model.keras")
DEFAULT_THRESHOLD = 0.3142
IMG_SIZE = (128, 128)

@st.cache_resource
def load_pneumonia_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return tf.keras.models.load_model(model_path, safe_mode=False)

def preprocess_uploaded_file(uploaded_file, target_size=IMG_SIZE):
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()
    if filename.endswith(".dcm"):
        dcm = pydicom.dcmread(io.BytesIO(file_bytes))
        img = dcm.pixel_array.astype(np.float32)
    else:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
        img = np.array(pil_img, dtype=np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    normalized = np.clip(resized / 255.0, 0.0, 1.0)
    tensor = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)
    return tensor, normalized

st.title("🩺 Chest X-Ray Pneumonia Detection")
st.write("Upload a chest X-ray image (**DICOM `.dcm`**, **`.png`**, or **`.jpg`**) to estimate pneumonia probability.")
threshold = st.sidebar.slider("Clinical Decision Threshold", 0.05, 0.90, DEFAULT_THRESHOLD, 0.0001, format="%.4f")

model = load_pneumonia_model(MODEL_PATH)
uploaded_file = st.file_uploader("Upload Chest Radiograph", type=["dcm", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_tensor, preview_img = preprocess_uploaded_file(uploaded_file)
    prob = float(model.predict(input_tensor, verbose=0).ravel()[0])
    pred_class = "Pneumonia Positive (Lung Opacity)" if prob >= threshold else "Normal / Pneumonia Negative"
    st.image((preview_img * 255).astype(np.uint8), caption=f"Preprocessed ({IMG_SIZE[0]}x{IMG_SIZE[1]}) — {uploaded_file.name}", use_container_width=True)
    col1, col2 = st.columns(2)
    col1.metric("Predicted Class", pred_class)
    col2.metric("Pneumonia Probability", f"{prob:.4f} ({prob * 100:.2f}%)")
