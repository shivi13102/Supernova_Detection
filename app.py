import streamlit as st
import tempfile
import os
from preprocessing import preprocess_supernova_image
from ultralytics import YOLO
import cv2
from PIL import Image
import pandas as pd

# Import the feature extractor
from feature_extraction import extract_features

# Load your trained YOLOv8 model
model = YOLO("runs/train/yolov8s_panstarrs23/weights/best.pt")

st.set_page_config(page_title="Supernova Detection", layout="centered")
st.title("🔭 Supernova Detection from Astronomical Images")
st.write("Upload a raw astronomical image. It will be preprocessed and then passed through the YOLOv8 model to detect potential supernovae. After detection, image features will also be extracted and displayed.")

uploaded_file = st.file_uploader("📁 Upload a raw PNG image", type=["png"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, "raw.png")
        preprocessed_path = os.path.join(tmpdir, "processed.png")

        # Save uploaded file to disk
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(Image.open(raw_path), caption="Original Uploaded Image", use_container_width=True)

        # Preprocess the image
        st.info("⏳ Preprocessing image...")
        preprocess_supernova_image(raw_path, preprocessed_path)

        if not os.path.exists(preprocessed_path):
            st.error("❌ Preprocessing failed. Please check your image.")
        else:
            st.success("✅ Preprocessing completed.")
            st.image(Image.open(preprocessed_path), caption="Preprocessed Image", use_container_width=True)

            # Run YOLOv8 inference
            st.info("🔍 Running supernova detection...")
            results = model(preprocessed_path)

            # Render prediction
            boxes_image = results[0].plot()  # Draw detections
            st.image(boxes_image, caption="🧠 Detection Result", use_container_width=True)

            # Optional: Show raw prediction data
            if st.checkbox("Show raw detection details"):
                st.json(results[0].tojson())

            # Extract and display features
            st.info("📊 Extracting features from the preprocessed image...")
            features = extract_features(preprocessed_path)

            if features:
                # List of feature names (update if needed)
                feature_names = [
                    "Image Path", "Mean Intensity", "Median Intensity", "Standard Deviation", "Skewness", "Kurtosis",
                    "Histogram Entropy", "Bright Pixel Count", "Edge Count", "Laplacian Variance",
                    "GLCM Contrast", "GLCM Homogeneity", "GLCM Energy", "GLCM Texture Entropy",
                    "Circularity", "Bounding Box Area", "FFT Energy", "Dominant Frequency"
                ]

                st.success("✅ Feature extraction complete.")
                st.subheader("🧬 Extracted Features")

                # Create vertical table
                feature_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Value": features
                })

                st.dataframe(feature_df.set_index("Feature"), use_container_width=True)
            else:
                st.error("❌ Feature extraction failed.")