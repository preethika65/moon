import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="Moon Surface Feature Detector", layout="centered")

st.title("🌕 Moon Surface Feature Detector")
st.markdown("Upload a Moon image to detect **Boulders** (🔵) and **Landslides** (🔴) using classical image processing.")

uploaded_file = st.file_uploader("📁 Upload a Moon Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("❌ Could not read the image.")
    else:
        # Resize for consistency
        img = cv2.resize(img, (800, 800))

        # Preprocessing
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)
        edges = cv2.Canny(equalized, 50, 150)

        # Contour detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Count features
        boulder_count = 0
        landslide_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if 100 < area < 2000 and 0.6 < circularity <= 1.3:
                # Boulder: small and round
                boulder_count += 1
                cv2.drawContours(output, [cnt], -1, (255, 0, 0), 2)
            elif area > 3000 and circularity < 0.4:
                # Landslide: large and irregular
                landslide_count += 1
                cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

        # Show stats
        st.success(f"🪨 Boulders Detected: {boulder_count}")
        st.success(f"⛰️ Landslides Detected: {landslide_count}")

        # Display image
        st.image(output, channels="BGR", caption="📸 Detected Features")

        # Download output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, output)
            st.download_button("📥 Download Result Image", open(tmp.name, "rb"), "moon_features_detected.jpg", "image/jpeg")
