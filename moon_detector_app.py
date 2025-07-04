import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import pandas as pd

st.set_page_config(page_title="🌕 Moon Feature Detector", layout="centered")

st.title("🌕 Moon Surface Feature Detector")
st.markdown("""
This tool helps analyze Moon surface images to detect:
- **Boulders** (🔵): Small, round features
- **Landslides** (🔴): Irregular, large features

You'll receive:
- 📸 Annotated image
- 📑 CSV report with geometry (length, diameter, area, circularity)
""")

# Upload image
uploaded_file = st.file_uploader("📁 Upload a Moon Surface Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("❌ Could not read image.")
    else:
        # Resize and pre-process image
        img = cv2.resize(img, (800, 800))
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)
        edges = cv2.Canny(equalized, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        boulder_count, landslide_count = 0, 0
        detection_data = []

        # Analyze contours
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

            if 100 < area < 2000 and 0.6 < circularity <= 1.3:
                boulder_count += 1
                cv2.drawContours(output, [cnt], -1, (255, 0, 0), 2)
                cv2.putText(output, f"B{boulder_count}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                detection_data.append(["Boulder", boulder_count, cx, cy, round(w, 2), round(h, 2), round(area, 2), round(circularity, 2)])

            elif area > 3000 and circularity < 0.4:
                landslide_count += 1
                cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)
                cv2.putText(output, f"L{landslide_count}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                detection_data.append(["Landslide", landslide_count, cx, cy, round(w, 2), round(h, 2), round(area, 2), round(circularity, 2)])

        # Display detection summary
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🪨 **Boulders Detected:** {boulder_count}")
        with col2:
            st.success(f"⛰️ **Landslides Detected:** {landslide_count}")

        # Display annotated image
        st.image(output, channels="BGR", caption="📸 Annotated Moon Surface")

        # Save annotated image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            cv2.imwrite(tmp_img.name, output)
            st.download_button("📥 Download Annotated Image", open(tmp_img.name, "rb"), "moon_features_detected.jpg", "image/jpeg")

        # Display and save CSV data
        df = pd.DataFrame(detection_data, columns=[
            "Type", "ID", "X_center", "Y_center", 
            "Length (px)", "Diameter (px)", "Area (px²)", "Circularity"
        ])

        st.markdown("### 📊 Detection Data")
        st.dataframe(df)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='') as tmp_csv:
            df.to_csv(tmp_csv.name, index=False)
            st.download_button("📑 Download Detection Data (CSV)", open(tmp_csv.name, "rb"), "detection_data.csv", "text/csv")
