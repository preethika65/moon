import streamlit as st
import numpy as np
import pandas as pd
import cv2
import rasterio
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Moon Feature Detector", layout="centered")
st.title("🌕 Moon Surface Feature Detector")

st.markdown("""
This tool helps analyze Moon surface images to detect:

- 🔵 **Boulders** (big rocks): Round, large features  
- 🔴 **Landslides**: Irregular, large terrain changes

You'll receive:
- 📸 Annotated image with labeled features  
- 📑 CSV report with geometry (length, diameter, area, circularity)
""")

uploaded = st.file_uploader(
    "📁 Upload a Moon Surface Image (TIF, IMG, JPG, PNG)", 
    type=None  # Allow all MIME types (especially .img)
)

def read_geotiff(file):
    with rasterio.open(file) as src:
        array = src.read(1)
        norm_img = ((array - array.min()) / (array.max() - array.min()) * 255).astype('uint8')
    return norm_img

def detect_features(img):
    img = cv2.resize(img, (800, 800))
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    edges = cv2.Canny(equalized, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    data = []
    b_id = l_id = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        if area > 2500 and 0.5 < circularity <= 1.2:
            b_id += 1
            cv2.drawContours(output, [cnt], -1, (255, 0, 0), 2)
            cv2.putText(output, f"B{b_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            data.append(["Boulder", b_id, cx, cy, w, h, area, round(circularity, 2)])

        elif area > 3000 and circularity < 0.4:
            l_id += 1
            cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)
            cv2.putText(output, f"L{l_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            data.append(["Landslide", l_id, cx, cy, w, h, area, round(circularity, 2)])

    return output, data

if uploaded:
    file_ext = os.path.splitext(uploaded.name)[-1].lower()

    if file_ext not in ['.tif', '.img', '.jpg', '.jpeg', '.png']:
        st.error("❌ Only .tif, .img, .jpg, .jpeg, and .png files are supported.")
    else:
        try:
            if file_ext in ['.tif', '.img']:
                img_gray = read_geotiff(uploaded)
            else:
                bytes_img = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                img_gray = cv2.imdecode(bytes_img, cv2.IMREAD_GRAYSCALE)

            output_img, features = detect_features(img_gray)
            st.image(output_img, caption="🛰️ Annotated Moon Surface", channels="BGR")

            df = pd.DataFrame(features, columns=["Type", "ID", "X", "Y", "Length", "Diameter", "Area", "Circularity"])
            st.dataframe(df)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                cv2.imwrite(tmp_img.name, output_img)
                st.download_button("📸 Download Annotated Image", open(tmp_img.name, "rb"), "moon_annotated.jpg", "image/jpeg")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline='') as tmp_csv:
                df.to_csv(tmp_csv.name, index=False)
                st.download_button("📑 Download Feature Data (CSV)", open(tmp_csv.name, "rb"), "moon_features.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
