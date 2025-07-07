import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tempfile

st.set_page_config(page_title="Moon Surface Feature Detector", layout="centered")
st.title("🌕 Moon Surface Feature Detector")

st.markdown("""
Detect **Boulders (🔵)** and **Landslides (🔴)** on Moon surface images.  
Upload 1 or 2 images to detect and compare features.  
Get **annotated output** and **CSV geometry reports**.
""")

uploaded_image = st.file_uploader("📁 Upload First Image (Current)", type=["jpg", "jpeg", "png"])
compare_image = st.file_uploader("📁 Upload Second Image (Older - Optional)", type=["jpg", "jpeg", "png"])

def process_image(img_bytes):
    np_img = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (800, 800))
    processed = cv2.equalizeHist(cv2.GaussianBlur(img, (5, 5), 0))
    edges = cv2.Canny(processed, 50, 150)
    return img, edges

def detect_features(img, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    data = []
    boulder_id = landslide_id = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        if 100 < area < 2000 and 0.6 < circularity <= 1.3:
            boulder_id += 1
            cv2.drawContours(output, [cnt], -1, (255, 0, 0), 2)
            cv2.putText(output, f"B{boulder_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            data.append(["Boulder", boulder_id, cx, cy, w, h, area, round(circularity, 2)])
        elif area > 3000 and circularity < 0.4:
            landslide_id += 1
            cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)
            cv2.putText(output, f"L{landslide_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            data.append(["Landslide", landslide_id, cx, cy, w, h, area, round(circularity, 2)])

    return output, data

if uploaded_image:
    img1, edges1 = process_image(uploaded_image)
    output1, features1 = detect_features(img1, edges1)
    
    st.image(output1, caption="🛰️ Annotated Moon Image", channels="BGR")
    df1 = pd.DataFrame(features1, columns=["Type", "ID", "X", "Y", "Length (px)", "Diameter (px)", "Area (px²)", "Circularity"])
    st.dataframe(df1)
    
    # Downloads
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
        cv2.imwrite(tmp_img.name, output1)
        st.download_button("📥 Download Annotated Image", open(tmp_img.name, "rb"), "moon_annotated.jpg", "image/jpeg")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='') as tmp_csv:
        df1.to_csv(tmp_csv.name, index=False)
        st.download_button("📑 Download Detection Data (CSV)", open(tmp_csv.name, "rb"), "moon_features.csv", "text/csv")

    # Temporal comparison if second image provided
    if compare_image:
        st.subheader("📊 Temporal Comparison")
        img2, edges2 = process_image(compare_image)
        _, features2 = detect_features(img2, edges2)
        df2 = pd.DataFrame(features2, columns=["Type", "ID", "X", "Y", "Length (px)", "Diameter (px)", "Area (px²)", "Circularity"])
        
        # Basic comparison based on count
        st.write(f"📌 Previous Image: {df2.Type.value_counts().to_dict()}")
        st.write(f"📌 Current Image: {df1.Type.value_counts().to_dict()}")

        new_boulders = df1[df1["Type"] == "Boulder"].shape[0] - df2[df2["Type"] == "Boulder"].shape[0]
        new_landslides = df1[df1["Type"] == "Landslide"].shape[0] - df2[df2["Type"] == "Landslide"].shape[0]
        st.info(f"🪨 Boulder change: {'+' if new_boulders >= 0 else ''}{new_boulders}")
        st.info(f"⛰️ Landslide change: {'+' if new_landslides >= 0 else ''}{new_landslides}")
