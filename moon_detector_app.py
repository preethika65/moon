import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import rasterio
import tempfile
import os
from skimage import feature, exposure, measure, filters, color, util

st.set_page_config(page_title="Moon Surface Feature Detector", layout="centered")
st.title("🌕 Moon Surface Feature Detector (No OpenCV)")

uploaded = st.file_uploader("📁 Upload a Moon Image (.tif, .img, .jpg, .png)", type=None)

def read_geotiff(file):
    with rasterio.open(file) as src:
        array = src.read(1)
        norm_img = ((array - array.min()) / (array.max() - array.min()) * 255).astype('uint8')
    return norm_img

if uploaded:
    ext = os.path.splitext(uploaded.name)[-1].lower()

    if ext not in ['.tif', '.img', '.jpg', '.jpeg', '.png']:
        st.error("❌ Only .tif, .img, .jpg, .jpeg, .png supported.")
    else:
        try:
            if ext in ['.tif', '.img']:
                gray = read_geotiff(uploaded)
            else:
                img = Image.open(uploaded).convert("L")
                gray = np.array(img)

            # Apply filters
            gray_eq = exposure.equalize_hist(gray)
            edges = feature.canny(gray_eq, sigma=2.0)

            # Measure regions
            contours = measure.find_contours(edges, 0.8)
            output_img = Image.fromarray(gray).convert("RGB")
            draw = ImageDraw.Draw(output_img)

            detections = []
            b_id = l_id = 0

            for contour in contours:
                contour = np.array(contour, dtype=int)
                x_coords = contour[:, 1]
                y_coords = contour[:, 0]

                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()

                w = x_max - x_min
                h = y_max - y_min
                area = w * h
                cx = x_min + w // 2
                cy = y_min + h // 2
                circularity = 4 * np.pi * area / ((w + h) ** 2 + 1e-6)

                if area > 2500 and 0.5 < circularity <= 1.2:
                    b_id += 1
                    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="blue", width=2)
                    draw.text((x_min, y_min - 10), f"B{b_id}", fill="blue")
                    detections.append(["Boulder", b_id, cx, cy, w, h, area, round(circularity, 2)])

                elif area > 3000 and circularity < 0.4:
                    l_id += 1
                    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)
                    draw.text((x_min, y_min - 10), f"L{l_id}", fill="red")
                    detections.append(["Landslide", l_id, cx, cy, w, h, area, round(circularity, 2)])

            # Display result
            st.image(output_img, caption="🛰️ Annotated Moon Surface")

            df = pd.DataFrame(detections, columns=["Type", "ID", "X", "Y", "Length", "Diameter", "Area", "Circularity"])
            st.dataframe(df)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                output_img.save(tmp_img.name)
                st.download_button("📸 Download Annotated Image", open(tmp_img.name, "rb"), "moon_annotated.png", "image/png")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='') as tmp_csv:
                df.to_csv(tmp_csv.name, index=False)
                st.download_button("📑 Download Feature Data (CSV)", open(tmp_csv.name, "rb"), "moon_features.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
