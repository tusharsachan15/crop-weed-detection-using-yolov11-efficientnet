# ==== Streamlit Page Config MUST come first ====
import streamlit as st
st.set_page_config(page_title="AgriVision 🌿", layout="centered", initial_sidebar_state="expanded")

# ==== Imports ====
import numpy as np
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import base64
import time
import os
from ultralytics import YOLO

# ==== Background Image ====
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image_path = "Agrivision_Streamlit/assets/bg.png"
bg_base64 = get_base64_image(bg_image_path)

st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{bg_base64}") no-repeat center center fixed;
        background-size: cover;
    }}

    .main-box {{
        background-color: rgba(0, 0, 0, 0.4);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem auto;
        max-width: 960px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
    }}

    .main-box h1,
    .main-box h2,
    .main-box p,
    .main-box label,
    .main-box .stRadio > div,
    .main-box .stSlider,
    .main-box .stSelectbox,
    .main-box .stMetric {{
        color: #ffffff !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ==== Sidebar ====
st.sidebar.markdown("🧠 Model Selection")
model_choice = st.sidebar.radio("Choose model", ["YOLOv11", "Hybrid", "Compare Both"])
conf_threshold = st.sidebar.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# ==== Load Models ====
@st.cache_resource
def load_yolov11():
    path = "Agrivision_Streamlit/app/yolov11n.pt"
    if not os.path.exists(path):
        st.error(f"🚫 YOLOv11 model not found at: {path}")
        st.stop()
    return YOLO(path)

@st.cache_resource
def load_hybrid_model():
    path = "Agrivision_Streamlit/app/yolov11_efficientnet.pt"
    if not os.path.exists(path):
        st.error(f"🚫 Hybrid model not found at: {path}")
        st.stop()
    return YOLO(path)

model_yolo = load_yolov11()
model_hybrid = load_hybrid_model()

CLASS_NAMES = ["crop", "weed"]
COLORS = [(255, 0, 0), (0, 255, 255)]

# ==== Detection Logic ====
def run_and_draw(image_bgr, model):
    start = time.time()
    results = model.predict(source=image_bgr, conf=conf_threshold, verbose=False)
    elapsed = time.time() - start
    result = results[0]
    output = image_bgr.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return output, 0, 0, elapsed

    classes = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy().astype(int)

    crop_count = sum(1 for cls in classes if cls == 0)
    weed_count = sum(1 for cls in classes if cls == 1)

    for i, cls in enumerate(classes):
        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
        color = COLORS[cls % len(COLORS)]
        conf = confs[i]
        x1, y1, x2, y2 = xyxy[i]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = y1 if y1 - th - 10 > 0 else y1 + th + 10
        cv2.rectangle(output, (x1, y_text - th - 4), (x1 + tw + 6, y_text), (0, 0, 0), -1)
        cv2.putText(output, label_text, (x1 + 3, y_text - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output, crop_count, weed_count, elapsed

# ==== Pie Chart ====
def show_pie_chart(crops, weeds, chart_key):
    fig = go.Figure(data=[go.Pie(
        labels=["Crops", "Weeds"],
        values=[crops, weeds],
        hole=0.4,
        marker=dict(colors=["#00cc96", "#636efa"]),
        textinfo="label+percent",
        textfont_size=16
    )])
    fig.update_layout(
        title={"text": "🌿 Crop vs Weed Distribution", "x": 0.5},
        paper_bgcolor="rgba(0,0,0,0.4)",
        plot_bgcolor="rgba(0,0,0,0.4)",
        font=dict(color="white")
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

# ==== UI Wrapper ====
with st.container():
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    st.markdown("<h1>🌿 AgriVision – Smart Crop & Weed Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px;'>Dual Model | Image, Camera & Video Support</p>", unsafe_allow_html=True)

    input_type = st.radio("📁 Choose Input Type", ["Image", "Camera"])

    if input_type == "Image":
        image_files = st.file_uploader("📤 Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if image_files:
            for i, img_file in enumerate(image_files):
                st.subheader(f"🖼️ Image {i+1}")
                image = Image.open(img_file).convert("RGB")
                image_np = np.array(image)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                st.image(image, caption="📷 Original Image", use_column_width=True)

                if model_choice == "Compare Both":
                    col1, col2 = st.columns(2)
                    with col1:
                        result_img1, crops1, weeds1, time1 = run_and_draw(image_bgr, model_yolo)
                        st.image(cv2.cvtColor(result_img1, cv2.COLOR_BGR2RGB), caption="YOLOv11", use_column_width=True)
                        st.metric("🌿 Crops", crops1)
                        st.metric("🌾 Weeds", weeds1)
                        st.info(f"⏱️ Time: {time1:.2f}s")

                    with col2:
                        result_img2, crops2, weeds2, time2 = run_and_draw(image_bgr, model_hybrid)
                        st.image(cv2.cvtColor(result_img2, cv2.COLOR_BGR2RGB), caption="Hybrid Model", use_column_width=True)
                        st.metric("🌿 Crops", crops2)
                        st.metric("🌾 Weeds", weeds2)
                        st.info(f"⏱️ Time: {time2:.2f}s")

                    comparison_df = pd.DataFrame({
                        "Model": ["YOLOv11", "Hybrid"],
                        "Crops": [crops1, crops2],
                        "Weeds": [weeds1, weeds2],
                        "Detection Time (s)": [time1, time2]
                    })
                    st.dataframe(comparison_df, use_container_width=True)
                    fig = px.bar(comparison_df.melt(id_vars="Model"),
                                 x="Model", y="value", color="variable",
                                 barmode="group", title="📈 Accuracy Comparison")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0.4)", plot_bgcolor="rgba(0,0,0,0.4)", font=dict(color="white"))
                    st.plotly_chart(fig, use_container_width=True, key=f"compare_pie_{i}")
                else:
                    model = model_yolo if model_choice == "YOLOv11" else model_hybrid
                    result_img, crops, weeds, elapsed = run_and_draw(image_bgr, model)
                    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, caption="🧠 Detected Image", use_column_width=True)
                    st.metric("🌿 Crops", crops)
                    st.metric("🌾 Weeds", weeds)
                    st.info(f"⏱️ Detection Time: {elapsed:.2f} seconds")
                    show_pie_chart(crops, weeds, chart_key=f"pie_{i}")

    elif input_type == "Camera":
        if st.toggle("📸 Enable Camera"):
            camera_image = st.camera_input("Take a Photo")
            if camera_image:
                image = Image.open(camera_image).convert("RGB")
                image_np = np.array(image)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                st.image(image, caption="📷 Captured Image", use_column_width=True)

                if model_choice == "Compare Both":
                    col1, col2 = st.columns(2)
                    with col1:
                        result_img1, crops1, weeds1, time1 = run_and_draw(image_bgr, model_yolo)
                        st.image(cv2.cvtColor(result_img1, cv2.COLOR_BGR2RGB), caption="YOLOv11", use_column_width=True)
                        st.metric("🌿 Crops", crops1)
                        st.metric("🌾 Weeds", weeds1)
                        st.info(f"⏱️ Time: {time1:.2f}s")

                    with col2:
                        result_img2, crops2, weeds2, time2 = run_and_draw(image_bgr, model_hybrid)
                        st.image(cv2.cvtColor(result_img2, cv2.COLOR_BGR2RGB), caption="Hybrid Model", use_column_width=True)
                        st.metric("🌿 Crops", crops2)
                        st.metric("🌾 Weeds", weeds2)
                        st.info(f"⏱️ Time: {time2:.2f}s")

                    comparison_df = pd.DataFrame({
                        "Model": ["YOLOv11", "Hybrid"],
                        "Crops": [crops1, crops2],
                        "Weeds": [weeds1, weeds2],
                        "Detection Time (s)": [time1, time2]
                    })
                    st.dataframe(comparison_df, use_container_width=True)
                    fig = px.bar(comparison_df.melt(id_vars="Model"),
                                 x="Model", y="value", color="variable",
                                 barmode="group", title="📊 Camera Input Comparison")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0.4)", plot_bgcolor="rgba(0,0,0,0.4)", font=dict(color="white"))
                    st.plotly_chart(fig, use_container_width=True, key="camera_comparison")
                else:
                    model = model_yolo if model_choice == "YOLOv11" else model_hybrid
                    result_img, crops, weeds, elapsed = run_and_draw(image_bgr, model)
                    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, caption="🧠 Detected Image", use_column_width=True)
                    st.metric("🌿 Crops", crops)
                    st.metric("🌾 Weeds", weeds)
                    st.info(f"⏱️ Detection Time: {elapsed:.2f} seconds")
                    show_pie_chart(crops, weeds, chart_key="camera_chart")

    st.markdown('</div>', unsafe_allow_html=True)
