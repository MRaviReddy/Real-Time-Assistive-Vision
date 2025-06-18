# --- Full Assistive Vision System with Properly Sized Images ---

import streamlit as st
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
import numpy as np
from PIL import Image
import model
import io
from gtts import gTTS
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from ultralytics import YOLO

# COCO class names
COCO_INSTANCE_CATEGORY_NAMES =  [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_frcnn(device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval().to(device)
    return model

@st.cache_resource
def load_enhancer(model_path, device):
    net = model.enhance_net_nopool().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return net

def enhance_image(image, enhancer, device):
    img_np = np.asarray(image.convert("RGB")) / 255.0
    tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        _, enhanced, _ = enhancer(tensor)
    enhanced_np = enhanced.squeeze().permute(1, 2, 0).cpu().numpy().clip(0, 1)
    return Image.fromarray((enhanced_np * 255).astype(np.uint8)), img_np, enhanced

def detect_yolo(image, model):
    results = model.predict(image, conf=0.35)
    labels = list(set([model.names[int(cls)] for cls in results[0].boxes.cls]))
    return labels, Image.fromarray(results[0].plot())

def detect_frcnn(image, model, device):
    transform = torchvision.transforms.ToTensor()
    tensor_img = transform(image).to(device)
    with torch.no_grad():
        prediction = model([tensor_img])[0]
    boxes = prediction['boxes'][prediction['scores'] > 0.5]
    labels = prediction['labels'][prediction['scores'] > 0.5]
    label_names = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in labels.cpu()]
    img_boxed = draw_bounding_boxes((tensor_img * 255).byte(), boxes.cpu(), labels=label_names, colors="green", width=4, font_size=20)
    return list(set(label_names)), F.to_pil_image(img_boxed)

def generate_speech(labels, lang='en'):
    object_list = ", ".join(labels)
    if lang == "en":
        text = f"The detected objects are {object_list}."
    elif lang == "hi":
        text = f"पता चले वस्तुएं हैं: {object_list}।"
    elif lang == "te":
        text = f"గుర్తించబడిన వస్తువులు: {object_list}."
    elif lang == "kn":
        text = f"ಹೆಚ್ಚುಗೊಂಡ ವಸ್ತುಗಳು: {object_list}."
    elif lang == "de":
        text = f"Die erkannten Objekte sind {object_list}."
    elif lang == "es":
        text = f"Los objetos detectados son {object_list}."
    else:
        text = f"The detected objects are {object_list}."
    tts = gTTS(text=text, lang=lang)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

st.set_page_config(page_title="Assistive Vision | YOLO + FRCNN", layout="wide")

# Custom CSS for centered images with fixed width
st.markdown("""
    <style>
        .centered-image {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }
        .centered-image img {
            width: 400px;
            height: auto;
            border: 2px solid #ccc;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .stDownloadButton > button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🦯 Assistive Vision System for BVI")
st.markdown("Upload ➤ Enhance ➤ Detect ➤ Speak")

uploaded_file = st.file_uploader("📤 Upload low-light image", type=["jpg", "jpeg", "png"])
language = st.selectbox("🗣️ Language for speech", ["en", "hi", "te", "kn", "de", "es"])

if "enhanced_image" not in st.session_state:
    st.session_state.enhanced_image = None
    st.session_state.original_np = None
    st.session_state.enhanced_tensor = None
    st.session_state.image = None

if uploaded_file:
    st.session_state.image = Image.open(uploaded_file).convert("RGB")
    st.markdown('<div class="centered-image">', unsafe_allow_html=True)
    st.image(st.session_state.image, caption="📷 Original Image")
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("✨ Enhance Image") and st.session_state.image:
    with st.spinner("Enhancing..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enhancer = load_enhancer(r"C:\\Users\\IndiaAI Data Lab\\new_project_gpu\\Zero-DCE-master\\Zero-DCE_code\\snapshots\\Epoch99.pth", device)
        enhanced_img, orig_np, enhanced_tensor = enhance_image(st.session_state.image, enhancer, device)
        st.session_state.enhanced_image = enhanced_img
        st.session_state.original_np = orig_np
        st.session_state.enhanced_tensor = enhanced_tensor
        st.markdown('<div class="centered-image">', unsafe_allow_html=True)
        st.image(enhanced_img, caption="🔆 Enhanced Image")
        st.markdown('</div>', unsafe_allow_html=True)

if st.button("🎯 Detect Objects (YOLO + FRCNN)") and st.session_state.enhanced_image:
    with st.spinner("Detecting..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        yolo_model = load_yolo()
        frcnn_model = load_frcnn(device)

        yolo_labels, yolo_annotated = detect_yolo(st.session_state.enhanced_image, yolo_model)
        frcnn_labels, frcnn_annotated = detect_frcnn(st.session_state.enhanced_image, frcnn_model, device)

        enhanced_np = np.asarray(st.session_state.enhanced_image) / 255.0
        psnr_val = psnr(st.session_state.original_np, enhanced_np, data_range=1.0)
        ssim_val = ssim(st.session_state.original_np, enhanced_np, channel_axis=-1, data_range=1.0)

        st.subheader("📸 Detection Comparison")
        col1, col2 = st.columns(2)
        col1.image(yolo_annotated, caption=f"YOLOv8: {', '.join(yolo_labels) or 'None'}", use_container_width=True)
        col2.image(frcnn_annotated, caption=f"Faster R-CNN: {', '.join(frcnn_labels) or 'None'}", use_container_width=True)

        st.subheader("📊 Enhancement Quality")
        st.markdown(f"- **PSNR**: `{psnr_val:.2f} dB`")
        st.markdown(f"- **SSIM**: `{ssim_val:.4f}`")

        if yolo_labels:
            st.subheader("🔊 YOLOv8 Speech")
            yolo_speech = generate_speech(yolo_labels, lang=language)
            st.audio(yolo_speech, format="audio/mp3")

        if frcnn_labels:
            st.subheader("🔊 Faster R-CNN Speech")
            frcnn_speech = generate_speech(frcnn_labels, lang=language)
            st.audio(frcnn_speech, format="audio/mp3")

        buf = io.BytesIO()
        st.session_state.enhanced_image.save(buf, format="PNG")
        st.download_button("⬇️ Download Enhanced Image", data=buf.getvalue(), file_name="enhanced.png", mime="image/png")
