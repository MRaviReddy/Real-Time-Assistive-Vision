import streamlit as st
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import model
import io
import tempfile
from gtts import gTTS
import cv2
import time
import pyrealsense2 as rs
from ultralytics import YOLO
import threading
import queue
from playsound import playsound
import os


# Thread-safe audio queue
audio_queue = queue.Queue()


COCO_INSTANCE_CATEGORY_NAMES = [ 
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
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
    boxes = prediction['boxes'][prediction['scores'] > 0.5].cpu()
    labels = prediction['labels'][prediction['scores'] > 0.5].cpu()
    label_names = [COCO_INSTANCE_CATEGORY_NAMES[i] if i < len(COCO_INSTANCE_CATEGORY_NAMES) else str(i) for i in labels]
    image_pil = image.copy()
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    for box, label in zip(boxes, label_names):
        x0, y0, x1, y1 = map(int, box.tolist())
        draw.rectangle([x0, y0, x1, y1], outline="green", width=4)
        text = label
        text_size = draw.textbbox((x0, y0), text, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        draw.rectangle([x0, y0 - text_height - 6, x0 + text_width + 6, y0], fill="green")
        draw.text((x0 + 3, y0 - text_height - 3), text, fill="white", font=font)
    return list(set(label_names)), image_pil

if 'audio_data' not in st.session_state:
    st.session_state['audio_data'] = None
if 'last_labels' not in st.session_state:
    st.session_state['last_labels'] = set()

def generate_speech(labels, lang='en'):
    if not labels:
        return None
    object_list = ", ".join(labels)
    text = {
        "en": f"The detected objects are {object_list}.",
        "hi": f"पता चले वस्तुएं हैं: {object_list}.",
        "te": f"గుర్తించబడిన వస్తువులు: {object_list}.",
        "kn": f"ಹೆಚ್ಚುಗೊಂಡ ವಸ್ತುಗಳು: {object_list}.",
        "de": f"Die erkannten Objekte sind {object_list}.",
        "es": f"Los objetos detectados son {object_list}."
    }.get(lang, f"The detected objects are {object_list}.")
    try:
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

from playsound import playsound
import tempfile
import os
audio_playback_lock = threading.Lock()


def speak_labels_threaded(labels, lang='en'):
    def speak():
        if audio_playback_lock.locked():
            return  # Skip if already playing audio

        with audio_playback_lock:
            audio_bytes = generate_speech(labels, lang)
            if audio_bytes:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tmp.write(audio_bytes)
                        temp_path = tmp.name
                    playsound(temp_path)
                    os.remove(temp_path)
                except Exception as e:
                    print(f"🔊 Playback error: {e}")

    threading.Thread(target=speak, daemon=True).start()

    



def run_video_stream_realsense(pipeline, enhancer, yolo_model, frcnn_model, device, lang):
    col1, col2, col3, col4 = st.columns(4)
    stframe_original = col1.empty()
    stframe_enhanced = col2.empty()
    stframe_yolo = col3.empty()
    stframe_frcnn = col4.empty()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            if frame is None or frame.size == 0:
                continue
            stframe_original.image(frame, channels="BGR", caption="🎥 Original Camera", use_container_width=True)
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            enhanced_pil, _, _ = enhance_image(image_pil, enhancer, device)
            enhanced_bgr = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
            stframe_enhanced.image(enhanced_bgr, channels="BGR", caption="✨ Enhanced", use_container_width=True)
            yolo_labels, detection_img_yolo = detect_yolo(enhanced_pil, yolo_model)
            yolo_bgr = cv2.cvtColor(np.array(detection_img_yolo), cv2.COLOR_RGB2BGR)
            stframe_yolo.image(yolo_bgr, channels="BGR", caption="🕵️ YOLO Detection", use_container_width=True)
            frcnn_labels, detection_img_frcnn = detect_frcnn(enhanced_pil, frcnn_model, device)
            frcnn_bgr = cv2.cvtColor(np.array(detection_img_frcnn), cv2.COLOR_RGB2BGR)
            stframe_frcnn.image(frcnn_bgr, channels="BGR", caption="🕵️ Faster R-CNN Detection", use_container_width=True)
            new_labels = set(yolo_labels) | set(frcnn_labels)
            if new_labels != st.session_state['last_labels']:
                st.session_state['last_labels'] = new_labels
                speak_labels_threaded(list(new_labels), lang)
    except Exception as e:
        st.error(f"⚠️ RealSense camera stream error: {e}")
    finally:
        pipeline.stop()

def run_video_stream_cv(cap, enhancer, yolo_model, frcnn_model, device, lang):
    col1, col2, col3, col4 = st.columns(4)
    stframe_original = col1.empty()
    stframe_enhanced = col2.empty()
    stframe_yolo = col3.empty()
    stframe_frcnn = col4.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("❌ Camera frame not received.")
            break
        stframe_original.image(frame, channels="BGR", caption="🎥 Original Camera", use_container_width=True)
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        enhanced_pil, _, _ = enhance_image(image_pil, enhancer, device)
        enhanced_bgr = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
        stframe_enhanced.image(enhanced_bgr, channels="BGR", caption="✨ Enhanced", use_container_width=True)
        yolo_labels, detection_img_yolo = detect_yolo(enhanced_pil, yolo_model)
        yolo_bgr = cv2.cvtColor(np.array(detection_img_yolo), cv2.COLOR_RGB2BGR)
        stframe_yolo.image(yolo_bgr, channels="BGR", caption="🕵️ YOLO Detection", use_container_width=True)
        frcnn_labels, detection_img_frcnn = detect_frcnn(enhanced_pil, frcnn_model, device)
        frcnn_bgr = cv2.cvtColor(np.array(detection_img_frcnn), cv2.COLOR_RGB2BGR)
        stframe_frcnn.image(frcnn_bgr, channels="BGR", caption="🕵️ Faster R-CNN Detection", use_container_width=True)
        new_labels = set(yolo_labels) | set(frcnn_labels)
        if new_labels != st.session_state['last_labels']:
            st.session_state['last_labels'] = new_labels
            speak_labels_threaded(list(new_labels), lang)
    cap.release()

st.set_page_config(page_title="Assistive Vision System", layout="wide")
st.title("🎥 Real-Time Assistive Vision (RealSense or Video Upload)")
language = st.selectbox("🗣️ Select language", ["en", "hi", "te", "kn", "de", "es"])
video_upload = st.file_uploader("🎼 Upload a video if RealSense is not available", type=["mp4", "avi", "mov"])
if st.button("🔴 Start Assistive Vision"):
    enhancer_model_path = "C:/Users/IndiaAI Data Lab/new_project_gpu/pyrealsense/Zero-DCE-master/Zero-DCE_code/snapshots/Epoch99.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enhancer = load_enhancer(enhancer_model_path, device)
    yolo_model = load_yolo()
    frcnn_model = load_frcnn(device)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        profile = pipeline.start(config)
        time.sleep(1)
        st.success("✅ RealSense camera initialized.")
        run_video_stream_realsense(pipeline, enhancer, yolo_model, frcnn_model, device, language)
    except Exception as e:
        st.warning(f"⚠️ RealSense failed: {e}")
        if video_upload is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_upload.read())
            cap = cv2.VideoCapture(tfile.name)
            st.info("🎼 Using uploaded video.")
            run_video_stream_cv(cap, enhancer, yolo_model, frcnn_model, device, language)
        else:
            st.error("❌ No RealSense camera found and no video uploaded.")
            st.stop()

