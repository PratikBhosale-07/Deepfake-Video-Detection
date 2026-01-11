"""
Deepfake Video Detection App
Streamlit web application for detecting deepfake videos using CNN+LSTM model
"""

import streamlit as st
import cv2
import torch
import numpy as np
from torchvision import models
import torch.nn as nn
from mtcnn import MTCNN
from PIL import Image
from torchvision import transforms
import tempfile
import os

# ============================================================
# Model Definition
# ============================================================

class CNN_LSTM(nn.Module):
    def __init__(self, hidden_dim=256):
        super(CNN_LSTM, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(2048, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        batch, seq, c, h, w = x.shape
        features = []
        for t in range(seq):
            f = self.cnn(x[:, t]).view(batch, -1)
            features.append(f)
        features = torch.stack(features, dim=1)
        out, _ = self.lstm(features)
        return self.fc(out[:, -1])

# ============================================================
# Load Model & Detector
# ============================================================

@st.cache_resource
def load_model_and_detector():
    """Load model and MTCNN detector (cached)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CNN_LSTM().to(device)
    model.load_state_dict(torch.load("model2_cnn_lstm.pth", map_location=device))
    model.eval()
    
    # MTCNN detector (TensorFlow-based, no parameters needed)
    detector = MTCNN()
    
    return model, detector, device

model, detector, device = load_model_and_detector()

# ============================================================
# Image Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================================
# Frame Extraction + Face Detection
# ============================================================

def extract_faces(video_path, seq_len=10):
    """Extract faces from video frames"""
    cap = cv2.VideoCapture(video_path)
    faces = []

    while len(faces) < seq_len:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb)

        if detections:
            x, y, w, h = detections[0]['box']
            face = rgb[y:y+h, x:x+w]
            face = Image.fromarray(face)
            face = transform(face)
            faces.append(face)

    cap.release()

    if len(faces) < seq_len:
        return None

    return torch.stack(faces)


# ============================================================
# Prediction
# ============================================================

def predict_video(video_path):
    """Predict if video is real or fake"""
    faces = extract_faces(video_path)

    if faces is None:
        return "Not enough faces detected", None

    faces = faces.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(faces)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][prediction].item()

    label = "FAKE" if prediction == 1 else "REAL"
    return label, confidence

# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 Deepfake Video Detection")
st.write("Upload a video to check if it's **REAL** or **DEEPFAKE**")

st.info("📌 The model analyzes facial features across video frames to detect manipulation")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    # Display video
    st.video(uploaded_file)

    # Analyze button
    if st.button("🔍 Analyze Video", type="primary"):
        with st.spinner("🧠 Analyzing video frames and detecting faces..."):
            result, confidence = predict_video(video_path)
            
            if confidence is None:
                st.warning(f"⚠️ {result}")
            else:
                # Display result with color coding
                if result == "REAL":
                    st.success(f"✅ Prediction: **{result}**")
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                else:
                    st.error(f"🚨 Prediction: **{result}**")
                    st.metric("Confidence", f"{confidence*100:.2f}%")

        # Clean up temporary file
        os.remove(video_path)

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses a CNN+LSTM deep learning model to detect deepfake videos.")
    
    st.subheader("How it works:")
    st.write("1. **Face Detection**: Extracts faces from video frames using MTCNN")
    st.write("2. **Feature Extraction**: Uses ResNet50 CNN to extract features")
    st.write("3. **Temporal Analysis**: LSTM analyzes temporal patterns")
    st.write("4. **Classification**: Predicts REAL or FAKE")
    
    st.divider()
    st.caption("Using device: " + device.upper())
