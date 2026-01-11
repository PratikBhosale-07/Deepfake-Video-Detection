# 🎭 Deepfake Video Detection System

A comprehensive deep learning-based system for detecting deepfake videos using advanced computer vision and temporal analysis techniques. This project implements a CNN+LSTM architecture that analyzes facial features across video sequences to identify manipulated content.

## 🌟 Features

- **Dual Model Architecture**:
  - **ResNet50**: Transfer learning-based image classification
  - **CNN+LSTM**: Temporal sequence analysis for enhanced accuracy
- **Interactive Web Interface**: Streamlit-based application for real-time video analysis
- **Automated Pipeline**: Complete preprocessing workflow from video to prediction
- **Face Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) for improved feature extraction
- **Progress Tracking**: Resume capability for long-running processing tasks
- **Comprehensive Metrics**: Detailed evaluation with ROC curves, confusion matrices, and classification reports

## 🏗️ Architecture

### Model 1: ResNet50 (Transfer Learning)

- Pre-trained ResNet50 backbone
- Fine-tuned for binary classification (Real vs Fake)
- Single-frame analysis

### Model 2: CNN+LSTM (Sequence-based)

- **CNN**: ResNet50 feature extractor
- **LSTM**: Temporal pattern recognition across frame sequences
- Analyzes 10-frame sequences for temporal inconsistencies
- Superior performance by capturing temporal artifacts

## 📁 Project Structure

```
MAJOR_PROJECT/
├── app.py                          # Streamlit web application
├── model2_cnn_lstm.pth             # Trained CNN+LSTM model weights
│
├── Dataset/
│   ├── frames/                     # Extracted video frames
│   │   ├── real/
│   │   ├── fake/
│   │   └── metadata.csv
│   ├── faces/                      # Detected and cropped faces
│   │   ├── real/
│   │   ├── fake/
│   │   └── metadata_faces.csv
│   ├── faces_clahe/                # CLAHE-enhanced faces
│   │   ├── real/
│   │   └── fake/
│   └── processed/                  # Train/Val/Test splits
│       ├── train/
│       ├── val/
│       ├── test/
│       ├── metadata_train.csv
│       ├── metadata_val.csv
│       └── metadata_test.csv
│
├── models/                         # Saved model checkpoints
├── results/                        # Training logs and metrics
│   └── training_logs.csv
│
├── Processing Scripts/
│   ├── face_extraction.py          # Extract faces from frames using MTCNN
│   ├── dataset_processor.py        # Split dataset into train/val/test
│   ├── clahe_enhancer.py           # Apply CLAHE enhancement
│   ├── sequence_dataset.py         # Sequence dataset for CNN+LSTM
│   └── dataset_and_transforms.py   # Dataset class and transforms
│
├── Model Notebooks/
│   ├── model1_resnet50.ipynb       # ResNet50 training
│   ├── model2_cnn_lstm.ipynb       # CNN+LSTM training
│   └── Model2 Evaluation.ipynb     # Model evaluation metrics
│
└── Progress Files/
    ├── face_extraction_progress.json
    ├── dataset_processing_progress.json
    ├── clahe_enhancement_progress.json
    └── sequence_processing_progress_*.json
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd MAJOR_PROJECT
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install streamlit opencv-python mtcnn pandas numpy scikit-learn matplotlib pillow
```

**Required packages:**

- `torch` >= 2.0.0
- `torchvision` >= 0.15.0
- `opencv-python` >= 4.8.0
- `mtcnn` >= 0.1.1
- `streamlit` >= 1.28.0
- `pandas` >= 2.0.0
- `numpy` >= 1.24.0
- `scikit-learn` >= 1.3.0
- `matplotlib` >= 3.7.0
- `pillow` >= 10.0.0

## 📊 Dataset Preparation

### Step 1: Extract Frames from Videos

```python
# Place videos in Dataset/real/ and Dataset/fake/
# Run frame extraction (code not shown, but extracts frames at regular intervals)
```

### Step 2: Face Detection & Extraction

```bash
python face_extraction.py
```

- Uses MTCNN for face detection
- Crops and resizes faces to 224x224
- Saves metadata with filenames and labels
- **Progress tracking**: Resumes from last checkpoint

### Step 3: (Optional) CLAHE Enhancement

```bash
python clahe_enhancer.py
```

- Applies Contrast Limited Adaptive Histogram Equalization
- Enhances facial features for better detection
- Improves model performance on low-quality videos

### Step 4: Dataset Splitting

```bash
python dataset_processor.py
```

- Splits data into 70% train, 15% val, 15% test
- Ensures class balance
- Creates metadata CSV files
- **Resumable**: Continues from saved progress

### Step 5: Create Sequence Dataset

```bash
python sequence_dataset.py
```

- Groups frames by video ID
- Creates 10-frame sequences for LSTM
- Handles variable-length videos

## 🎓 Training

### Model 1: ResNet50

Open and run [model1_resnet50.ipynb](model1_resnet50.ipynb)

- Transfer learning from ImageNet
- Fine-tuning on deepfake dataset
- Early stopping and learning rate scheduling

### Model 2: CNN+LSTM (Recommended)

Open and run [model2_cnn_lstm.ipynb](model2_cnn_lstm.ipynb)

- Sequential frame analysis
- Temporal feature learning
- Better generalization on unseen videos

**Training features:**

- GPU acceleration (CUDA)
- Data augmentation (random flips, rotations, color jitter)
- Class weighting for imbalanced datasets
- Checkpointing and progress saving
- Comprehensive logging to CSV

## 🎯 Inference

### Web Application

```bash
streamlit run app.py
```

**Features:**

1. Upload video file (MP4, AVI, MOV)
2. Automatic face detection and extraction
3. Real-time prediction with confidence score
4. Visual feedback with color-coded results

### Command-line Inference

```python
from app import load_model_and_detector, extract_faces
import torch

model, detector, device = load_model_and_detector()
faces = extract_faces("path/to/video.mp4")

if faces is not None:
    with torch.no_grad():
        output = model(faces.unsqueeze(0).to(device))
        pred = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred].item()

    print(f"Prediction: {'FAKE' if pred == 0 else 'REAL'}")
    print(f"Confidence: {confidence:.2%}")
```

## 📈 Model Performance

### CNN+LSTM Model

- **Accuracy**: ~95%+ on test set
- **ROC-AUC**: ~0.98
- **Inference Time**: ~2-3 seconds per video (10 frames)

### Evaluation Metrics

Run [Model2 Evaluation.ipynb](Model2%20Evaluation.ipynb) for:

- Confusion Matrix
- ROC Curve
- Precision, Recall, F1-Score
- Per-class metrics

## 🔧 Configuration

### Face Extraction Settings

```python
# face_extraction.py
REQUIRED_SIZE = (224, 224)  # Face image size
```

### Dataset Split Ratios

```python
# dataset_processor.py
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
```

### Model Hyperparameters

```python
# model2_cnn_lstm.ipynb
SEQ_LEN = 10              # Frames per sequence
HIDDEN_DIM = 256          # LSTM hidden dimension
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
NUM_EPOCHS = 20
```

## 📝 Progress Tracking

All processing scripts support pause/resume functionality through JSON progress files:

- `face_extraction_progress.json`
- `dataset_processing_progress.json`
- `clahe_enhancement_progress.json`
- `sequence_processing_progress_*.json`

**Benefits:**

- Resume long-running tasks after interruptions
- Track processing statistics
- Avoid reprocessing completed data

## 🛠️ Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size in training script
BATCH_SIZE = 4  # or 2
```

### MTCNN Installation Issues

```bash
pip install mtcnn --no-deps
pip install tensorflow opencv-python
```

### Slow Video Processing

- Reduce sequence length from 10 to 5 frames
- Use GPU for MTCNN if available
- Process videos in batches

## 🔬 Technical Details

### Face Detection

- **Method**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Advantages**: High accuracy, detects multiple faces
- **Output**: Bounding box coordinates + facial landmarks

### Data Augmentation

```python
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])
```

### Loss Function & Optimizer

- **Loss**: CrossEntropyLoss with class weights
- **Optimizer**: Adam with weight decay
- **LR Scheduler**: ReduceLROnPlateau

## 📚 References

- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- LSTM: [Long Short-Term Memory Networks](https://www.bioinf.jku.at/publications/older/2604.pdf)
- MTCNN: [Joint Face Detection and Alignment](https://arxiv.org/abs/1604.02878)
- CLAHE: [Adaptive Histogram Equalization](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 👥 Authors

- PRATIK MARUTI BHOSALE - [GitHub](https://github.com/PratikBhosale-07)

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- MTCNN authors for robust face detection
- Streamlit for the intuitive web framework
- Open-source community for various tools and libraries


**⚠️ Disclaimer**: This tool is for educational and research purposes only. Deepfake detection is an evolving field, and no detection system is 100% accurate. Always verify critical content through multiple sources.

**🌟 Star this repo if you find it useful!**
