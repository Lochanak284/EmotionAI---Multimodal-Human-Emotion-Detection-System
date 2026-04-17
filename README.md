# EmotionAI — Multimodal Human Emotion Detection System

## Overview
EmotionAI is a full-stack web application that detects human emotions using three 
input modalities — face, voice, and text — processed by three independently trained 
deep learning models. The predictions are combined using a confidence-based dynamic 
weighted fusion engine to produce a final emotion label along with secondary 
analytical insights including stress level, engagement score, and an empathy 
response suggestion.

## Tech Stach

## Backend
Framework: FastAPI(Python)
Face Model: Custom CNN (PyTorch)
Voice Model: Custom CNN (PyTorch)
Text Model: DistilBERT (Hugging Face Transformers)
Audio processing: librosa, soundfile
Image Processiong: PIL (Pillow), torchvision
Server: Uvicorn

## Frontend
HTML5, CSS3, Vanilla Javascript.

### Training [ Google Colab ]

NOTE: To increase the training efficiency, model is trained in T4 GPU using Google Colab. The training file used is also kept in this 
folder. [emotion_detection_system.ipynb file] 

Framework: PyTorch
Text Model: HuggingFace Transformers
Audio Features: librosa (MFCC)
Visualization: matplotlib, seaborn
Experiment UI: Gradio

## Project Structure
emotion_detection/
│
├── backend/
│   ├── app.py                  
│   ├── requirements.txt       
│   ├── .env                   
│   └── models/
│       ├── face_cnn_best.pth          
│       ├── voice_cnn_best.pth          
│       └── text_distilbert_best/       
│           ├── config.json
│           ├── model.safetensors
│           ├── tokenizer.json
│           └── tokenizer_config.json
│
├── frontend/
│   ├── index.html              
│   ├── style.css               
│   └── script.js               
│
└── README.md

### Datasets Used
Face = FER2013
Voice = RAVDESS [1400 audio files from 24 professional actors]
Text = ISEAR [ taken from HuggingFace NLP]

### Model Architecture

### Face CNN
- Input: 64×64×3 RGB image
- 4 Convolutional blocks (32 → 64 → 128 → 256 filters)
- Each block: Conv2D + BatchNorm + ReLU + MaxPool
- Classifier: Dropout(0.5) → FC(512) → ReLU → Dropout(0.3) → FC(7)
- Output: 7-class softmax probability vector

### Voice CNN
- Input: MFCC spectrogram — 1×40×174 (single channel)
- 3 Convolutional blocks with BatchNorm + ReLU
- AdaptiveAvgPool to fixed 4×4 output
- Classifier: Dropout(0.5) → FC(256) → ReLU → Dropout(0.3) → FC(7)
- Output: 7-class softmax probability vector

### Fusion Engine
- Late Fusion Technology
- Strategy: Confidence-based dynamic weighted averaging
- Weight of each modality = its maximum confidence score
- Formula: w_k = max(prob_k) / Σ max(prob_m)
- Final: P_fused = Σ w_k × prob_k
- Missing modalities are excluded and weights renormalized

### How to run the project file

Install Python with version higher than 3.10
Install ffmpeg and add that to PATH

download and extract the zip file

install all the requirements [ provided in requirement.txt]

in terminal, run these 2 commands: 
1. cd backend
2. python -m uvicorn app:app --reload --port 8000

### wait for this output before opening the browser:
 Loading models from: .../backend/models
✅ Face CNN loaded!
✅ Voice CNN loaded!
✅ DistilBERT loaded!
🚀 All models ready!
INFO: Uvicorn running on http://127.0.0.1:8000

### open in browser: http://127.0.0.1:8000

### How to Use

### Live Analysis Mode
1. Click **Start Camera** to enable webcam face detection
2. Click **Start Recording** and speak naturally for up to 20 seconds
3. Click **Stop Recording** when done
4. Type any text in the text box (optional)
5. Click **Analyze Emotion**
6. View results — emotion, confidence, stress, engagement, empathy response

### Upload Mode
1. Click the **Upload Files** tab
2. Upload a face image (JPG or PNG)
3. Upload a voice recording (OGG, WAV, or MP3)
4. Type any text (optional)
5. Click **Analyze Emotion**
6. View results

### At least one parameter must be provided.

### Files Not Included in GitHub Repository
Large model files (.safetensors, .pt, .h5, .bin) are not added due to GitHub size limitations (100MB)
backend/models/ folder is excluded to avoid uploading heavy trained models
Environment files (.env, backend/.env) are not included to protect sensitive data like API keys
Any secret or configuration file containing credentials is excluded for security reasons

### Known Limitations
 Voice needs expressive speech 
 Glass weared images misclassified as fear
No GPU at inference. Runs on CPU only. Inference takes 1–3 seconds per request 
English only. Models trained on English data. Non-English speech and text not supported 
Chrome/Edge only for captions. Firefox does not support real-time captions |


## Future Scope
- Replace Voice CNN with Wav2Vec 2.0 or HuBERT for improved audio accuracy
- Train face model on diverse dataset including glasses and mask wearers
- Session-level reporting for teachers showing class engagement statistics
- Cross-modal transformer fusion for better conflict resolution
- Mobile and edge device deployment through model quantization
- Multilingual support for non-English inputs

## Authors
Team Lead: Lochana K - 1RUA24CSE0226
Team Members: 
1. Hema B L - 1RUA24CSE0161
2. M Nandini -  1RUA24CSE0230
3. Maithri G -  1RUA24CSE0236

## Project Guide
Dr. Karthikeyan Periyasami
Associate Professor
School of Computer Science and Engineering. 
RV University, Bengaluru.

## License

This project was developed for academic purposes as part of the 2nd year Machine Learning Course.
All rights reserved by the authors and the institution.




