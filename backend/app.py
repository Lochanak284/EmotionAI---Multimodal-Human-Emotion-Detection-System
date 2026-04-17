from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from PIL import Image
from torchvision import transforms
import numpy as np
import soundfile as sf
import librosa, io, tempfile, os
import anthropic

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

DEVICE = torch.device('cpu')
UNIFIED_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_EMOJI    = {'angry':'😠','disgust':'🤢','fear':'😨','happy':'😊','neutral':'😐','sad':'😢','surprise':'😲'}

STRESS_WEIGHTS     = {'angry':0.85,'fear':0.90,'disgust':0.60,'sad':0.50,'surprise':0.30,'neutral':0.05,'happy':0.00}
ENGAGEMENT_WEIGHTS = {'happy':0.90,'surprise':0.85,'angry':0.70,'fear':0.60,'disgust':0.50,'sad':0.20,'neutral':0.00}

def compute_stress(fused_probs):
    weights = np.array([STRESS_WEIGHTS[e] for e in UNIFIED_EMOTIONS])
    return round(float(min(np.dot(fused_probs, weights), 1.0)), 3)

def compute_engagement(fused_probs):
    weights = np.array([ENGAGEMENT_WEIGHTS[e] for e in UNIFIED_EMOTIONS])
    return round(float(min(np.dot(fused_probs, weights), 1.0)), 3)

# ── 1. Face CNN ──
class FaceEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ── 2. Voice CNN ──
class VoiceEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, (3,3), padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, (3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, (3,3), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128*4*4, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

print(f"📂 Loading models from: {MODELS_DIR}")

face_model = FaceEmotionCNN().to(DEVICE)
face_model.load_state_dict(torch.load(
    os.path.join(MODELS_DIR, "face_cnn_best.pth"), map_location=DEVICE))
face_model.eval()
print("✅ Face CNN loaded!")

voice_model = VoiceEmotionCNN().to(DEVICE)
voice_model.load_state_dict(torch.load(
    os.path.join(MODELS_DIR, "voice_cnn_best.pth"), map_location=DEVICE))
voice_model.eval()
print("✅ Voice CNN loaded!")

tokenizer = DistilBertTokenizer.from_pretrained(
    os.path.join(MODELS_DIR, "text_distilbert_best"))
text_model = DistilBertForSequenceClassification.from_pretrained(
    os.path.join(MODELS_DIR, "text_distilbert_best")).to(DEVICE)
text_model.eval()
print("✅ DistilBERT loaded!")

print("🚀 All models ready!")

face_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def infer_face(pil_img):
    img = face_transform(pil_img.convert('RGB')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return F.softmax(face_model(img), dim=1).cpu().numpy()[0]

def infer_voice(audio_bytes):
    import subprocess

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
        f.write(audio_bytes)
        tmp_in = f.name

    tmp_wav = tmp_in + ".wav"

    try:
        print(f"🎤 Audio received: {len(audio_bytes)} bytes")

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in, "-ar", "22050", "-ac", "1", tmp_wav],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"❌ ffmpeg error: {result.stderr[-300:]}")
            return None

        y, sr = sf.read(tmp_wav, dtype='float32')

        if y.ndim > 1:
            y = y.mean(axis=1)

        print(f"✅ Voice decoded: {len(y)/sr:.1f}s at {sr}Hz")

        if len(y) < 1000:
            print("⚠️ Audio too short")
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = np.pad(mfcc, ((0,0),(0, max(0, 174-mfcc.shape[1]))), mode='constant')[:, :174]
        inp  = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = F.softmax(voice_model(inp), dim=1).cpu().numpy()[0]
            print(f"✅ Voice result: {UNIFIED_EMOTIONS[int(np.argmax(probs))]} ({float(np.max(probs))*100:.1f}%)")
            return probs

    except Exception as e:
        print(f"❌ Voice exception: {e}")
        return None

    finally:
        for f in [tmp_in, tmp_wav]:
            if os.path.exists(f):
                os.unlink(f)

def infer_text(text):
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding='max_length')
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        return F.softmax(text_model(**enc).logits, dim=1).cpu().numpy()[0]

# ── Dynamic confidence-based fusion ──
def fuse(face_p, voice_p, text_p):
    BASE = {'face': 0.20, 'voice': 0.35, 'text': 0.45}

    active = {
        k: v for k, v in [('face', face_p), ('voice', voice_p), ('text', text_p)]
        if v is not None
    }
    if not active:
        return np.ones(7) / 7

    total = sum(BASE[k] for k in active)
    result = np.zeros(7)
    for k, probs in active.items():
        result += (BASE[k] / total) * probs

    return result

def detect_conflict(face_p, voice_p, text_p):
    active = {
        k: v for k, v in [('face', face_p), ('voice', voice_p), ('text', text_p)]
        if v is not None
    }
    if len(active) < 2:
        return False
    top_emotions = [UNIFIED_EMOTIONS[int(np.argmax(p))] for p in active.values()]
    return len(set(top_emotions)) > 1

def get_empathy_response(emotion: str, stress_score: float, engagement_score: float, user_text: str = "") -> str:
    stress_pct  = round(stress_score * 100)
    engage_pct  = round(engagement_score * 100)

    responses = {
        'happy': (
            "This person is in a great mood — match their energy with warmth and enthusiasm! "
            "Celebrate the moment with them and keep the positive vibe going.",
            "Share in their joy genuinely — ask what's making them happy today."
        ),
        'sad': (
            "This person seems to be going through a tough time and needs to feel heard. "
            "Approach them gently, without rushing to fix things — just being present matters most.",
            "Start with 'I'm here for you' — don't jump to solutions right away."
        ),
        'angry': (
            "This person is frustrated and needs to feel acknowledged before anything else. "
            "Stay calm, don't get defensive, and give them space to express themselves.",
            "Say 'I understand why you feel that way' — validation reduces anger faster than explanations."
        ),
        'fear': (
            "This person feels anxious or scared and needs reassurance right now. "
            "Speak softly, stay steady, and remind them they are not alone in this.",
            "Offer concrete reassurance — vague comfort doesn't help anxiety, specifics do."
        ),
        'disgust': (
            "This person is uncomfortable or unsettled about something. "
            "Listen without judgment and avoid pushing them to explain more than they want to.",
            "Give them space and don't minimize what they're feeling — just acknowledge it."
        ),
        'surprise': (
            "This person just experienced something unexpected — give them a moment to process. "
            "Be patient and let them lead the conversation at their own pace.",
            "Ask open questions like 'How are you feeling about that?' rather than assuming."
        ),
        'neutral': (
            "This person seems calm and composed right now. "
            "A relaxed, friendly tone works best — no need to escalate emotionally.",
            "Keep the conversation light and open — they may be in a listening mood."
        ),
    }

    main, tip = responses.get(emotion, responses['neutral'])

    stress_note = ""
    if stress_pct >= 75:
        stress_note = f" Their stress is very high ({stress_pct}%) — be extra gentle and patient."
    elif stress_pct >= 45:
        stress_note = f" Moderate stress detected ({stress_pct}%) — keep your tone calm and steady."

    engage_note = ""
    if engage_pct < 30:
        engage_note = " They seem disengaged — try asking a simple open-ended question to draw them in."
    elif engage_pct >= 75:
        engage_note = " They are highly engaged — this is a good moment for meaningful conversation."

    print(f"✅ Empathy response generated for emotion: {emotion} (stress:{stress_pct}% engage:{engage_pct}%)")
    return f"{main}{stress_note}{engage_note}\nTIP: {tip}"

@app.post("/predict")
async def predict(
    text:  str        = Form(default=""),
    face:  UploadFile = File(default=None),
    voice: UploadFile = File(default=None)
):
    face_p = voice_p = text_p = None

    if face and face.filename:
        print(f"📷 Face received: {face.filename}")
        try:
            face_p = infer_face(Image.open(io.BytesIO(await face.read())))
            print(f"✅ Face inference: {UNIFIED_EMOTIONS[int(np.argmax(face_p))]} ({float(np.max(face_p))*100:.1f}%)")
        except Exception as e:
            print(f"❌ Face inference failed: {e}")

    if voice and voice.filename:
        voice_bytes = await voice.read()
        print(f"🎤 Voice received: {voice.filename}, size: {len(voice_bytes)} bytes")
        if len(voice_bytes) > 500:
            voice_p = infer_voice(voice_bytes)
        else:
            print("⚠️ Voice file too small, skipping")

    if text.strip():
        print(f"💬 Text received: {text[:60]}")
        try:
            text_p = infer_text(text.strip())
            print(f"✅ Text inference: {UNIFIED_EMOTIONS[int(np.argmax(text_p))]} ({float(np.max(text_p))*100:.1f}%)")
        except Exception as e:
            print(f"❌ Text inference failed: {e}")

    fused   = fuse(face_p, voice_p, text_p)
    idx     = int(np.argmax(fused))
    emotion = UNIFIED_EMOTIONS[idx]

    confidence        = float(fused[idx])
    stress_score      = compute_stress(fused)
    engagement_score  = compute_engagement(fused)
    is_conflicted     = detect_conflict(face_p, voice_p, text_p)
    is_low_confidence = confidence < 0.55

    empathy_response = get_empathy_response(emotion, stress_score, engagement_score, text)

    print(f"\n{'='*40}")
    print(f"🧠 FUSION RESULT: {emotion} ({confidence*100:.1f}%)")
    print(f"   Conflicted: {is_conflicted} | Low confidence: {is_low_confidence}")
    print(f"   Stress: {stress_score} | Engagement: {engagement_score}")
    print(f"{'='*40}\n")

    return {
        "emotion":            emotion,
        "emoji":              EMOTION_EMOJI[emotion],
        "confidence":         confidence,
        "is_conflicted":      is_conflicted,
        "is_low_confidence":  is_low_confidence,
        "stress_score":       stress_score,
        "engagement_score":   engagement_score,
        "empathy_response":   empathy_response,
        "probabilities":      dict(zip(UNIFIED_EMOTIONS, fused.tolist())),
        "modalities": {
            "face":  {"emotion": UNIFIED_EMOTIONS[int(np.argmax(face_p))],  "confidence": float(np.max(face_p))}  if face_p  is not None else None,
            "voice": {"emotion": UNIFIED_EMOTIONS[int(np.argmax(voice_p))], "confidence": float(np.max(voice_p))} if voice_p is not None else None,
            "text":  {"emotion": UNIFIED_EMOTIONS[int(np.argmax(text_p))],  "confidence": float(np.max(text_p))}  if text_p  is not None else None,
        }
    }

app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "..", "frontend"), html=True), name="static")