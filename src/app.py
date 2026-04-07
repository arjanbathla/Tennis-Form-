"""
TennisForm — Streamlit dashboard for stroke analysis.
Custom encoder layer for guaranteed attention weight extraction.
"""

import streamlit as st
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import cv2
import mediapipe as mp_lib
import tempfile
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dtw import dtw
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math
import warnings
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
MODEL_DIR = PROJECT_ROOT / "models"

mp_pose = mp_lib.solutions.pose
mp_drawing = mp_lib.solutions.drawing_utils
mp_drawing_styles = mp_lib.solutions.drawing_styles

JOINT_NAMES = ["left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"]
JOINT_DISPLAY = {"left_shoulder":"Left Shoulder","right_shoulder":"Right Shoulder",
    "left_elbow":"Left Elbow","right_elbow":"Right Elbow",
    "left_wrist":"Left Wrist","right_wrist":"Right Wrist",
    "left_hip":"Left Hip","right_hip":"Right Hip",
    "left_knee":"Left Knee","right_knee":"Right Knee",
    "left_ankle":"Left Ankle","right_ankle":"Right Ankle"}
SELECTED_JOINTS = [11,12,13,14,15,16,23,24,25,26,27,28]
LEFT_HIP,RIGHT_HIP = 23,24
LEFT_SHOULDER,RIGHT_SHOULDER = 11,12
SMOOTHING_WINDOW = 3
MAX_SEQ_LENGTH = 200
INPUT_DIM = 24
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 3
EMBEDDING_SIZE = 64
DROPOUT = 0.1
MARGIN = 1.0

JOINT_FEEDBACK = {
    "right_wrist": {"body_part":"Racket-Hand Wrist",
        "good":"Your racket-hand wrist movement closely matches professional technique. Good control through the swing.",
        "moderate":"Your wrist path is slightly different from the expert. Focus on keeping your wrist firm through contact and letting it naturally roll over during follow-through.",
        "poor":"Your wrist movement differs significantly from expert technique. This often means the racket face angle is inconsistent. Practice shadow swings focusing on a smooth, controlled wrist through the contact zone."},
    "left_wrist": {"body_part":"Non-Racket Wrist",
        "good":"Good use of your non-racket arm for balance and momentum.",
        "moderate":"Your non-racket arm could be more active. Try pointing it toward the ball during preparation.",
        "poor":"Your non-racket arm is not contributing to the stroke. Practice pointing your free hand at the incoming ball."},
    "right_elbow": {"body_part":"Racket-Arm Elbow",
        "good":"Your elbow position through the swing matches expert form well.",
        "moderate":"Your elbow is slightly out of position. Keep it slightly bent and close to your body during the forward swing.",
        "poor":"Your elbow position differs a lot from experts. Practice keeping your elbow tucked and leading the swing with your forearm."},
    "left_elbow": {"body_part":"Non-Racket Elbow",
        "good":"Good positioning of your non-racket elbow, helping with balance.",
        "moderate":"Your non-racket elbow could help more with balance. Let it extend naturally during preparation.",
        "poor":"Your non-racket arm is too passive. Extend it during the backswing to help rotate your shoulders."},
    "right_shoulder": {"body_part":"Racket-Side Shoulder",
        "good":"Excellent shoulder rotation matching professional technique.",
        "moderate":"Your shoulder rotation is slightly limited. Focus on turning your shoulders more during preparation.",
        "poor":"Insufficient shoulder rotation is limiting your power. Practice turning your hitting shoulder back further."},
    "left_shoulder": {"body_part":"Non-Racket Shoulder",
        "good":"Good shoulder alignment and rotation through the stroke.",
        "moderate":"Your non-racket shoulder could lead the rotation more.",
        "poor":"Your shoulders are not rotating enough. The front shoulder should point toward the ball during preparation."},
    "right_hip": {"body_part":"Right Hip",
        "good":"Strong hip rotation matching expert technique.",
        "moderate":"Your hip rotation is slightly limited. Initiate the forward swing with your hips before your arm.",
        "poor":"Your hips are too static. Practice rotating your hips toward the target before your arm swings."},
    "left_hip": {"body_part":"Left Hip",
        "good":"Good hip positioning and weight transfer.",
        "moderate":"Focus on transferring weight from back foot to front foot during the swing.",
        "poor":"Your hips are not involved enough. Practice stepping into the shot and rotating your hips."},
    "right_knee": {"body_part":"Right Knee",
        "good":"Good knee bend and leg drive.",
        "moderate":"Slightly more knee bend would help. Staying lower gives better balance.",
        "poor":"Your legs are too straight. Bend your knees more and push up through the shot."},
    "left_knee": {"body_part":"Left Knee",
        "good":"Good knee positioning supporting your balance.",
        "moderate":"Try bending your front knee slightly as you step into the shot.",
        "poor":"Your front leg is too rigid. A slight bend helps absorb impact and keeps you balanced."},
    "right_ankle": {"body_part":"Right Ankle",
        "good":"Good footwork and ankle positioning.",
        "moderate":"Stay on the balls of your feet, not flat-footed.",
        "poor":"Your footwork needs attention. Stay on the balls of your feet and practice adjustment steps."},
    "left_ankle": {"body_part":"Left Ankle",
        "good":"Good foot placement and weight transfer.",
        "moderate":"Step toward the target with your front foot more consistently.",
        "poor":"Your front foot is not stepping into the shot. Practice stepping forward as you swing."},
}


# must match siamese_transformer.py exactly
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.last_attn_weights = None

    def forward(self, src, src_key_padding_mask=None):
        attn_output, attn_weights = self.self_attn(
            src, src, src, key_padding_mask=src_key_padding_mask,
            need_weights=True, average_attn_weights=True)
        self.last_attn_weights = attn_weights.detach().cpu()
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)
        return src

class StrokeTransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(INPUT_DIM, EMBED_DIM)
        self.pos_encoder = PositionalEncoding(EMBED_DIM, max_len=MAX_SEQ_LENGTH + 10)
        self.layers = nn.ModuleList([
            CustomEncoderLayer(EMBED_DIM, NUM_HEADS, EMBED_DIM * 4, DROPOUT)
            for _ in range(NUM_LAYERS)])
        self.output_projection = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBEDDING_SIZE), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE))

    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        if mask is not None:
            m = (~mask).unsqueeze(-1).float()
            x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return self.output_projection(x)

    def get_attention_weights(self):
        return self.layers[-1].last_attn_weights

class SiameseTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StrokeTransformerEncoder()
    def forward(self, x1, x2, mask1=None, mask2=None):
        e1, e2 = self.encoder(x1, mask1), self.encoder(x2, mask2)
        return torch.sqrt(torch.sum((e1 - e2) ** 2, dim=1) + 1e-8), e1, e2
    def get_embedding(self, x, mask=None):
        return self.encoder(x, mask)


def process_uploaded_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
    all_kp, detected = [], 0
    with mp_pose.Pose(static_image_mode=False, model_complexity=2,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                all_kp.append([[lm.x, lm.y] for lm in res.pose_landmarks.landmark])
                detected += 1
            else: all_kp.append([[0.0,0.0]]*33)
            out.write(frame)
    cap.release(); out.release()
    if not all_kp: return None, None, None
    return np.array(all_kp), out_path, {"fps":fps,"total_frames":total,"detected_frames":detected,
        "detection_rate":round(detected/total*100,1) if total>0 else 0}

def preprocess_keypoints(kp):
    sums = np.sum(np.abs(kp), axis=(1,2))
    kp = kp[sums > 0]
    if len(kp) < 5: return None
    hm = (kp[:,LEFT_HIP,:] + kp[:,RIGHT_HIP,:]) / 2.0
    kp = kp - hm[:,np.newaxis,:]
    sm = (kp[:,LEFT_SHOULDER,:] + kp[:,RIGHT_SHOULDER,:]) / 2.0
    hm2 = (kp[:,LEFT_HIP,:] + kp[:,RIGHT_HIP,:]) / 2.0
    torso = np.mean(np.linalg.norm(sm - hm2, axis=1))
    if torso < 0.01: return None
    kp = kp / torso
    kp = kp[:, SELECTED_JOINTS, :]
    smoothed = np.copy(kp).astype(float)
    half = SMOOTHING_WINDOW // 2
    for i in range(len(kp)):
        s, e = max(0, i-half), min(len(kp), i+half+1)
        smoothed[i] = np.mean(kp[s:e], axis=0)
    return smoothed

def compare_against_experts(prep, stroke_type):
    folder = PREPROCESSED_DIR / "thetis" / stroke_type
    if not folder.exists(): return None, None
    files = sorted([f for f in folder.iterdir() if f.name.endswith("_preprocessed.npy")])
    if len(files) > 15: files = [files[i] for i in np.random.choice(len(files), 15, replace=False)]
    od, pj = [], {j:[] for j in JOINT_NAMES}
    fp = prep.reshape(len(prep), -1)
    for ef in files:
        exp = np.load(str(ef))
        fe = exp.reshape(len(exp), -1)
        od.append(dtw(fp, fe, keep_internals=False).normalizedDistance)
        for j, jn in enumerate(JOINT_NAMES):
            pj[jn].append(dtw(prep[:,j,:], exp[:,j,:], keep_internals=False).normalizedDistance)
    ao = np.mean(od)
    ap = {j: np.mean(d) for j, d in pj.items()}
    return {"overall_distance":round(float(ao),4), "similarity_score":round(100*np.exp(-ao),1),
            "per_joint_distances":{k:round(float(v),4) for k,v in ap.items()}}, ap

def pad_single(seq):
    flat = seq.reshape(len(seq),-1)
    sl = len(flat)
    if sl >= MAX_SEQ_LENGTH:
        return torch.FloatTensor(flat[:MAX_SEQ_LENGTH]).unsqueeze(0), torch.BoolTensor([False]*MAX_SEQ_LENGTH).unsqueeze(0)
    pad = np.zeros((MAX_SEQ_LENGTH-sl, flat.shape[1]))
    return torch.FloatTensor(np.vstack([flat,pad])).unsqueeze(0), torch.BoolTensor([False]*sl+[True]*(MAX_SEQ_LENGTH-sl)).unsqueeze(0)

@st.cache_resource
def load_siamese_model():
    mp = MODEL_DIR / "siamese_transformer_best.pth"
    if not mp.exists(): return None
    model = SiameseTransformer()
    model.load_state_dict(torch.load(str(mp), map_location="cpu", weights_only=True))
    model.eval()
    return model

def siamese_compare(model, prep, stroke_type):
    folder = PREPROCESSED_DIR / "thetis" / stroke_type
    if not folder.exists() or model is None: return None, None
    files = sorted([f for f in folder.iterdir() if f.name.endswith("_preprocessed.npy")])
    if len(files) > 10: files = [files[i] for i in np.random.choice(len(files), 10, replace=False)]
    dists = []
    pp, pm = pad_single(prep)
    for ef in files:
        ep, em = pad_single(np.load(str(ef)))
        with torch.no_grad():
            d, _, _ = model(pp, ep, pm, em)
            dists.append(d.item())
    avg = np.mean(dists)
    sim = round(100*np.exp(-avg), 1)

    attention = None
    with torch.no_grad():
        _ = model.get_embedding(pp, pm)
    attn = model.encoder.get_attention_weights()
    if attn is not None:
        an = attn[0].numpy()
        sl = int((~pm[0]).sum())
        fi = an[:sl,:sl].sum(axis=0)
        fi = fi / fi.max()
        attention = fi

    return {"siamese_similarity":sim, "siamese_distance":round(float(avg),4)}, attention

def compute_joint_angle(a,b,c):
    ba, bc = a-b, c-b
    cos = np.sum(ba*bc,axis=-1)/(np.linalg.norm(ba,axis=-1)*np.linalg.norm(bc,axis=-1)+1e-8)
    return np.degrees(np.arccos(np.clip(cos,-1,1)))

def extract_features(kp):
    f = [len(kp)]
    for j in range(12):
        jd = kp[:,j,:]
        f.extend([np.mean(jd[:,0]),np.mean(jd[:,1]),np.std(jd[:,0]),np.std(jd[:,1]),
                  np.ptp(jd[:,0]),np.ptp(jd[:,1]),np.min(jd[:,0]),np.max(jd[:,0]),np.min(jd[:,1]),np.max(jd[:,1])])
    for j in range(12):
        jd = kp[:,j,:]
        if len(jd)>1:
            v = np.linalg.norm(np.diff(jd,axis=0),axis=1)
            f.extend([np.mean(v),np.max(v),np.std(v)])
        else: f.extend([0,0,0])
    for a,b,c in [(0,2,4),(1,3,5),(6,8,10),(7,9,11),(2,0,6),(3,1,7)]:
        ang = compute_joint_angle(kp[:,a,:],kp[:,b,:],kp[:,c,:])
        f.extend([np.mean(ang),np.std(ang),np.min(ang),np.max(ang),np.ptp(ang)])
    if len(kp)>1:
        u = np.mean([np.mean(np.linalg.norm(np.diff(kp[:,j,:],axis=0),axis=1)) for j in range(6)])
        l = np.mean([np.mean(np.linalg.norm(np.diff(kp[:,j,:],axis=0),axis=1)) for j in range(6,12)])
        f.append(u/(l+1e-8))
        le = np.mean([np.mean(np.linalg.norm(np.diff(kp[:,j,:],axis=0),axis=1)) for j in [0,2,4,6,8,10]])
        ri = np.mean([np.mean(np.linalg.norm(np.diff(kp[:,j,:],axis=0),axis=1)) for j in [1,3,5,7,9,11]])
        f.append(le/(ri+1e-8))
    else: f.extend([0,0])
    return np.array(f)

@st.cache_resource
def train_ml():
    ef, bf = [], []
    for s in ["forehand","backhand"]:
        for lv, cont in [("thetis",ef),("thetis_beginners",bf)]:
            fd = PREPROCESSED_DIR/lv/s
            if not fd.exists(): continue
            for fl in fd.iterdir():
                if fl.name.endswith("_preprocessed.npy"): cont.append(extract_features(np.load(str(fl))))
    if not ef or not bf: return None
    X = np.nan_to_num(np.array(ef+bf))
    y = np.array(["expert"]*len(ef)+["beginner"]*len(bf))
    models = {}
    for n, c in [("Random Forest",RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,class_weight="balanced")),
                 ("SVM",SVC(kernel="rbf",C=1.0,probability=True,class_weight="balanced",random_state=42)),
                 ("KNN",KNeighborsClassifier(n_neighbors=5,weights="distance"))]:
        p = Pipeline([("scaler",StandardScaler()),("clf",c)]); p.fit(X,y); models[n] = p
    return models

def ml_predict(models, prep):
    if not models: return None
    f = np.nan_to_num(extract_features(prep).reshape(1,-1))
    return {n:{"prediction":p.predict(f)[0],
               "expert_probability":round(float(p.predict_proba(f)[0][np.where(p.classes_=="expert")[0][0]]),4)}
            for n,p in models.items()}

def generate_feedback(pj):
    items = []
    for jn,dist in sorted(pj.items(), key=lambda x:x[1], reverse=True):
        fb = JOINT_FEEDBACK.get(jn)
        if not fb: continue
        if dist<0.2: lv,ic = "good","✅"
        elif dist<0.4: lv,ic = "moderate","⚠️"
        else: lv,ic = "poor","🔴"
        items.append({"joint":jn,"body_part":fb["body_part"],"distance":dist,"level":lv,"icon":ic,"message":fb[lv]})
    return items

@st.cache_data
def load_json(fn):
    p = RESULTS_DIR/fn
    if p.exists():
        with open(str(p)) as f: return json.load(f)
    return None

def get_stroke_type(f): return "Forehand" if "forehand" in f else "Backhand" if "backhand" in f else "Unknown"
def get_ball_type(f): return "Shadow Swing" if "without_ball" in f else "With Ball" if "with_ball" in f else "Unknown"


def score_color(score):
    if score >= 40: return "#22C55E"
    elif score >= 25: return "#F59E0B"
    return "#EF4444"

def score_bg(score):
    if score >= 40: return "#DCFCE7"
    elif score >= 25: return "#FEF3C7"
    return "#FEE2E2"

def score_label(score):
    if score >= 40: return "Good"
    elif score >= 25: return "Fair"
    return "Needs Work"

def dist_color(d):
    if d < 0.2: return "#22C55E"
    elif d < 0.4: return "#F59E0B"
    return "#EF4444"

CHART_COLORS = {
    "primary": "#1E40AF",
    "secondary": "#3B82F6",
    "accent": "#F59E0B",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "muted": "#94A3B8",
}

def base_layout(title="", height=400, showlegend=True):
    return dict(
        title=dict(text=title, font=dict(size=15, color="#1E293B", family="Inter, sans-serif"), x=0, xanchor="left"),
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="#F8FAFC",
        font=dict(family="Inter, sans-serif", color="#334155", size=12),
        margin=dict(t=55 if title else 30, b=45, l=55, r=20),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=12)),
        xaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", tickfont=dict(size=11), zeroline=False),
        yaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", tickfont=dict(size=11), zeroline=False),
    )


st.set_page_config(page_title="TennisForm", page_icon="🎾", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap');

html, body, [data-testid="stApp"] {
    font-family: 'Inter', sans-serif;
    background-color: #F1F5F9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
    border-right: 1px solid #1E293B !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #94A3B8 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 4px 0 !important;
    transition: color 0.15s !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: #F1F5F9 !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown div,
section[data-testid="stSidebar"] p {
    color: #64748B !important;
}

/* Main container */
.main .block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white !important;
    border-radius: 14px !important;
    padding: 1.25rem !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricLabel"] > div {
    color: #64748B !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stMetricValue"] > div {
    color: #1E40AF !important;
    font-family: 'Fira Code', monospace !important;
    font-weight: 700 !important;
    font-size: 1.9rem !important;
}

/* Headings */
h1 {
    color: #0F172A !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    font-family: 'Inter', sans-serif !important;
}
h2 { color: #1E293B !important; font-weight: 700 !important; }
h3 { color: #334155 !important; font-weight: 600 !important; }

/* Primary button */
.stButton > button {
    background: linear-gradient(135deg, #1E40AF, #2563EB) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.5rem 1.5rem !important;
    transition: box-shadow 0.2s, transform 0.15s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 14px rgba(30, 64, 175, 0.45) !important;
    transform: translateY(-1px) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #F8FAFC !important;
    border: 2px dashed #CBD5E1 !important;
    border-radius: 14px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3B82F6 !important;
    background: #EFF6FF !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid #E2E8F0 !important;
}

/* Alerts */
[data-testid="stAlert"] { border-radius: 10px !important; }
.stInfo, .stSuccess, .stWarning, .stError { border-radius: 10px !important; }

/* Expander */
details > summary {
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Divider */
hr { border-color: #E2E8F0 !important; margin: 1.5rem 0 !important; }

/* Radio buttons in main area */
.stRadio > label {
    font-weight: 600 !important;
    color: #334155 !important;
}

/* Spinner text */
.stSpinner p { color: #64748B !important; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    border-radius: 8px !important;
    border-color: #CBD5E1 !important;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.75rem 1rem 1.25rem; border-bottom: 1px solid #1E293B; margin-bottom: 0.75rem;">
        <div style="font-size: 1.65rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.5px; font-family: 'Inter', sans-serif; line-height: 1;">
            TennisForm
        </div>
        <div style="font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 2.5px; margin-top: 6px; font-family: 'Inter', sans-serif;">
            AI Stroke Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Analyse My Stroke", "DTW Analysis", "ML Classification",
         "Siamese Transformer", "Method Comparison", "Individual Stroke"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="margin-top: 2.5rem; padding: 0.875rem 1rem; background: rgba(255,255,255,0.04);
         border-radius: 10px; border: 1px solid #1E293B;">
        <div style="font-size: 11px; color: #475569; font-family: 'Inter', sans-serif; line-height: 1.6;">
            <span style="color: #64748B;">Final Year Project</span><br>
            <span style="color: #94A3B8; font-weight: 600;">Nanyang Technological University</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if page == "Overview":
    st.markdown("""
    <div style="margin-bottom: 1.75rem;">
        <h1 style="font-size: 2.1rem; font-weight: 800; color: #0F172A; margin: 0; letter-spacing: -0.5px;">
            TennisForm
        </h1>
        <p style="color: #64748B; font-size: 1rem; margin-top: 6px; margin-bottom: 0;">
            AI-powered tennis stroke analysis — compare your technique against professional players
        </p>
    </div>
    """, unsafe_allow_html=True)

    dd = load_json("dtw_comparison_results.json")
    sd = load_json("siamese_transformer_results.json")
    md = load_json("ml_classification_results.json")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Videos Analysed", "81" if dd else "N/A")
    with c2:
        if dd:
            st.metric("Avg DTW Similarity", f"{np.mean([r['similarity_score'] for r in dd]):.1f}%")
        else:
            st.metric("Avg DTW Similarity", "N/A")
    with c3:
        if sd:
            st.metric("Avg Siamese Similarity", f"{np.mean([r['siamese_similarity'] for r in sd]):.1f}%")
        else:
            st.metric("Avg Siamese Similarity", "N/A")
    with c4:
        if md:
            st.metric("Best ML Accuracy", f"{max(v['cv_accuracy_mean'] for v in md.values()):.1%}")
        else:
            st.metric("Best ML Accuracy", "N/A")

    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 1rem; font-weight: 700; color: #1E293B; margin-bottom: 1rem;
         padding-bottom: 0.5rem; border-bottom: 2px solid #1E40AF; display: inline-block;">
        Analysis Methods
    </div>
    """, unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("""
        <div style="background:white; border-radius:14px; padding:1.5rem; border:1px solid #E2E8F0;
             box-shadow:0 1px 3px rgba(0,0,0,0.07); height:100%;">
            <div style="width:42px;height:42px;background:#EFF6FF;border-radius:10px;
                 display:flex;align-items:center;justify-content:center;margin-bottom:1rem;">
                <svg width="20" height="20" fill="none" stroke="#1E40AF" stroke-width="2" stroke-linecap="round" viewBox="0 0 24 24">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
            </div>
            <div style="font-size:0.95rem; font-weight:700; color:#1E293B; margin-bottom:0.5rem;">DTW Analysis</div>
            <div style="font-size:13px; color:#64748B; line-height:1.65;">
                Dynamic Time Warping measures temporal shape similarity between your stroke
                and expert sequences, robust to differences in speed.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown("""
        <div style="background:white; border-radius:14px; padding:1.5rem; border:1px solid #E2E8F0;
             box-shadow:0 1px 3px rgba(0,0,0,0.07); height:100%;">
            <div style="width:42px;height:42px;background:#ECFDF5;border-radius:10px;
                 display:flex;align-items:center;justify-content:center;margin-bottom:1rem;">
                <svg width="20" height="20" fill="none" stroke="#16A34A" stroke-width="2" stroke-linecap="round" viewBox="0 0 24 24">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
                </svg>
            </div>
            <div style="font-size:0.95rem; font-weight:700; color:#1E293B; margin-bottom:0.5rem;">Siamese Transformer</div>
            <div style="font-size:13px; color:#64748B; line-height:1.65;">
                Deep learning model trained on contrastive pairs learns rich stroke embeddings
                with self-attention to identify the most critical frames.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with mc3:
        st.markdown("""
        <div style="background:white; border-radius:14px; padding:1.5rem; border:1px solid #E2E8F0;
             box-shadow:0 1px 3px rgba(0,0,0,0.07); height:100%;">
            <div style="width:42px;height:42px;background:#FFF7ED;border-radius:10px;
                 display:flex;align-items:center;justify-content:center;margin-bottom:1rem;">
                <svg width="20" height="20" fill="none" stroke="#EA580C" stroke-width="2" stroke-linecap="round" viewBox="0 0 24 24">
                    <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/>
                </svg>
            </div>
            <div style="font-size:0.95rem; font-weight:700; color:#1E293B; margin-bottom:0.5rem;">ML Classification</div>
            <div style="font-size:13px; color:#64748B; line-height:1.65;">
                Random Forest, SVM and KNN classifiers trained on engineered biomechanical
                features to classify expert vs beginner technique.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height: 1.25rem;"></div>', unsafe_allow_html=True)
    st.info("Upload your stroke video in **Analyse My Stroke** to get personalised AI-powered coaching feedback.")


elif page == "Analyse My Stroke":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2rem; font-weight: 800; color: #0F172A; margin: 0;">Analyse My Stroke</h1>
        <p style="color: #64748B; font-size: 0.95rem; margin-top: 6px; margin-bottom: 0;">
            Upload a video of your stroke and get instant AI-powered coaching feedback
        </p>
    </div>
    """, unsafe_allow_html=True)

    cu, ct = st.columns([2, 1])
    with ct:
        st.markdown("""
        <div style="background:white; border-radius:14px; padding:1.25rem; border:1px solid #E2E8F0;
             box-shadow:0 1px 3px rgba(0,0,0,0.07);">
            <div style="font-size:13px; font-weight:700; color:#1E293B; text-transform:uppercase;
                 letter-spacing:0.8px; margin-bottom:0.75rem;">Stroke Type</div>
        """, unsafe_allow_html=True)
        stroke_type = st.radio("What stroke is this?", ["forehand", "backhand"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#F0FDF4; border-radius:14px; padding:1.25rem; border:1px solid #BBF7D0; margin-top:0.75rem;">
            <div style="font-size:13px; font-weight:700; color:#16A34A; margin-bottom:0.5rem;">Recording Tips</div>
            <div style="font-size:12px; color:#374151; line-height:1.7;">
                Front-on camera angle<br>
                Full body visible in frame<br>
                Good consistent lighting<br>
                One complete stroke per video
            </div>
        </div>
        """, unsafe_allow_html=True)

    with cu:
        uploaded = st.file_uploader(
            "Upload your stroke video",
            type=["mov", "mp4", "avi"],
            help="MP4, MOV or AVI format"
        )

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.markdown("---")
        with st.spinner("Extracting pose landmarks..."):
            kp, overlay, info = process_uploaded_video(tmp_path)

        if kp is not None and overlay:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div style="font-size:13px; font-weight:700; color:#1E293B; margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:0.8px;">Your Video</div>', unsafe_allow_html=True)
                st.video(tmp_path)
            with c2:
                st.markdown('<div style="font-size:13px; font-weight:700; color:#1E293B; margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:0.8px;">Skeleton Overlay</div>', unsafe_allow_html=True)
                st.video(overlay)

            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Total Frames", info["total_frames"])
            with m2: st.metric("Pose Detected", f"{info['detection_rate']}%")
            with m3: st.metric("Frame Rate", f"{info['fps']:.0f} fps")

            st.markdown("---")
            prep = preprocess_keypoints(kp)

            if prep is not None:
                st.markdown(f"""
                <div style="display:inline-flex;align-items:center;gap:0.5rem;background:#EFF6FF;
                     border-radius:8px;padding:0.4rem 1rem;border:1px solid #BFDBFE;margin-bottom:1rem;">
                    <span style="font-size:13px;font-weight:700;color:#1E40AF;text-transform:uppercase;
                          letter-spacing:0.5px;">Stroke detected:</span>
                    <span style="font-size:14px;font-weight:600;color:#1E3A8A;">{stroke_type.title()}</span>
                </div>
                """, unsafe_allow_html=True)

                with st.spinner("Running DTW comparison against expert database..."):
                    dtw_r, pj = compare_against_experts(prep, stroke_type)

                sm = load_siamese_model()
                sr, attn = None, None
                if sm:
                    with st.spinner("Running Siamese Transformer inference..."):
                        sr, attn = siamese_compare(sm, prep, stroke_type)

                mlm = train_ml()
                mlr = ml_predict(mlm, prep) if mlm else None

                if dtw_r:
                    st.markdown("""
                    <div style="font-size:1rem; font-weight:700; color:#1E293B; margin-bottom:1rem;
                         padding-bottom:0.5rem; border-bottom:2px solid #1E40AF; display:inline-block;">
                        Similarity Scores
                    </div>
                    """, unsafe_allow_html=True)

                    s1, s2, s3 = st.columns(3)
                    ds = dtw_r["similarity_score"]
                    with s1:
                        st.metric("DTW Similarity", f"{ds}%")
                    with s2:
                        if sr:
                            ss = sr["siamese_similarity"]
                            st.metric("Siamese Similarity", f"{ss}%")
                        else:
                            st.metric("Siamese Similarity", "N/A")
                    with s3:
                        if mlr:
                            expert_votes = sum(1 for m in mlr.values() if m["prediction"] == "expert")
                            st.metric("ML Consensus", f"{expert_votes}/3 Expert")

                    scores, labels, colors = [ds], ["DTW"], [score_color(ds)]
                    if sr:
                        ss = sr["siamese_similarity"]
                        scores.insert(0, ss)
                        labels.insert(0, "Siamese")
                        colors.insert(0, score_color(ss))

                    fig = go.Figure()
                    for i, (lbl, val, col) in enumerate(zip(labels, scores, colors)):
                        fig.add_trace(go.Bar(
                            x=[val], y=[lbl], orientation="h",
                            name=f"{lbl}: {val}%",
                            marker=dict(color=col, line=dict(width=0)),
                            text=[f"  {val}% — {score_label(val)}"],
                            textposition="inside",
                            textfont=dict(size=13, color="white", family="Inter, sans-serif"),
                            showlegend=False,
                            hovertemplate=f"<b>{lbl}</b>: {val}%<extra></extra>",
                        ))
                        fig.add_trace(go.Bar(
                            x=[100 - val], y=[lbl], orientation="h",
                            marker=dict(color="#E2E8F0", line=dict(width=0)),
                            showlegend=False, hoverinfo="skip",
                        ))

                    layout = base_layout(title="Similarity to Expert Technique (%)", height=130 + 50 * len(scores))
                    layout.update(barmode="stack", xaxis=dict(range=[0, 100], ticksuffix="%",
                                  gridcolor="#E2E8F0", linecolor="#CBD5E1", tickfont=dict(size=11)),
                                  yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)"))
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)

                    if mlr:
                        st.markdown("""
                        <div style="font-size:13px; font-weight:700; color:#1E293B; margin:0.5rem 0;
                             text-transform:uppercase; letter-spacing:0.8px;">ML Classification Results</div>
                        """, unsafe_allow_html=True)
                        ml_df = pd.DataFrame([
                            {"Model": n,
                             "Prediction": r["prediction"].title(),
                             "Expert Probability": f"{r['expert_probability']:.1%}"}
                            for n, r in mlr.items()
                        ])
                        st.dataframe(ml_df, use_container_width=True, hide_index=True)

                    st.markdown("---")
                    st.markdown("""
                    <div style="font-size:1rem; font-weight:700; color:#1E293B; margin-bottom:1rem;
                         padding-bottom:0.5rem; border-bottom:2px solid #1E40AF; display:inline-block;">
                        Body Part Breakdown
                    </div>
                    """, unsafe_allow_html=True)

                    sj = sorted(pj.items(), key=lambda x: x[1], reverse=True)
                    jnames = [JOINT_DISPLAY[j[0]] for j in sj]
                    jvals = [j[1] for j in sj]
                    jcols = [dist_color(v) for v in jvals]

                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=jvals[::-1], y=jnames[::-1],
                        orientation="h",
                        marker=dict(color=jcols[::-1], line=dict(width=0)),
                        text=[f"  {'Close' if v < 0.2 else 'Needs Work' if v < 0.4 else 'Focus Area'} ({v:.3f})"
                              for v in jvals[::-1]],
                        textposition="outside",
                        textfont=dict(size=11, color="#334155"),
                        hovertemplate="<b>%{y}</b><br>Deviation: %{x:.3f}<extra></extra>",
                        showlegend=False,
                    ))
                    layout2 = base_layout(title="Deviation from Expert (lower is better)", height=420)
                    layout2.update(xaxis=dict(title="DTW Distance", range=[0, max(jvals) * 1.4],
                                   gridcolor="#E2E8F0"), margin=dict(l=130, r=20, t=55, b=45))
                    fig2.update_layout(**layout2)
                    st.plotly_chart(fig2, use_container_width=True)

                    if attn is not None:
                        st.markdown("---")
                        st.markdown("""
                        <div style="font-size:1rem; font-weight:700; color:#1E293B; margin-bottom:0.25rem;
                             padding-bottom:0.5rem; border-bottom:2px solid #1E40AF; display:inline-block;">
                            Transformer Attention
                        </div>
                        <p style="color:#64748B; font-size:13px; margin-top:0.5rem; margin-bottom:1rem;">
                            Which frames the Siamese Transformer focuses on when evaluating your stroke.
                            <strong style="color:#EF4444;">Red bars</strong> are the 5 most critical frames.
                        </p>
                        """, unsafe_allow_html=True)

                        top5 = np.argsort(attn)[-5:]
                        bar_colors = ["#EF4444" if i in top5 else "#3B82F6" for i in range(len(attn))]

                        fig3 = go.Figure()
                        fig3.add_trace(go.Bar(
                            x=list(range(len(attn))), y=attn,
                            marker=dict(color=bar_colors, line=dict(width=0)),
                            hovertemplate="Frame %{x}<br>Attention: %{y:.3f}<extra></extra>",
                            showlegend=False,
                        ))
                        layout3 = base_layout(title="Frame-Level Attention Weights", height=280)
                        layout3.update(xaxis=dict(title="Frame Number"), yaxis=dict(title="Attention Weight"))
                        fig3.update_layout(**layout3)
                        st.plotly_chart(fig3, use_container_width=True)

                        total_f = len(attn)
                        top_sorted = sorted(top5)
                        early = [f for f in top_sorted if f < total_f * 0.33]
                        mid = [f for f in top_sorted if total_f * 0.33 <= f < total_f * 0.66]
                        late = [f for f in top_sorted if f >= total_f * 0.66]
                        parts = []
                        if early: parts.append(f"**Preparation phase** (frames {', '.join(str(f) for f in early)})")
                        if mid: parts.append(f"**Forward swing / contact** (frames {', '.join(str(f) for f in mid)})")
                        if late: parts.append(f"**Follow-through** (frames {', '.join(str(f) for f in late)})")
                        if parts:
                            st.info("The model focuses most on: " + ", ".join(parts))
                    else:
                        st.info("Attention heatmap unavailable. Retrain the Siamese Transformer with the updated script.")

                    st.markdown("---")
                    st.markdown("""
                    <div style="font-size:1rem; font-weight:700; color:#1E293B; margin-bottom:1rem;
                         padding-bottom:0.5rem; border-bottom:2px solid #1E40AF; display:inline-block;">
                        Coaching Feedback
                    </div>
                    """, unsafe_allow_html=True)

                    fbi = generate_feedback(pj)
                    focus = [f for f in fbi if f["level"] == "poor"]
                    attn_i = [f for f in fbi if f["level"] == "moderate"]
                    good = [f for f in fbi if f["level"] == "good"]

                    s1, s2, s3 = st.columns(3)
                    with s1: st.metric("Focus Areas", len(focus))
                    with s2: st.metric("Needs Attention", len(attn_i))
                    with s3: st.metric("Doing Well", len(good))

                    st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)

                    if focus:
                        st.markdown("""
                        <div style="font-size:14px; font-weight:700; color:#DC2626; margin-bottom:0.5rem;
                             display:flex; align-items:center; gap:0.5rem;">
                            <svg width="16" height="16" fill="none" stroke="#DC2626" stroke-width="2" viewBox="0 0 24 24">
                                <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
                                <line x1="12" y1="16" x2="12.01" y2="16"/>
                            </svg>
                            Priority Focus Areas
                        </div>
                        """, unsafe_allow_html=True)
                        for fb in focus:
                            with st.expander(f"{fb['body_part']}  ·  deviation {fb['distance']:.3f}"):
                                st.markdown(f"""
                                <div style="background:#FEF2F2; border-left:4px solid #EF4444; border-radius:0 8px 8px 0;
                                     padding:0.75rem 1rem; font-size:14px; color:#374151; line-height:1.6;">
                                    {fb['message']}
                                </div>
                                """, unsafe_allow_html=True)

                    if attn_i:
                        st.markdown("""
                        <div style="font-size:14px; font-weight:700; color:#D97706; margin:0.75rem 0 0.5rem;
                             display:flex; align-items:center; gap:0.5rem;">
                            <svg width="16" height="16" fill="none" stroke="#D97706" stroke-width="2" viewBox="0 0 24 24">
                                <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
                                <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                            </svg>
                            Needs Attention
                        </div>
                        """, unsafe_allow_html=True)
                        for fb in attn_i:
                            with st.expander(f"{fb['body_part']}  ·  deviation {fb['distance']:.3f}"):
                                st.markdown(f"""
                                <div style="background:#FFFBEB; border-left:4px solid #F59E0B; border-radius:0 8px 8px 0;
                                     padding:0.75rem 1rem; font-size:14px; color:#374151; line-height:1.6;">
                                    {fb['message']}
                                </div>
                                """, unsafe_allow_html=True)

                    if good:
                        st.markdown("""
                        <div style="font-size:14px; font-weight:700; color:#16A34A; margin:0.75rem 0 0.5rem;
                             display:flex; align-items:center; gap:0.5rem;">
                            <svg width="16" height="16" fill="none" stroke="#16A34A" stroke-width="2" viewBox="0 0 24 24">
                                <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/>
                                <polyline points="22 4 12 14.01 9 11.01"/>
                            </svg>
                            Doing Well
                        </div>
                        """, unsafe_allow_html=True)
                        for fb in good:
                            with st.expander(f"{fb['body_part']}  ·  deviation {fb['distance']:.3f}"):
                                st.markdown(f"""
                                <div style="background:#F0FDF4; border-left:4px solid #22C55E; border-radius:0 8px 8px 0;
                                     padding:0.75rem 1rem; font-size:14px; color:#374151; line-height:1.6;">
                                    {fb['message']}
                                </div>
                                """, unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("""
                    <div style="font-size:1rem; font-weight:700; color:#1E293B; margin-bottom:1rem;
                         padding-bottom:0.5rem; border-bottom:2px solid #1E40AF; display:inline-block;">
                        Top 3 Coaching Tips
                    </div>
                    """, unsafe_allow_html=True)
                    for i, fb in enumerate(fbi[:3]):
                        level_color = "#DC2626" if fb["level"] == "poor" else "#D97706" if fb["level"] == "moderate" else "#16A34A"
                        level_bg = "#FEF2F2" if fb["level"] == "poor" else "#FFFBEB" if fb["level"] == "moderate" else "#F0FDF4"
                        st.markdown(f"""
                        <div style="background:{level_bg}; border-radius:12px; padding:1rem 1.25rem;
                             margin-bottom:0.75rem; border-left:4px solid {level_color};">
                            <div style="font-size:13px; font-weight:700; color:{level_color}; margin-bottom:4px;">
                                {i+1}. {fb['body_part']}
                            </div>
                            <div style="font-size:13px; color:#374151; line-height:1.6;">{fb['message']}</div>
                        </div>
                        """, unsafe_allow_html=True)

        try:
            os.unlink(tmp_path)
            if overlay: os.unlink(overlay)
        except:
            pass

        if kp is None:
            st.error("Could not process the video — check format and make sure a person is visible in frame.")


elif page == "DTW Analysis":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2rem; font-weight: 800; color: #0F172A; margin: 0;">DTW Analysis</h1>
        <p style="color: #64748B; font-size: 0.95rem; margin-top: 6px; margin-bottom: 0;">
            Dynamic Time Warping similarity scores across the expert video dataset
        </p>
    </div>
    """, unsafe_allow_html=True)

    dd = load_json("dtw_comparison_results.json")
    if not dd:
        st.error("Results file not found. Run dtw_comparison.py first.")
    else:
        df = pd.DataFrame(dd)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Strokes", len(df))
        with c2: st.metric("Mean Similarity", f"{df['similarity_score'].mean():.1f}%")
        with c3: st.metric("Max Similarity", f"{df['similarity_score'].max():.1f}%")
        with c4: st.metric("Min Similarity", f"{df['similarity_score'].min():.1f}%")

        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

        fds = sorted(df["folder"].unique())
        box_colors = ["#1E40AF", "#3B82F6", "#F59E0B", "#22C55E"]
        box_fill_colors = ["rgba(30,64,175,0.15)", "rgba(59,130,246,0.15)",
                           "rgba(245,158,11,0.15)", "rgba(34,197,94,0.15)"]

        fig = go.Figure()
        for i, fd in enumerate(fds):
            vals = df[df["folder"] == fd]["similarity_score"].values
            label = fd.replace("_tennis_", " / ").replace("_", " ").title()
            fig.add_trace(go.Box(
                y=vals, name=label,
                marker=dict(color=box_colors[i % len(box_colors)], size=5),
                line=dict(color=box_colors[i % len(box_colors)], width=2),
                fillcolor=box_fill_colors[i % len(box_fill_colors)],
                boxmean=True,
                hovertemplate="<b>%{fullData.name}</b><br>Similarity: %{y:.1f}%<extra></extra>",
            ))
        layout = base_layout(title="Similarity Score Distribution by Stroke Category", height=400, showlegend=True)
        layout.update(yaxis=dict(title="Similarity to Expert (%)", ticksuffix="%"),
                      xaxis=dict(gridcolor="rgba(0,0,0,0)"))
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div style="font-size:1rem; font-weight:700; color:#1E293B; margin:0.75rem 0 1rem;
             padding-bottom:0.5rem; border-bottom:2px solid #1E40AF; display:inline-block;">
            Average Deviation per Body Part
        </div>
        """, unsafe_allow_html=True)

        ja = {j: np.mean([r["per_joint_distances"][j] for r in dd]) for j in JOINT_NAMES}
        sj = sorted(ja.items(), key=lambda x: x[1], reverse=True)
        jnames = [JOINT_DISPLAY[j[0]] for j in sj]
        jvals = [j[1] for j in sj]
        jcols = [dist_color(v) for v in jvals]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=jvals[::-1], y=jnames[::-1],
            orientation="h",
            marker=dict(color=jcols[::-1], line=dict(width=0)),
            text=[f"  {v:.3f}" for v in jvals[::-1]],
            textposition="outside",
            textfont=dict(size=11, color="#334155"),
            hovertemplate="<b>%{y}</b><br>Avg DTW Distance: %{x:.3f}<extra></extra>",
            showlegend=False,
        ))
        layout2 = base_layout(title="Average DTW Distance per Joint (lower is better)", height=380)
        layout2.update(xaxis=dict(title="Avg DTW Distance", range=[0, max(jvals) * 1.3]),
                       margin=dict(l=130, r=20, t=55, b=45))
        fig2.update_layout(**layout2)
        st.plotly_chart(fig2, use_container_width=True)


elif page == "ML Classification":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2rem; font-weight: 800; color: #0F172A; margin: 0;">ML Classification</h1>
        <p style="color: #64748B; font-size: 0.95rem; margin-top: 6px; margin-bottom: 0;">
            Expert vs beginner classification using engineered biomechanical features
        </p>
    </div>
    """, unsafe_allow_html=True)

    md = load_json("ml_classification_results.json")
    fi = load_json("feature_importance.json")

    if not md:
        st.error("Results file not found. Run ml_classifiers.py first.")
    else:
        ms = list(md.keys())
        ac = [md[m]["cv_accuracy_mean"] for m in ms]
        ac_std = [md[m]["cv_accuracy_std"] for m in ms]
        mc_colors = {"Random Forest": "#1E40AF", "SVM": "#3B82F6", "KNN": "#F59E0B"}

        c1, c2, c3 = st.columns(3)
        for col, name in zip([c1, c2, c3], ms):
            with col:
                st.metric(name, f"{md[name]['cv_accuracy_mean']:.1%}",
                          delta=f"±{md[name]['cv_accuracy_std']:.1%} std")

        st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)

        fig = go.Figure()
        for m_name, m_acc, m_std in zip(ms, ac, ac_std):
            fig.add_trace(go.Bar(
                x=[m_name], y=[m_acc],
                name=m_name,
                marker=dict(color=mc_colors.get(m_name, "#3B82F6"), line=dict(width=0)),
                error_y=dict(type="data", array=[m_std], visible=True, color="#64748B"),
                text=[f"{m_acc:.1%}"],
                textposition="outside",
                textfont=dict(size=14, family="Fira Code, monospace", color="#1E293B"),
                hovertemplate=f"<b>{m_name}</b><br>Accuracy: {m_acc:.1%} ± {m_std:.1%}<extra></extra>",
            ))
        layout = base_layout(title="Cross-Validation Accuracy by Model", height=360, showlegend=False)
        layout.update(yaxis=dict(range=[0.75, 1.02], tickformat=".0%", title="CV Accuracy"),
                      xaxis=dict(title=""), bargap=0.4)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div style="font-size:1rem; font-weight:700; color:#1E293B; margin:0.5rem 0 1rem;
             padding-bottom:0.5rem; border-bottom:2px solid #1E40AF; display:inline-block;">
            Confusion Matrices
        </div>
        """, unsafe_allow_html=True)

        cm_cols = st.columns(3)
        for col, (n, r) in zip(cm_cols, md.items()):
            with col:
                st.markdown(f'<div style="font-size:13px; font-weight:700; color:#1E293B; text-align:center; margin-bottom:0.5rem;">{n}</div>', unsafe_allow_html=True)
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(np.array(r["confusion_matrix"]), annot=True, fmt="d",
                            cmap="Blues", ax=ax_cm,
                            xticklabels=["Expert", "Beginner"],
                            yticklabels=["Expert", "Beginner"],
                            linewidths=0.5, linecolor="#E2E8F0")
                ax_cm.set_xlabel("Predicted", fontsize=10)
                ax_cm.set_ylabel("Actual", fontsize=10)
                ax_cm.tick_params(labelsize=9)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()

        if fi:
            st.markdown("""
            <div style="font-size:1rem; font-weight:700; color:#1E293B; margin:0.5rem 0 1rem;
                 padding-bottom:0.5rem; border-bottom:2px solid #1E40AF; display:inline-block;">
                Top 15 Feature Importances (Random Forest)
            </div>
            """, unsafe_allow_html=True)

            t = fi[:15]
            fnames = [f["feature"].replace("_", " ").title() for f in t]
            fimps = [f["importance"] for f in t]

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=fimps[::-1], y=fnames[::-1],
                orientation="h",
                marker=dict(
                    color=fimps[::-1],
                    colorscale=[[0, "#BFDBFE"], [1, "#1E40AF"]],
                    line=dict(width=0),
                ),
                text=[f"  {v:.4f}" for v in fimps[::-1]],
                textposition="outside",
                textfont=dict(size=11, color="#334155"),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
                showlegend=False,
            ))
            layout3 = base_layout(title="Random Forest Feature Importance", height=420)
            layout3.update(xaxis=dict(title="Importance Score"), margin=dict(l=180, r=20, t=55, b=45))
            fig3.update_layout(**layout3)
            st.plotly_chart(fig3, use_container_width=True)


elif page == "Siamese Transformer":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2rem; font-weight: 800; color: #0F172A; margin: 0;">Siamese Transformer</h1>
        <p style="color: #64748B; font-size: 0.95rem; margin-top: 6px; margin-bottom: 0;">
            Deep learning model architecture, training curves, and attention analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    sd = load_json("siamese_transformer_results.json")
    sh = load_json("siamese_training_history.json")
    mi = load_json("siamese_model_info.json")

    if not sd:
        st.error("Results file not found. Run siamese_transformer.py first.")
    else:
        if mi:
            st.markdown("""
            <div style="font-size:1rem; font-weight:700; color:#1E293B; margin-bottom:0.75rem;
                 padding-bottom:0.5rem; border-bottom:2px solid #1E40AF; display:inline-block;">
                Model Architecture
            </div>
            """, unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Transformer Layers", mi.get("num_layers", "N/A"))
            with c2: st.metric("Attention Heads", mi.get("num_heads", "N/A"))
            with c3: st.metric("Embed Dimension", mi.get("embed_dim", "N/A"))
            with c4: st.metric("Total Parameters", f"{mi.get('total_parameters', 0):,}")

        if sh:
            st.markdown("""
            <div style="font-size:1rem; font-weight:700; color:#1E293B; margin:1rem 0 0.75rem;
                 padding-bottom:0.5rem; border-bottom:2px solid #1E40AF; display:inline-block;">
                Training History
            </div>
            """, unsafe_allow_html=True)

            dh = pd.DataFrame(sh)
            tc1, tc2 = st.columns(2)

            with tc1:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=dh["epoch"], y=dh["train_loss"], name="Train Loss",
                    line=dict(color="#1E40AF", width=2.5),
                    hovertemplate="Epoch %{x}<br>Train Loss: %{y:.4f}<extra></extra>",
                ))
                fig_loss.add_trace(go.Scatter(
                    x=dh["epoch"], y=dh["val_loss"], name="Val Loss",
                    line=dict(color="#EF4444", width=2.5, dash="dash"),
                    hovertemplate="Epoch %{x}<br>Val Loss: %{y:.4f}<extra></extra>",
                ))
                layout_loss = base_layout(title="Training & Validation Loss", height=320)
                layout_loss.update(xaxis=dict(title="Epoch"), yaxis=dict(title="Loss"))
                fig_loss.update_layout(**layout_loss)
                st.plotly_chart(fig_loss, use_container_width=True)

            with tc2:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    x=dh["epoch"], y=dh["val_accuracy"], name="Val Accuracy",
                    line=dict(color="#22C55E", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(34, 197, 94, 0.08)",
                    hovertemplate="Epoch %{x}<br>Accuracy: %{y:.1%}<extra></extra>",
                ))
                layout_acc = base_layout(title="Validation Accuracy", height=320, showlegend=False)
                layout_acc.update(xaxis=dict(title="Epoch"),
                                  yaxis=dict(title="Accuracy", range=[0.5, 1.0], tickformat=".0%"))
                fig_acc.update_layout(**layout_acc)
                st.plotly_chart(fig_acc, use_container_width=True)

        entries = [r for r in sd if "attention_weights" in r]
        if entries:
            st.markdown("""
            <div style="font-size:1rem; font-weight:700; color:#1E293B; margin:0.5rem 0 0.75rem;
                 padding-bottom:0.5rem; border-bottom:2px solid #1E40AF; display:inline-block;">
                Attention Viewer
            </div>
            """, unsafe_allow_html=True)

            opts = [f"{r['video']} ({r['folder']})" for r in entries[:20]]
            sel = st.selectbox("Select a stroke to inspect:", opts)
            if sel:
                e = entries[opts.index(sel)]
                a = np.array(e["attention_weights"])
                top_frames = e.get("top_attention_frames", [])

                bar_cols = ["#EF4444" if i in top_frames else "#3B82F6" for i in range(len(a))]

                fig_attn = go.Figure()
                fig_attn.add_trace(go.Bar(
                    x=list(range(len(a))), y=a,
                    marker=dict(color=bar_cols, line=dict(width=0)),
                    hovertemplate="Frame %{x}<br>Attention: %{y:.3f}<extra></extra>",
                    showlegend=False,
                ))
                layout_attn = base_layout(title=f"Attention Weights — {e['video']}", height=280)
                layout_attn.update(xaxis=dict(title="Frame Number"), yaxis=dict(title="Attention Weight"))
                fig_attn.update_layout(**layout_attn)
                st.plotly_chart(fig_attn, use_container_width=True)


elif page == "Method Comparison":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2rem; font-weight: 800; color: #0F172A; margin: 0;">Method Comparison</h1>
        <p style="color: #64748B; font-size: 0.95rem; margin-top: 6px; margin-bottom: 0;">
            Head-to-head comparison of DTW and Siamese Transformer similarity scores
        </p>
    </div>
    """, unsafe_allow_html=True)

    dd = load_json("dtw_comparison_results.json")
    sd = load_json("siamese_transformer_results.json")

    if not dd or not sd:
        st.error("One or both results files not found.")
    else:
        fds = sorted(set(r["folder"] for r in dd))
        dv = [np.mean([r["similarity_score"] for r in dd if r["folder"] == f]) for f in fds]
        sv = [np.mean([r["siamese_similarity"] for r in sd if r["folder"] == f]) for f in fds]
        labels = [f.replace("_tennis_", " / ").replace("_", " ").title() for f in fds]

        c1, c2 = st.columns(2)
        with c1: st.metric("Avg DTW Similarity (all)", f"{np.mean(dv):.1f}%")
        with c2: st.metric("Avg Siamese Similarity (all)", f"{np.mean(sv):.1f}%")

        st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="DTW", x=labels, y=dv,
            marker=dict(color="#1E40AF", line=dict(width=0)),
            text=[f"{v:.1f}%" for v in dv], textposition="outside",
            textfont=dict(size=11, color="#1E293B"),
            hovertemplate="<b>DTW</b> — %{x}<br>Similarity: %{y:.1f}%<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="Siamese Transformer", x=labels, y=sv,
            marker=dict(color="#F59E0B", line=dict(width=0)),
            text=[f"{v:.1f}%" for v in sv], textposition="outside",
            textfont=dict(size=11, color="#1E293B"),
            hovertemplate="<b>Siamese</b> — %{x}<br>Similarity: %{y:.1f}%<extra></extra>",
        ))
        layout = base_layout(title="DTW vs Siamese Transformer — Average Similarity by Stroke Category", height=420)
        layout.update(barmode="group", bargap=0.25, bargroupgap=0.08,
                      yaxis=dict(title="Similarity (%)", ticksuffix="%", range=[0, max(max(dv), max(sv)) * 1.2]),
                      xaxis=dict(title=""))
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

        all_dd = {r["video"]: r["similarity_score"] for r in dd}
        all_sd = {r["video"]: r["siamese_similarity"] for r in sd}
        common = set(all_dd.keys()) & set(all_sd.keys())

        if common:
            xs = [all_dd[v] for v in common]
            ys = [all_sd[v] for v in common]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(color="#3B82F6", size=7, opacity=0.65, line=dict(color="white", width=0.5)),
                hovertemplate="DTW: %{x:.1f}%<br>Siamese: %{y:.1f}%<extra></extra>",
                showlegend=False,
            ))
            layout2 = base_layout(title="DTW vs Siamese Similarity — Per-Stroke Correlation", height=360)
            layout2.update(xaxis=dict(title="DTW Similarity (%)", ticksuffix="%"),
                           yaxis=dict(title="Siamese Similarity (%)", ticksuffix="%"))
            fig2.update_layout(**layout2)
            st.plotly_chart(fig2, use_container_width=True)


elif page == "Individual Stroke":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2rem; font-weight: 800; color: #0F172A; margin: 0;">Individual Stroke</h1>
        <p style="color: #64748B; font-size: 0.95rem; margin-top: 6px; margin-bottom: 0;">
            Drill down into any individual stroke from the dataset
        </p>
    </div>
    """, unsafe_allow_html=True)

    dd = load_json("dtw_comparison_results.json")
    sd = load_json("siamese_transformer_results.json")

    if not dd:
        st.error("Results file not found.")
    else:
        sel = st.selectbox("Select a stroke:", sorted(set(r["video"] for r in dd)))

        if sel:
            e = next((r for r in dd if r["video"] == sel), None)
            if e:
                se = next((r for r in sd if r["video"] == sel), None) if sd else None

                info_col, score_col = st.columns([1, 2])
                with info_col:
                    st.markdown(f"""
                    <div style="background:white; border-radius:14px; padding:1.25rem; border:1px solid #E2E8F0;
                         box-shadow:0 1px 3px rgba(0,0,0,0.07);">
                        <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;
                             letter-spacing:0.8px; margin-bottom:0.5rem;">Stroke Info</div>
                        <div style="font-size:15px; font-weight:700; color:#1E293B; margin-bottom:4px;">
                            {get_stroke_type(e['folder'])}
                        </div>
                        <div style="font-size:13px; color:#64748B;">{get_ball_type(e['folder'])}</div>
                        <div style="font-size:12px; color:#94A3B8; margin-top:6px; font-family:'Fira Code',monospace;">
                            {sel}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with score_col:
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        dtw_score = e["similarity_score"]
                        st.metric("DTW Similarity", f"{dtw_score}%")
                    with sc2:
                        if se:
                            siam_score = se["siamese_similarity"]
                            st.metric("Siamese Similarity", f"{siam_score}%")
                        else:
                            st.metric("Siamese Similarity", "N/A")

                st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)

                if "per_joint_distances" in e:
                    pj = e["per_joint_distances"]
                    fb = generate_feedback(pj)
                    sj = sorted(pj.items(), key=lambda x: x[1], reverse=True)
                    jnames = [JOINT_DISPLAY[j[0]] for j in sj]
                    jvals = [j[1] for j in sj]
                    jcols = [dist_color(v) for v in jvals]

                    st.markdown("""
                    <div style="font-size:1rem; font-weight:700; color:#1E293B; margin-bottom:0.75rem;
                         padding-bottom:0.5rem; border-bottom:2px solid #1E40AF; display:inline-block;">
                        Body Part Deviation
                    </div>
                    """, unsafe_allow_html=True)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=jvals[::-1], y=jnames[::-1],
                        orientation="h",
                        marker=dict(color=jcols[::-1], line=dict(width=0)),
                        text=[f"  {'Close' if v < 0.2 else 'Work' if v < 0.4 else 'Focus'} ({v:.3f})"
                              for v in jvals[::-1]],
                        textposition="outside",
                        textfont=dict(size=11, color="#334155"),
                        hovertemplate="<b>%{y}</b><br>Deviation: %{x:.3f}<extra></extra>",
                        showlegend=False,
                    ))
                    layout = base_layout(title="", height=380)
                    layout.update(xaxis=dict(title="DTW Distance", range=[0, max(jvals) * 1.4]),
                                  margin=dict(l=130, r=20, t=20, b=45))
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("""
                    <div style="font-size:1rem; font-weight:700; color:#1E293B; margin-bottom:0.75rem;
                         padding-bottom:0.5rem; border-bottom:2px solid #1E40AF; display:inline-block;">
                        Coaching Feedback
                    </div>
                    """, unsafe_allow_html=True)

                    for f in [f for f in fb if f["level"] == "poor"]:
                        with st.expander(f"Priority — {f['body_part']}  ·  {f['distance']:.3f}"):
                            st.markdown(f"""
                            <div style="background:#FEF2F2; border-left:4px solid #EF4444; border-radius:0 8px 8px 0;
                                 padding:0.75rem 1rem; font-size:14px; color:#374151; line-height:1.6;">
                                {f['message']}
                            </div>
                            """, unsafe_allow_html=True)

                    for f in [f for f in fb if f["level"] == "moderate"]:
                        with st.expander(f"Attention — {f['body_part']}  ·  {f['distance']:.3f}"):
                            st.markdown(f"""
                            <div style="background:#FFFBEB; border-left:4px solid #F59E0B; border-radius:0 8px 8px 0;
                                 padding:0.75rem 1rem; font-size:14px; color:#374151; line-height:1.6;">
                                {f['message']}
                            </div>
                            """, unsafe_allow_html=True)

                    for f in [f for f in fb if f["level"] == "good"]:
                        with st.expander(f"Good — {f['body_part']}  ·  {f['distance']:.3f}"):
                            st.markdown(f"""
                            <div style="background:#F0FDF4; border-left:4px solid #22C55E; border-radius:0 8px 8px 0;
                                 padding:0.75rem 1rem; font-size:14px; color:#374151; line-height:1.6;">
                                {f['message']}
                            </div>
                            """, unsafe_allow_html=True)
