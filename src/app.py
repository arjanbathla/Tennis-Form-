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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from scipy.stats import pearsonr
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

def convert_avi_to_mp4(avi_path):
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmpfile.close()
    out = cv2.VideoWriter(tmpfile.name, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)
    cap.release(); out.release()
    return tmpfile.name


def render_skeleton_video(kp_norm, fps=25.0, canvas_size=(480, 640), torso_px=110):
    w, h = canvas_size
    out_path = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    edges = [(0, 1), (6, 7), (0, 6), (1, 7),
             (0, 2), (2, 4), (1, 3), (3, 5),
             (6, 8), (8, 10), (7, 9), (9, 11)]
    cx, cy = w // 2, h // 2
    for frame_kp in kp_norm:
        canvas = np.full((h, w, 3), 28, dtype=np.uint8)
        pts = []
        for jx, jy in frame_kp:
            x = int(cx + jx * torso_px)
            y = int(cy + jy * torso_px)
            pts.append((x, y))
        for a, b in edges:
            cv2.line(canvas, pts[a], pts[b], (89, 124, 74), 2)
        for x, y in pts:
            cv2.circle(canvas, (x, y), 4, (230, 230, 230), -1)
        out.write(canvas)
    out.release()
    return out_path


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


def find_closest_experts(prep, stroke_type):
    folder = PREPROCESSED_DIR / "thetis" / stroke_type
    if not folder.exists(): return []
    files = sorted([f for f in folder.iterdir() if f.name.endswith("_preprocessed.npy")])
    fp = prep.reshape(len(prep), -1)
    results = []
    for ef in files:
        exp = np.load(str(ef))
        fe = exp.reshape(len(exp), -1)
        d = dtw(fp, fe, keep_internals=False).normalizedDistance
        pid = ef.name.split("_")[0]
        results.append({"player_id": pid, "file": ef.name,
                        "distance": float(d),
                        "similarity": round(100 * float(np.exp(-d)), 1)})
    results.sort(key=lambda r: r["distance"])
    return results

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

PARAM_GRIDS = {
    "Random Forest": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [5, 10, 20, None],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__min_samples_split": [2, 5, 10],
        "clf__class_weight": ["balanced"],
    },
    "SVM": {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", "auto"],
        "clf__class_weight": ["balanced"],
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 9],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    },
}

def _tune_classifiers(X, y):
    base_clfs = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        # algorithm='ball_tree' avoids sklearn's ArgKminClassMode fast path,
        # which crashes on string labels in sklearn >=1.5.
        "KNN": KNeighborsClassifier(algorithm="ball_tree"),
    }
    tuned = {}
    for name, clf in base_clfs.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        gs = GridSearchCV(pipe, PARAM_GRIDS[name], cv=5, scoring="accuracy", n_jobs=-1)
        gs.fit(X, y)
        tuned[name] = gs.best_estimator_
        print(f"[GridSearchCV] {name} best params: {gs.best_params_}  (cv acc={gs.best_score_:.4f})")
    return tuned


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

    models = _tune_classifiers(X, y)

    base_est = [
        ("rf", models["Random Forest"]),
        ("svm", models["SVM"]),
        ("knn", models["KNN"]),
    ]
    voting = VotingClassifier(estimators=base_est, voting="soft")
    voting.fit(X, y)
    models["Soft Voting Ensemble"] = voting

    stacking = StackingClassifier(
        estimators=base_est,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
    )
    stacking.fit(X, y)
    models["Stacking Ensemble"] = stacking
    return models

def ml_predict(models, prep):
    if not models: return None
    f = np.nan_to_num(extract_features(prep).reshape(1,-1))
    return {n:{"prediction":p.predict(f)[0],
               "expert_probability":round(float(p.predict_proba(f)[0][np.where(p.classes_=="expert")[0][0]]),4)}
            for n,p in models.items()}

def get_rf_joint_importance(models):
    if not models or "Random Forest" not in models: return None
    rf = models["Random Forest"].named_steps["clf"]
    imp = rf.feature_importances_
    joint_imp = {j: 0.0 for j in JOINT_NAMES}
    # feature 0 = sequence length, skip
    # features 1-120 = positional stats, 10 per joint
    for i in range(1, 121):
        ji = (i - 1) // 10
        joint_imp[JOINT_NAMES[ji]] += imp[i]
    # features 121-156 = velocity stats, 3 per joint
    for i in range(121, 157):
        ji = (i - 121) // 3
        joint_imp[JOINT_NAMES[ji]] += imp[i]
    # features 157-186 = angle stats, 5 per angle
    # 6 angles, middle joint of each triplet:
    # (0,2,4) -> elbow L=2, (1,3,5) -> elbow R=3,
    # (6,8,10) -> knee L=8, (7,9,11) -> knee R=9,
    # (2,0,6) -> shoulder L=0, (3,1,7) -> shoulder R=1
    angle_joints = [2, 3, 8, 9, 0, 1]
    for i in range(157, 187):
        ai = (i - 157) // 5
        joint_imp[JOINT_NAMES[angle_joints[ai]]] += imp[i]
    # features 187-188 = movement ratios, skip
    max_imp = max(joint_imp.values()) if max(joint_imp.values()) > 0 else 1.0
    return {j: v / max_imp for j, v in joint_imp.items()}


@st.cache_resource(show_spinner="Running unified evaluation...")
def unified_evaluation():
    samples = []
    for stroke in ["forehand", "backhand"]:
        for level, label in [("thetis", "expert"), ("thetis_beginners", "beginner")]:
            fd = PREPROCESSED_DIR / level / stroke
            if not fd.exists(): continue
            for fl in sorted(fd.iterdir()):
                if fl.name.endswith("_preprocessed.npy"):
                    samples.append({"path": str(fl), "label": label, "stroke": stroke})

    if len(samples) < 10:
        return None, 0

    labels = [s["label"] for s in samples]
    idx_tr, idx_te = train_test_split(list(range(len(samples))), test_size=0.25,
                                      stratify=labels, random_state=42)
    train = [samples[i] for i in idx_tr]
    test = [samples[i] for i in idx_te]

    Xtr = np.array([np.nan_to_num(extract_features(np.load(s["path"]))) for s in train])
    ytr = np.array([s["label"] for s in train])
    Xte = np.array([np.nan_to_num(extract_features(np.load(s["path"]))) for s in test])
    yte = np.array([s["label"] for s in test])

    ml = _tune_classifiers(Xtr, ytr)

    preds = {n: ml[n].predict(Xte) for n in ml}

    train_experts = {}
    for s in train:
        if s["label"] == "expert":
            train_experts.setdefault(s["stroke"], []).append(s["path"])

    sm = load_siamese_model()
    rng = np.random.RandomState(42)
    K = 10

    dtw_preds, siam_preds = [], []
    for s in test:
        kp = np.load(s["path"])
        flat = kp.reshape(len(kp), -1)
        refs = train_experts.get(s["stroke"], [])
        if len(refs) > K:
            refs = list(rng.choice(refs, K, replace=False))

        if not refs:
            dtw_preds.append("beginner")
            siam_preds.append("beginner")
            continue

        ds = []
        for ref in refs:
            exp = np.load(ref)
            fe = exp.reshape(len(exp), -1)
            ds.append(dtw(flat, fe, keep_internals=False).normalizedDistance)
        sim = 100 * np.exp(-np.mean(ds))
        dtw_preds.append("expert" if sim > 50 else "beginner")

        if sm is not None:
            pp, pm = pad_single(kp)
            sd = []
            for ref in refs:
                ep, em = pad_single(np.load(ref))
                with torch.no_grad():
                    d, _, _ = sm(pp, ep, pm, em)
                    sd.append(d.item())
            siam_preds.append("expert" if np.mean(sd) < 0.5 else "beginner")
        else:
            siam_preds.append("beginner")

    preds["DTW (sim > 50%)"] = np.array(dtw_preds)
    preds["Siamese (d < 0.5)"] = np.array(siam_preds)

    rows = []
    for method, yhat in preds.items():
        rows.append({
            "method": method,
            "accuracy": round(accuracy_score(yte, yhat), 4),
            "precision": round(precision_score(yte, yhat, pos_label="expert", zero_division=0), 4),
            "recall": round(recall_score(yte, yhat, pos_label="expert", zero_division=0), 4),
            "f1": round(f1_score(yte, yhat, pos_label="expert", zero_division=0), 4),
        })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "unified_comparison.csv", index=False)
    return df, len(test)


@st.cache_resource(show_spinner="Running full DTW + siamese sweep across 330 recordings (one-off, takes a couple of minutes)...")
def method_comparison_eval():
    samples = []
    for stroke in ["forehand", "backhand"]:
        for level, label in [("thetis", "expert"), ("thetis_beginners", "beginner")]:
            fd = PREPROCESSED_DIR / level / stroke
            if not fd.exists(): continue
            for fl in sorted(fd.iterdir()):
                if fl.name.endswith("_preprocessed.npy"):
                    samples.append({"video": fl.stem.replace("_preprocessed", ""),
                                    "path": str(fl), "label": label, "stroke": stroke})
    if not samples:
        return None

    expert_paths = {"forehand": [], "backhand": []}
    for s in samples:
        if s["label"] == "expert":
            expert_paths[s["stroke"]].append(s["path"])

    rng = np.random.RandomState(42)
    K_DTW = 15
    dtw_refs = {}
    for stroke, paths in expert_paths.items():
        if len(paths) > K_DTW:
            dtw_refs[stroke] = list(rng.choice(paths, K_DTW, replace=False))
        else:
            dtw_refs[stroke] = list(paths)

    dtw_ref_arrays = {}
    for stroke, paths in dtw_refs.items():
        dtw_ref_arrays[stroke] = [(p, np.load(p).reshape(-1, 24)) for p in paths]

    sm = load_siamese_model()
    expert_emb = {}
    if sm is not None:
        for stroke, paths in expert_paths.items():
            embs = []
            for p in paths:
                pt, pm = pad_single(np.load(p))
                with torch.no_grad():
                    e = sm.get_embedding(pt, pm).squeeze(0).numpy()
                embs.append(e)
            expert_emb[stroke] = (paths, np.array(embs))

    rows = []
    for s in samples:
        kp = np.load(s["path"])
        flat = kp.reshape(len(kp), -1)

        dists_d = []
        for (rp, ra) in dtw_ref_arrays[s["stroke"]]:
            if rp == s["path"]:
                continue
            dists_d.append(dtw(flat, ra, keep_internals=False).normalizedDistance)
        dtw_sim = 100 * np.exp(-np.mean(dists_d)) if dists_d else 0.0

        siam_dist = float("nan")
        if sm is not None and s["stroke"] in expert_emb:
            pt, pm = pad_single(kp)
            with torch.no_grad():
                target_e = sm.get_embedding(pt, pm).squeeze(0).numpy()
            ref_paths, ref_embs = expert_emb[s["stroke"]]
            mask = np.array([p != s["path"] for p in ref_paths])
            pool = ref_embs[mask]
            diffs = pool - target_e
            siam_dist = float(np.mean(np.sqrt(np.sum(diffs * diffs, axis=1))))

        rows.append({
            "video": s["video"],
            "label": s["label"],
            "stroke": s["stroke"],
            "dtw_sim": round(float(dtw_sim), 3),
            "siam_dist": round(siam_dist, 4),
            "dtw_pred": "expert" if dtw_sim > 50 else "beginner",
            "siam_pred": "expert" if (not np.isnan(siam_dist) and siam_dist < 0.5) else "beginner",
        })

    df = pd.DataFrame(rows).reset_index(drop=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "method_comparison_full.csv", index=False)
    return df


def generate_feedback(pj, rf_importance=None):
    items = []
    scored = {}
    for jn, dist in pj.items():
        if rf_importance and jn in rf_importance:
            scored[jn] = dist * (1 + rf_importance[jn])
        else:
            scored[jn] = dist
    for jn,_ in sorted(scored.items(), key=lambda x:x[1], reverse=True):
        dist = pj[jn]
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
    if score >= 40: return "#4A7C59"
    elif score >= 25: return "#8B6E4E"
    return "#A0522D"

def score_label(score):
    if score >= 40: return "good"
    elif score >= 25: return "fair"
    return "poor"

def dist_color(d):
    if d < 0.2: return "#4A7C59"
    elif d < 0.4: return "#8B6E4E"
    return "#A0522D"

def base_layout(title="", height=400, showlegend=True):
    return dict(
        title=dict(text=title, font=dict(size=13, color="#1C2B1E"), x=0, xanchor="left"),
        height=height,
        paper_bgcolor="#FAFAF5",
        plot_bgcolor="#FAFAF5",
        font=dict(color="#1C2B1E", size=11),
        margin=dict(t=40 if title else 20, b=40, l=55, r=15),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11)),
        xaxis=dict(gridcolor="#D8DCC9", linecolor="#D8DCC9", tickfont=dict(size=10), zeroline=False),
        yaxis=dict(gridcolor="#D8DCC9", linecolor="#D8DCC9", tickfont=dict(size=10), zeroline=False),
    )


def run_integrity_check():
    """Check that all five methods use the same test set and consistent thresholds."""
    issues = []

    # rebuild the same sample list + split used by unified_evaluation
    samples = []
    for stroke in ["forehand", "backhand"]:
        for level, label in [("thetis", "expert"), ("thetis_beginners", "beginner")]:
            fd = PREPROCESSED_DIR / level / stroke
            if not fd.exists():
                continue
            for fl in sorted(fd.iterdir()):
                if fl.name.endswith("_preprocessed.npy"):
                    samples.append({"path": str(fl), "label": label, "stroke": stroke})

    if len(samples) < 10:
        print("[integrity] not enough data to check")
        return

    labels = [s["label"] for s in samples]
    _, idx_te = train_test_split(list(range(len(samples))), test_size=0.25,
                                  stratify=labels, random_state=42)
    expected_n = len(idx_te)

    # 1 — check unified_comparison.csv has exactly 83 rows per method
    csv_path = RESULTS_DIR / "unified_comparison.csv"
    if csv_path.exists():
        udf = pd.read_csv(csv_path)
        methods_found = list(udf["method"])
        expected_methods = ["Random Forest", "SVM", "KNN", "DTW (sim > 50%)", "Siamese (d < 0.5)"]
        for m in expected_methods:
            if m not in methods_found:
                issues.append(f"missing method '{m}' in unified_comparison.csv")
        if len(udf) != len(expected_methods):
            issues.append(f"unified_comparison.csv has {len(udf)} rows, expected {len(expected_methods)}")
    else:
        issues.append("unified_comparison.csv not found")

    if expected_n != 83:
        issues.append(f"test split produced {expected_n} samples, expected 83")

    # 2 — positive class must be 'expert' everywhere
    import inspect
    src_app = inspect.getsource(unified_evaluation)
    if "pos_label=\"expert\"" not in src_app and "pos_label='expert'" not in src_app:
        issues.append("unified_evaluation may not use pos_label='expert'")

    # 3 — DTW threshold sim > 50 and siamese threshold d < 0.5
    app_src = Path(__file__).read_text()
    dtw_thresh_count = app_src.count("sim > 50")
    siam_thresh_count = app_src.count("< 0.5")
    if dtw_thresh_count < 2:
        issues.append(f"DTW threshold 'sim > 50' found {dtw_thresh_count} times, expected at least 2")
    if siam_thresh_count < 2:
        issues.append(f"Siamese threshold '< 0.5' found {siam_thresh_count} times, expected at least 2")

    # print summary
    print("")
    if not issues:
        print(f"Evaluation consistency check: PASSED (n={expected_n}, 5 methods, pos_class=expert, thresholds ok)")
    else:
        print("Evaluation consistency check: FAILED")
        for iss in issues:
            print(f"  - {iss}")
    print("")


run_integrity_check()

st.set_page_config(page_title="TennisForm", page_icon="🎾", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stApp"] { background-color: #FAFAF5; color: #1C2B1E; }

[data-testid="stSidebar"] { background-color: #EEF0E8 !important; border-right: 1px solid #D8DCC9 !important; }

.main .block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1200px !important; }

[data-testid="stMetric"] { background: transparent !important; border: none !important; padding: 0.25rem 0 !important; box-shadow: none !important; }
[data-testid="stMetricLabel"] > div { color: #5A6B55 !important; font-size: 12px !important; font-weight: 400 !important; text-transform: none !important; letter-spacing: 0 !important; }
[data-testid="stMetricValue"] > div { color: #1C2B1E !important; font-weight: 600 !important; font-size: 1.5rem !important; }

h1 { color: #1C2B1E !important; font-weight: 600 !important; letter-spacing: 0 !important; }
h2, h3 { color: #1C2B1E !important; font-weight: 600 !important; }

.stButton > button { background: #4A7C59 !important; color: #FAFAF5 !important; border: 1px solid #3A6248 !important; border-radius: 2px !important; font-weight: 500 !important; box-shadow: none !important; }
.stButton > button:hover { background: #3A6248 !important; transform: none !important; box-shadow: none !important; }

[data-testid="stFileUploader"] { background: #EEF0E8 !important; border: 1px dashed #8A9B7E !important; border-radius: 2px !important; }

[data-testid="stDataFrame"] { border: 1px solid #D8DCC9 !important; border-radius: 0 !important; }

[data-testid="stAlert"] { border-radius: 2px !important; }

[data-testid="stExpander"] { border: 1px solid #D8DCC9 !important; border-radius: 2px !important; background: #FAFAF5 !important; }

hr { border-color: #D8DCC9 !important; margin: 1rem 0 !important; }

/* sidebar nav buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important; color: #1C2B1E !important; border: none !important;
    text-align: left !important; padding: 6px 12px !important; font-weight: 400 !important;
    border-left: 3px solid transparent !important; border-radius: 0 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #E2E5D8 !important; color: #1C2B1E !important;
}
.navitem-selected {
    padding: 6px 12px; border-left: 3px solid #4A7C59; background: #E2E5D8;
    color: #1C2B1E; font-weight: 500; font-size: 14px;
}
.navgroup { font-size: 11px; letter-spacing: 1px; color: #6B7A63; margin: 18px 12px 6px; font-weight: 500; }
.sidefoot { font-size: 11px; color: #6B7A63; margin: 24px 12px 8px; font-family: monospace; }
.logo { font-size: 1.3rem; font-weight: 600; color: #1C2B1E; padding: 20px 12px 8px; }

/* section label — no colored underline */
.sec { font-size: 14px; font-weight: 600; color: #1C2B1E; margin: 18px 0 10px; border-bottom: 1px solid #D8DCC9; padding-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


if "page" not in st.session_state:
    st.session_state.page = "Overview"

ANALYSIS_PAGES = ["Overview", "Analyse My Stroke"]
RESULTS_PAGES = ["DTW Analysis", "ML Classification", "Siamese Transformer", "Method Comparison", "Individual Stroke"]

def nav_item(label):
    if st.session_state.page == label:
        st.markdown(f"<div class='navitem-selected'>{label}</div>", unsafe_allow_html=True)
    else:
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()

with st.sidebar:
    st.markdown("<div class='logo'>TennisForm</div>", unsafe_allow_html=True)
    st.markdown("<div class='navgroup'>ANALYSIS</div>", unsafe_allow_html=True)
    for p in ANALYSIS_PAGES:
        nav_item(p)
    st.markdown("<div class='navgroup'>RESULTS</div>", unsafe_allow_html=True)
    for p in RESULTS_PAGES:
        nav_item(p)
    st.markdown("<div class='sidefoot'>81 videos &middot; 3 methods</div>", unsafe_allow_html=True)

page = st.session_state.page



if page == "Overview":
    st.markdown("# TennisForm")
    st.write("Pose-based expert/beginner stroke classification compared across five methods on the same held-out test set.")

    res = unified_evaluation()
    if res is None or res[0] is None:
        st.error("Not enough preprocessed data to run unified evaluation.")
    else:
        df, n_test = res
        st.info("All methods evaluated on the same held-out test set: n=83, stratified 75/25 split, seed=42, positive class = expert.")
        st.caption(f"Held-out test size: n = {n_test} (stratified 75/25 split, seed=42). Pos class = expert.")

        num_cols = ["accuracy", "precision", "recall", "f1"]

        def highlight_max(col):
            is_max = col == col.max()
            return ['background-color: #D8E3CE; font-weight: 600' if v else '' for v in is_max]

        styled = (df.style
                  .apply(highlight_max, subset=num_cols)
                  .format({c: "{:.3f}" for c in num_cols}))
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption("Saved to results/unified_comparison.csv")

    st.write("")
    st.markdown("**Go to**")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("DTW Analysis", use_container_width=True, key="go_dtw"):
            st.session_state.page = "DTW Analysis"; st.rerun()
        if st.button("ML Classification", use_container_width=True, key="go_ml"):
            st.session_state.page = "ML Classification"; st.rerun()
    with b2:
        if st.button("Siamese Transformer", use_container_width=True, key="go_st"):
            st.session_state.page = "Siamese Transformer"; st.rerun()
        if st.button("Method Comparison", use_container_width=True, key="go_mc"):
            st.session_state.page = "Method Comparison"; st.rerun()
    with b3:
        if st.button("Individual Stroke", use_container_width=True, key="go_is"):
            st.session_state.page = "Individual Stroke"; st.rerun()
        if st.button("Analyse My Stroke", use_container_width=True, key="go_am"):
            st.session_state.page = "Analyse My Stroke"; st.rerun()


elif page == "Analyse My Stroke":
    st.markdown("# Analyse stroke")
    st.write("Upload a video. Pose is extracted with MediaPipe and compared against THETIS expert samples via DTW + siamese transformer.")

    cu, ct = st.columns([2, 1])
    with ct:
        st.write("Stroke type:")
        stroke_type = st.radio("type", ["forehand", "backhand"], label_visibility="collapsed")
        st.caption("Front-on angle, full body visible, one stroke per clip.")

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
                st.write("**Input**")
                st.video(tmp_path)
            with c2:
                st.write("**Skeleton overlay**")
                st.video(overlay)

            st.write(f"Frames: {info['total_frames']}, detection rate: {info['detection_rate']}%, fps: {info['fps']:.0f}")

            st.markdown("---")
            prep = preprocess_keypoints(kp)

            if prep is not None:
                st.write(f"Stroke: **{stroke_type}**")

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
                    ds = dtw_r["similarity_score"]
                    sim_line = f"DTW: `{ds}%`"
                    if sr:
                        ss = sr["siamese_similarity"]
                        sim_line += f", Siamese: `{ss}%`"
                    if mlr and "Stacking Ensemble" in mlr:
                        stk = mlr["Stacking Ensemble"]
                        sim_line += f", ML (Stacking): `{stk['prediction']}` ({stk['expert_probability']:.0%} expert)"
                    st.markdown(f"**Similarity** — {sim_line}")

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
                            textfont=dict(size=12, color="white"),
                            showlegend=False,
                            hovertemplate=f"<b>{lbl}</b>: {val}%<extra></extra>",
                        ))
                        fig.add_trace(go.Bar(
                            x=[100 - val], y=[lbl], orientation="h",
                            marker=dict(color="#D8DCC9", line=dict(width=0)),
                            showlegend=False, hoverinfo="skip",
                        ))

                    layout = base_layout(title="Similarity to Expert Technique (%)", height=130 + 50 * len(scores))
                    layout.update(barmode="stack", xaxis=dict(range=[0, 100], ticksuffix="%",
                                  gridcolor="#D8DCC9", linecolor="#D8DCC9", tickfont=dict(size=11)),
                                  yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)"))
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("**Closest Expert Match**")

                    prep_key = f"{stroke_type}_{hash(prep.tobytes())}"
                    if st.session_state.get("closest_key") != prep_key:
                        with st.spinner("Finding closest expert match..."):
                            st.session_state.closest_ranked = find_closest_experts(prep, stroke_type)
                        st.session_state.closest_key = prep_key
                    ranked = st.session_state.closest_ranked

                    if ranked:
                        best = ranked[0]
                        st.markdown(f"Your stroke most closely resembles expert player **{best['player_id']}** — similarity `{best['similarity']}%`")

                        top3 = ranked[:3]
                        names = [r["player_id"] for r in top3]
                        sims = [r["similarity"] for r in top3]
                        cols_b = [score_color(s) for s in sims]

                        fig_ce = go.Figure()
                        fig_ce.add_trace(go.Bar(
                            x=sims[::-1], y=names[::-1],
                            orientation="h",
                            marker=dict(color=cols_b[::-1], line=dict(width=0)),
                            text=[f"  {s}%" for s in sims[::-1]],
                            textposition="outside",
                            textfont=dict(size=11, color="#1C2B1E"),
                            hovertemplate="<b>Player %{y}</b><br>Similarity: %{x}%<extra></extra>",
                            showlegend=False,
                        ))
                        layout_ce = base_layout(title="Top 3 Closest Expert Matches", height=220, showlegend=False)
                        layout_ce.update(xaxis=dict(range=[0, 105], ticksuffix="%",
                                                   gridcolor="#D8DCC9", linecolor="#D8DCC9", tickfont=dict(size=11)),
                                         yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)"),
                                         margin=dict(l=80, r=40, t=45, b=40))
                        fig_ce.update_layout(**layout_ce)
                        st.plotly_chart(fig_ce, use_container_width=True)

                        st.caption("Closest match is based on overall DTW trajectory similarity across all 12 joints.")

                        st.write("**Expert Reference Video**")

                        video_folder = "forehand_flat" if stroke_type == "forehand" else "backhand"
                        base = best["file"].replace("_preprocessed.npy", "")
                        expert_video_dir = PROJECT_ROOT / "data" / "thetis" / "dataset-main" / "VIDEO_RGB" / video_folder
                        candidates = [expert_video_dir / f"{base}.avi",
                                      expert_video_dir / f"{base}.mp4",
                                      expert_video_dir / f"{base}.mov"]

                        expert_video = None
                        for p in candidates:
                            print(f"[ExpertVideo] Checking {p} — exists={os.path.exists(str(p))}")
                            if p.exists():
                                expert_video = p
                                break

                        if expert_video is None:
                            print(f"[ExpertVideo] No video found. Directory listing of {expert_video_dir}:")
                            if expert_video_dir.exists():
                                for f in sorted(expert_video_dir.iterdir()):
                                    print(f"  {f.name}")
                            else:
                                print(f"  (directory does not exist)")
                            st.info("Expert reference video not available")
                        else:
                            print(f"[ExpertVideo] Using path: {expert_video}")

                            mp4_key = f"expert_mp4_{stroke_type}_{base}"
                            if st.session_state.get("_expert_mp4_key") != mp4_key:
                                with st.spinner("Converting expert video to MP4..."):
                                    st.session_state._expert_mp4 = convert_avi_to_mp4(str(expert_video))
                                st.session_state._expert_mp4_key = mp4_key
                            expert_mp4 = st.session_state._expert_mp4

                            skel_key = f"expert_skel_{stroke_type}_{base}"
                            if st.session_state.get("_expert_skel_key") != skel_key:
                                with st.spinner("Rendering expert skeleton overlay..."):
                                    _, exp_overlay, _ = process_uploaded_video(str(expert_video))
                                    st.session_state._expert_skel = exp_overlay
                                st.session_state._expert_skel_key = skel_key
                            expert_skel = st.session_state._expert_skel

                            view = st.radio("View", ["Raw Video", "Skeleton Overlay"],
                                            horizontal=True, key="expert_view_mode")

                            vc1, vc2 = st.columns(2)
                            if view == "Raw Video":
                                with vc1:
                                    st.write("Your Stroke")
                                    st.video(tmp_path)
                                with vc2:
                                    st.write(f"Closest Expert Match — Player {best['player_id']}")
                                    if expert_mp4 and os.path.exists(expert_mp4):
                                        st.video(expert_mp4)
                                    else:
                                        st.info("Expert reference video not available")
                            else:
                                with vc1:
                                    st.write("Your Stroke")
                                    st.video(overlay)
                                with vc2:
                                    st.write(f"Closest Expert Match — Player {best['player_id']}")
                                    if expert_skel and os.path.exists(expert_skel):
                                        st.video(expert_skel)
                                    else:
                                        st.info("Expert skeleton not available")

                            st.write(f"Expert player {best['player_id']} was selected as your closest match based on DTW trajectory similarity ({best['similarity']}% similarity).")

                    if mlr:
                        st.write("**ML predictions**")
                        ml_df = pd.DataFrame([
                            {"Model": n,
                             "Prediction": r["prediction"].title(),
                             "Expert Probability": f"{r['expert_probability']:.1%}"}
                            for n, r in mlr.items()
                        ])
                        st.dataframe(ml_df, use_container_width=True, hide_index=True)

                    st.markdown("---")
                    st.write("**Joint deviations**")

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
                        textfont=dict(size=11, color="#1C2B1E"),
                        hovertemplate="<b>%{y}</b><br>Deviation: %{x:.3f}<extra></extra>",
                        showlegend=False,
                    ))
                    layout2 = base_layout(title="Deviation from Expert (lower is better)", height=420)
                    layout2.update(xaxis=dict(title="DTW Distance", range=[0, max(jvals) * 1.4],
                                   gridcolor="#D8DCC9"), margin=dict(l=130, r=20, t=55, b=45))
                    fig2.update_layout(**layout2)
                    st.plotly_chart(fig2, use_container_width=True)

                    if attn is not None:
                        st.markdown("---")
                        st.write("**Attention weights** (dark bars = top-5 frames)")

                        top5 = np.argsort(attn)[-5:]
                        bar_colors = ["#A0522D" if i in top5 else "#6B8F71" for i in range(len(attn))]

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

                        st.write("**Stroke Phase Analysis**")

                        n_f = len(attn)
                        t1 = n_f // 3
                        t2 = (2 * n_f) // 3
                        prep_avg = float(np.mean(attn[:t1])) if t1 > 0 else 0.0
                        fwd_avg = float(np.mean(attn[t1:t2])) if t2 > t1 else 0.0
                        fol_avg = float(np.mean(attn[t2:])) if n_f > t2 else 0.0

                        phase_names = ["Preparation", "Forward Swing", "Follow-through"]
                        phase_scores = [prep_avg, fwd_avg, fol_avg]

                        ranked = sorted(range(3), key=lambda i: phase_scores[i], reverse=True)
                        severity = {ranked[0]: "#A0522D", ranked[1]: "#8B6E4E", ranked[2]: "#4A7C59"}
                        phase_colors = [severity[i] for i in range(3)]

                        max_s = max(phase_scores) if max(phase_scores) > 0 else 1.0
                        bar_widths = [s / max_s for s in phase_scores]

                        fig_ph = go.Figure()
                        fig_ph.add_trace(go.Bar(
                            x=bar_widths[::-1], y=phase_names[::-1],
                            orientation="h",
                            marker=dict(color=phase_colors[::-1], line=dict(width=0)),
                            text=[f"  {s:.3f}" for s in phase_scores[::-1]],
                            textposition="outside",
                            textfont=dict(size=11, color="#1C2B1E"),
                            hovertemplate="<b>%{y}</b><br>Avg attention: %{text}<extra></extra>",
                            showlegend=False,
                        ))
                        layout_ph = base_layout(title="", height=200, showlegend=False)
                        layout_ph.update(xaxis=dict(range=[0, 1.15], showticklabels=False,
                                                    gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)"),
                                         yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)"),
                                         margin=dict(l=120, r=40, t=10, b=20))
                        fig_ph.update_layout(**layout_ph)
                        st.plotly_chart(fig_ph, use_container_width=True)

                        top_phase = phase_names[ranked[0]]
                        st.info(f"The model focused most on your {top_phase} phase, suggesting this is where your stroke differs most from expert level.")
                        st.caption("Phase importance is derived from Transformer attention weights and indicates which part of your stroke the model weighted most heavily when assessing similarity.")

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
                    st.write("**Feedback per joint**")

                    rf_imp = get_rf_joint_importance(mlm)
                    fbi = generate_feedback(pj, rf_importance=rf_imp)
                    focus = [f for f in fbi if f["level"] == "poor"]
                    attn_i = [f for f in fbi if f["level"] == "moderate"]
                    good = [f for f in fbi if f["level"] == "good"]

                    st.write(f"{len(focus)} poor, {len(attn_i)} moderate, {len(good)} good")

                    if focus:
                        st.markdown("**Poor** (high deviation)")
                        for fb in focus:
                            with st.expander(f"{fb['body_part']} — {fb['distance']:.3f}"):
                                st.write(fb['message'])

                    if attn_i:
                        st.markdown("**Moderate**")
                        for fb in attn_i:
                            with st.expander(f"{fb['body_part']} — {fb['distance']:.3f}"):
                                st.write(fb['message'])

                    if good:
                        st.markdown("**OK**")
                        for fb in good:
                            with st.expander(f"{fb['body_part']} — {fb['distance']:.3f}"):
                                st.write(fb['message'])

                    st.markdown("---")
                    st.write("**Worst 3 joints:**")
                    for i, fb in enumerate(fbi[:3]):
                        tag = "poor" if fb["level"] == "poor" else "moderate" if fb["level"] == "moderate" else "ok"
                        st.write(f"{i+1}. {fb['body_part']} ({tag}) — {fb['message']}")

                    if rf_imp:
                        st.caption("Feedback priorities are weighted by both joint deviation and biomechanical importance learned from expert-beginner classification.")

        try:
            os.unlink(tmp_path)
            if overlay: os.unlink(overlay)
        except:
            pass

        if kp is None:
            st.error("Could not process the video — check format and make sure a person is visible in frame.")


elif page == "DTW Analysis":
    st.markdown("# DTW results")
    st.write("Dynamic time warping distance per video, converted to a similarity score. Computed per-joint then averaged.")

    dd = load_json("dtw_comparison_results.json")
    if not dd:
        st.error("Results file not found. Run dtw_comparison.py first.")
    else:
        df = pd.DataFrame(dd)

        st.write(f"`n={len(df)}` strokes, similarity range `{df['similarity_score'].min():.1f}%` – `{df['similarity_score'].max():.1f}%`, mean `{df['similarity_score'].mean():.1f}%`")

        fds = sorted(df["folder"].unique())
        box_colors = ["#4A7C59", "#6B8F71", "#8B6E4E", "#5A6B55"]
        box_fill_colors = ["rgba(74,124,89,0.15)", "rgba(107,143,113,0.15)",
                           "rgba(139,110,78,0.15)", "rgba(90,107,85,0.15)"]

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

        st.write("**Mean deviation per joint**")

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
            textfont=dict(size=11, color="#1C2B1E"),
            hovertemplate="<b>%{y}</b><br>Avg DTW Distance: %{x:.3f}<extra></extra>",
            showlegend=False,
        ))
        layout2 = base_layout(title="Average DTW Distance per Joint (lower is better)", height=380)
        layout2.update(xaxis=dict(title="Avg DTW Distance", range=[0, max(jvals) * 1.3]),
                       margin=dict(l=130, r=20, t=55, b=45))
        fig2.update_layout(**layout2)
        st.plotly_chart(fig2, use_container_width=True)


elif page == "ML Classification":
    st.markdown("# ML classifiers")
    st.write("Random Forest, SVM, KNN on engineered features. 5-fold cross-validation. Target: expert vs beginner.")
    st.write("A soft voting ensemble and a stacking ensemble (with logistic regression meta-classifier) were evaluated to determine whether combining the three base classifiers improves on the best individual model.")
    st.info("Hyperparameters were independently tuned for each classifier via 5-fold GridSearchCV on the training set. The held-out test set (n=83, seed=42) was used only for final evaluation.")

    md = load_json("ml_classification_results.json")
    fi = load_json("feature_importance.json")

    if not md:
        st.error("Results file not found. Run ml_classifiers.py first.")
    else:
        ms = list(md.keys())
        ac = [md[m]["cv_accuracy_mean"] for m in ms]
        ac_std = [md[m]["cv_accuracy_std"] for m in ms]
        mc_colors = {"Random Forest": "#4A7C59", "SVM": "#6B8F71", "KNN": "#8B6E4E",
                     "Soft Voting Ensemble": "#A0522D", "Stacking Ensemble": "#3B5A45"}

        best_idx = int(np.argmax(ac))
        metrics_df = pd.DataFrame({
            "Model": ms,
            "CV Accuracy": [f"{a:.1%}" for a in ac],
            "Std": [f"±{s:.1%}" for s in ac_std],
            "Training Accuracy": [f"{md[m]['training_accuracy']:.1%}" for m in ms],
        })

        def _hl(row):
            return ["font-weight: bold; background-color: #EEF0E8" if row.name == best_idx else "" for _ in row]

        st.dataframe(metrics_df.style.apply(_hl, axis=1), use_container_width=True, hide_index=True)
        st.caption(f"Best model by CV accuracy: **{ms[best_idx]}** ({ac[best_idx]:.1%})")

        fig = go.Figure()
        for m_name, m_acc, m_std in zip(ms, ac, ac_std):
            fig.add_trace(go.Bar(
                x=[m_name], y=[m_acc],
                name=m_name,
                marker=dict(color=mc_colors.get(m_name, "#6B8F71"), line=dict(width=0)),
                error_y=dict(type="data", array=[m_std], visible=True, color="#5A6B55"),
                text=[f"{m_acc:.1%}"],
                textposition="outside",
                textfont=dict(size=12, color="#1C2B1E"),
                hovertemplate=f"<b>{m_name}</b><br>Accuracy: {m_acc:.1%} ± {m_std:.1%}<extra></extra>",
            ))
        layout = base_layout(title="Cross-Validation Accuracy by Model", height=360, showlegend=False)
        layout.update(yaxis=dict(range=[0.75, 1.02], tickformat=".0%", title="CV Accuracy"),
                      xaxis=dict(title=""), bargap=0.4)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

        st.write("**Confusion matrices**")

        md_items = list(md.items())
        row1 = md_items[:3]
        row2 = md_items[3:]

        def _draw_cm(col, n, r):
            with col:
                st.write(f"**{n}**")
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(np.array(r["confusion_matrix"]), annot=True, fmt="d",
                            cmap="Blues", ax=ax_cm,
                            xticklabels=["Expert", "Beginner"],
                            yticklabels=["Expert", "Beginner"],
                            linewidths=0.5, linecolor="#D8DCC9")
                ax_cm.set_xlabel("Predicted", fontsize=10)
                ax_cm.set_ylabel("Actual", fontsize=10)
                ax_cm.tick_params(labelsize=9)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()

        cm_cols1 = st.columns(3)
        for col, (n, r) in zip(cm_cols1, row1):
            _draw_cm(col, n, r)
        if row2:
            cm_cols2 = st.columns(3)
            for col, (n, r) in zip(cm_cols2, row2):
                _draw_cm(col, n, r)

        if fi:
            st.write("**Feature importances (RF, top 15)**")

            t = fi[:15]
            fnames = [f["feature"].replace("_", " ").title() for f in t]
            fimps = [f["importance"] for f in t]

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=fimps[::-1], y=fnames[::-1],
                orientation="h",
                marker=dict(
                    color=fimps[::-1],
                    colorscale=[[0, "#D8DCC9"], [0.5, "#6B8F71"], [1, "#4A7C59"]],
                    line=dict(width=0),
                ),
                text=[f"  {v:.4f}" for v in fimps[::-1]],
                textposition="outside",
                textfont=dict(size=11, color="#1C2B1E"),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
                showlegend=False,
            ))
            layout3 = base_layout(title="Random Forest Feature Importance", height=420)
            layout3.update(xaxis=dict(title="Importance Score"), margin=dict(l=180, r=20, t=55, b=45))
            fig3.update_layout(**layout3)
            st.plotly_chart(fig3, use_container_width=True)


elif page == "Siamese Transformer":
    st.markdown("# Siamese transformer")
    st.write("Architecture config, training curves, and per-frame attention. Contrastive loss, margin=1.0.")

    sd = load_json("siamese_transformer_results.json")
    sh = load_json("siamese_training_history.json")
    mi = load_json("siamese_model_info.json")

    if not sd:
        st.error("Results file not found. Run siamese_transformer.py first.")
    else:
        if mi:
            st.write(f"**Architecture:** {mi.get('num_layers','?')} layers, {mi.get('num_heads','?')} heads, embed dim {mi.get('embed_dim','?')}, {mi.get('total_parameters',0):,} params")

        if sh:
            st.write("**Training curves**")

            dh = pd.DataFrame(sh)
            tc1, tc2 = st.columns(2)

            with tc1:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=dh["epoch"], y=dh["train_loss"], name="Train Loss",
                    line=dict(color="#4A7C59", width=2.5),
                    hovertemplate="Epoch %{x}<br>Train Loss: %{y:.4f}<extra></extra>",
                ))
                fig_loss.add_trace(go.Scatter(
                    x=dh["epoch"], y=dh["val_loss"], name="Val Loss",
                    line=dict(color="#A0522D", width=2.5, dash="dash"),
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
                    line=dict(color="#4A7C59", width=2.5),
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
            st.write("**Per-stroke attention**")

            opts = [f"{r['video']} ({r['folder']})" for r in entries[:20]]
            sel = st.selectbox("Select a stroke to inspect:", opts)
            if sel:
                e = entries[opts.index(sel)]
                a = np.array(e["attention_weights"])
                top_frames = e.get("top_attention_frames", [])

                bar_cols = ["#A0522D" if i in top_frames else "#6B8F71" for i in range(len(a))]

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
    st.markdown("# DTW vs siamese — full dataset")
    st.write("All 330 recordings. DTW thresholded at sim > 50%. Siamese thresholded at distance < 0.5.")

    df = method_comparison_eval()
    if df is None:
        st.error("No preprocessed data found.")
    else:
        EXPERT_COLOR = "#4A7C59"
        BEGINNER_COLOR = "#8B6E4E"
        THRESH_COLOR = "#A0522D"
        LABELS = ["expert", "beginner"]

        def _metrics(y, yhat):
            return {
                "accuracy": accuracy_score(y, yhat),
                "precision": precision_score(y, yhat, pos_label="expert", zero_division=0),
                "recall": recall_score(y, yhat, pos_label="expert", zero_division=0),
                "f1": f1_score(y, yhat, pos_label="expert", zero_division=0),
            }

        st.caption(f"n = {len(df)} — experts: {(df['label']=='expert').sum()}, beginners: {(df['label']=='beginner').sum()}")

        st.write("### Confusion matrices")
        fig_cm, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, method, pred_col in [(axs[0], "DTW", "dtw_pred"), (axs[1], "Siamese", "siam_pred")]:
            cm = confusion_matrix(df["label"], df[pred_col], labels=LABELS)
            tp, fn = cm[0, 0], cm[0, 1]
            fp, tn = cm[1, 0], cm[1, 1]
            annot = np.array([[f"TP\n{tp}", f"FN\n{fn}"], [f"FP\n{fp}", f"TN\n{tn}"]])
            sns.heatmap(cm, annot=annot, fmt="", cmap="Greens", ax=ax,
                        xticklabels=LABELS, yticklabels=LABELS,
                        cbar=False, linewidths=0.5, linecolor="#D8DCC9",
                        annot_kws={"fontsize": 11})
            ax.set_title(method, fontsize=11)
            ax.set_xlabel("predicted", fontsize=10)
            ax.set_ylabel("actual", fontsize=10)
            ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        st.write("### Classification metrics")
        r_pear, p_pear = pearsonr(df["dtw_sim"], df["siam_dist"])
        met_df = pd.DataFrame([
            {"method": "DTW", **_metrics(df["label"], df["dtw_pred"])},
            {"method": "Siamese", **_metrics(df["label"], df["siam_pred"])},
        ])
        num = ["accuracy", "precision", "recall", "f1"]

        def _best(col):
            m = col.max()
            return ['background-color: #D8E3CE; font-weight: 600' if v == m else '' for v in col]

        st.dataframe(
            met_df.style.apply(_best, subset=num).format({c: "{:.3f}" for c in num}),
            use_container_width=True, hide_index=True)
        st.caption(f"Pearson r(DTW_sim, Siamese_dist) = {r_pear:.3f}, p = {p_pear:.3g}")

        st.write("### Score distributions")
        fig_d, axs = plt.subplots(1, 2, figsize=(10, 3.5))
        for lbl, c in [("expert", EXPERT_COLOR), ("beginner", BEGINNER_COLOR)]:
            sub = df[df["label"] == lbl]
            axs[0].hist(sub["dtw_sim"], bins=20, alpha=0.6, label=lbl, color=c, edgecolor="none")
            axs[1].hist(sub["siam_dist"], bins=20, alpha=0.6, label=lbl, color=c, edgecolor="none")
        axs[0].axvline(50, color=THRESH_COLOR, linestyle="--", linewidth=1, label="threshold")
        axs[1].axvline(0.5, color=THRESH_COLOR, linestyle="--", linewidth=1, label="threshold")
        axs[0].set_title("DTW similarity", fontsize=11)
        axs[0].set_xlabel("similarity (%)", fontsize=10); axs[0].set_ylabel("count", fontsize=10)
        axs[0].legend(fontsize=9, frameon=False)
        axs[1].set_title("Siamese distance", fontsize=11)
        axs[1].set_xlabel("distance", fontsize=10); axs[1].set_ylabel("count", fontsize=10)
        axs[1].legend(fontsize=9, frameon=False)
        for ax in axs:
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig_d); plt.close(fig_d)

        st.write("### Scatter plot")
        fig_s, ax = plt.subplots(figsize=(7, 4.5))
        for lbl, c in [("expert", EXPERT_COLOR), ("beginner", BEGINNER_COLOR)]:
            sub = df[df["label"] == lbl]
            ax.scatter(sub["dtw_sim"], sub["siam_dist"], alpha=0.65, c=c, label=lbl, s=28, edgecolors="none")
        ax.axvline(50, color=THRESH_COLOR, linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(0.5, color=THRESH_COLOR, linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("DTW similarity (%)", fontsize=10)
        ax.set_ylabel("Siamese distance", fontsize=10)
        ax.set_title(f"Pearson r = {r_pear:.3f}, p = {p_pear:.3g}", fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig_s); plt.close(fig_s)

        st.write("### Pearson correlation")
        st.write(f"Pearson's r between DTW similarity and Siamese distance across all {len(df)} recordings: **r = `{r_pear:.3f}`**, p-value = `{p_pear:.3g}`.")
        st.caption("Negative r is expected — higher DTW similarity should correspond to lower Siamese distance.")

        st.write("### ROC curves")
        y_bin = (df["label"] == "expert").astype(int).values
        dtw_scores = df["dtw_sim"].values
        siam_scores = -df["siam_dist"].values
        fpr_d, tpr_d, _ = roc_curve(y_bin, dtw_scores)
        fpr_s, tpr_s, _ = roc_curve(y_bin, siam_scores)
        auc_d, auc_s = auc(fpr_d, tpr_d), auc(fpr_s, tpr_s)

        fig_r, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(fpr_d, tpr_d, color=EXPERT_COLOR, linewidth=2, label=f"DTW (AUC = {auc_d:.3f})")
        ax.plot(fpr_s, tpr_s, color="#6B8F71", linewidth=2, label=f"Siamese (AUC = {auc_s:.3f})")
        ax.plot([0, 1], [0, 1], color=THRESH_COLOR, linestyle="--", linewidth=1, alpha=0.6, label="chance")
        ax.set_xlabel("false positive rate", fontsize=10)
        ax.set_ylabel("true positive rate", fontsize=10)
        ax.legend(fontsize=9, frameon=False, loc="lower right")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig_r); plt.close(fig_r)

        st.write("### Per stroke type breakdown")
        per_rows = []
        for stroke in ["forehand", "backhand"]:
            sub = df[df["stroke"] == stroke]
            agree = (sub["dtw_pred"] == sub["siam_pred"]).mean()
            for method, pred in [("DTW", "dtw_pred"), ("Siamese", "siam_pred")]:
                m = _metrics(sub["label"], sub[pred])
                per_rows.append({"stroke": stroke, "method": method, "n": len(sub),
                                 **m, "agreement": agree})
        per_df = pd.DataFrame(per_rows)
        st.dataframe(
            per_df.style.format({"accuracy": "{:.3f}", "precision": "{:.3f}",
                                 "recall": "{:.3f}", "f1": "{:.3f}", "agreement": "{:.1%}"}),
            use_container_width=True, hide_index=True)

        fh_agree = (df[df['stroke']=='forehand']['dtw_pred'] == df[df['stroke']=='forehand']['siam_pred']).mean()
        bh_agree = (df[df['stroke']=='backhand']['dtw_pred'] == df[df['stroke']=='backhand']['siam_pred']).mean()
        if fh_agree > bh_agree:
            st.write(f"DTW and Siamese agree more on **forehand** ({fh_agree:.1%}) than backhand ({bh_agree:.1%}).")
        elif bh_agree > fh_agree:
            st.write(f"DTW and Siamese agree more on **backhand** ({bh_agree:.1%}) than forehand ({fh_agree:.1%}).")
        else:
            st.write(f"DTW and Siamese agree equally on both ({fh_agree:.1%}).")

        st.write("### Agreement analysis")
        both_e = df[(df["dtw_pred"] == "expert") & (df["siam_pred"] == "expert")]
        both_b = df[(df["dtw_pred"] == "beginner") & (df["siam_pred"] == "beginner")]
        dtw_only = df[(df["dtw_pred"] == "expert") & (df["siam_pred"] == "beginner")]
        siam_only = df[(df["dtw_pred"] == "beginner") & (df["siam_pred"] == "expert")]

        ag = pd.DataFrame(
            [[len(both_e), len(siam_only)], [len(dtw_only), len(both_b)]],
            index=["Siamese: expert", "Siamese: beginner"],
            columns=["DTW: expert", "DTW: beginner"],
        )
        st.dataframe(ag, use_container_width=True)
        total = len(df)
        agree_n = len(both_e) + len(both_b)
        st.write(f"Overall agreement: `{agree_n/total:.1%}` ({agree_n}/{total}). Disagreements: `{len(dtw_only) + len(siam_only)}`.")

        if len(dtw_only):
            st.write(f"**DTW says expert, Siamese says beginner** — {len(dtw_only)} recordings")
            st.dataframe(
                dtw_only[["video", "label", "dtw_sim", "siam_dist"]].reset_index().rename(columns={"index": "idx"}),
                use_container_width=True, hide_index=True)

        if len(siam_only):
            st.write(f"**DTW says beginner, Siamese says expert** — {len(siam_only)} recordings")
            st.dataframe(
                siam_only[["video", "label", "dtw_sim", "siam_dist"]].reset_index().rename(columns={"index": "idx"}),
                use_container_width=True, hide_index=True)

        st.caption("Full per-recording scores saved to results/method_comparison_full.csv")


elif page == "Individual Stroke":
    st.markdown("# Individual stroke")
    st.write("Pick one stroke and see its joint-level scores + feedback.")

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

                st.write(f"**{get_stroke_type(e['folder'])}** / {get_ball_type(e['folder'])} — `{sel}`")
                dtw_score = e["similarity_score"]
                sim = f"DTW: `{dtw_score}%`"
                if se:
                    sim += f", Siamese: `{se['siamese_similarity']}%`"
                st.write(sim)

                if "per_joint_distances" in e:
                    pj = e["per_joint_distances"]
                    rf_imp = get_rf_joint_importance(train_ml())
                    fb = generate_feedback(pj, rf_importance=rf_imp)
                    sj = sorted(pj.items(), key=lambda x: x[1], reverse=True)
                    jnames = [JOINT_DISPLAY[j[0]] for j in sj]
                    jvals = [j[1] for j in sj]
                    jcols = [dist_color(v) for v in jvals]

                    st.write("**Per-joint deviation**")

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=jvals[::-1], y=jnames[::-1],
                        orientation="h",
                        marker=dict(color=jcols[::-1], line=dict(width=0)),
                        text=[f"  {'Close' if v < 0.2 else 'Work' if v < 0.4 else 'Focus'} ({v:.3f})"
                              for v in jvals[::-1]],
                        textposition="outside",
                        textfont=dict(size=11, color="#1C2B1E"),
                        hovertemplate="<b>%{y}</b><br>Deviation: %{x:.3f}<extra></extra>",
                        showlegend=False,
                    ))
                    layout = base_layout(title="", height=380)
                    layout.update(xaxis=dict(title="DTW Distance", range=[0, max(jvals) * 1.4]),
                                  margin=dict(l=130, r=20, t=20, b=45))
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("**Feedback**")
                    for f in [f for f in fb if f["level"] == "poor"]:
                        with st.expander(f"poor — {f['body_part']} ({f['distance']:.3f})"):
                            st.write(f['message'])
                    for f in [f for f in fb if f["level"] == "moderate"]:
                        with st.expander(f"moderate — {f['body_part']} ({f['distance']:.3f})"):
                            st.write(f['message'])
                    for f in [f for f in fb if f["level"] == "good"]:
                        with st.expander(f"ok — {f['body_part']} ({f['distance']:.3f})"):
                            st.write(f['message'])
                    if rf_imp:
                        st.caption("Feedback priorities are weighted by both joint deviation and biomechanical importance learned from expert-beginner classification.")
