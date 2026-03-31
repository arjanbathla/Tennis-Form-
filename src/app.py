"""
TennisForm — Tennis Stroke Analysis Dashboard
==============================================
- Manual stroke type selection (forehand/backhand)
- Uses DTW, Siamese Transformer, and ML classifiers for analysis
- Provides constructive coaching feedback
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
from pathlib import Path
from dtw import dtw
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# PATHS AND CONFIG
# ================================================================
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
MODEL_DIR = PROJECT_ROOT / "models"

mp_pose = mp_lib.solutions.pose
mp_drawing = mp_lib.solutions.drawing_utils
mp_drawing_styles = mp_lib.solutions.drawing_styles

JOINT_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
JOINT_DISPLAY = {
    "left_shoulder": "Left Shoulder", "right_shoulder": "Right Shoulder",
    "left_elbow": "Left Elbow", "right_elbow": "Right Elbow",
    "left_wrist": "Left Wrist", "right_wrist": "Right Wrist",
    "left_hip": "Left Hip", "right_hip": "Right Hip",
    "left_knee": "Left Knee", "right_knee": "Right Knee",
    "left_ankle": "Left Ankle", "right_ankle": "Right Ankle",
}
SELECTED_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
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
    "right_wrist": {
        "body_part": "Racket-Hand Wrist",
        "good": "Your racket-hand wrist movement closely matches professional technique. Good control through the swing.",
        "moderate": "Your wrist path is slightly different from the expert. Focus on keeping your wrist firm through contact and letting it naturally roll over during follow-through.",
        "poor": "Your wrist movement differs significantly from expert technique. This often means the racket face angle is inconsistent. Practice shadow swings focusing on a smooth, controlled wrist through the contact zone.",
    },
    "left_wrist": {
        "body_part": "Non-Racket Wrist",
        "good": "Good use of your non-racket arm for balance and momentum.",
        "moderate": "Your non-racket arm could be more active. Try pointing it toward the ball during preparation to help with shoulder turn and balance.",
        "poor": "Your non-racket arm is not contributing to the stroke. Professionals use it for balance and to initiate shoulder rotation. Practice pointing your free hand at the incoming ball.",
    },
    "right_elbow": {
        "body_part": "Racket-Arm Elbow",
        "good": "Your elbow position through the swing matches expert form well.",
        "moderate": "Your elbow is slightly out of position. Keep your elbow slightly bent and close to your body during the forward swing.",
        "poor": "Your elbow position differs a lot from experts. A common issue is a stiff, extended elbow. Practice keeping your elbow tucked and leading the swing with your forearm.",
    },
    "left_elbow": {
        "body_part": "Non-Racket Elbow",
        "good": "Good positioning of your non-racket elbow, helping with balance.",
        "moderate": "Your non-racket elbow could help more with balance. Let it extend naturally during preparation and tuck in during follow-through.",
        "poor": "Your non-racket arm is too passive. Extend it during the backswing to help rotate your shoulders, then pull it back to your body during the forward swing.",
    },
    "right_shoulder": {
        "body_part": "Racket-Side Shoulder",
        "good": "Excellent shoulder rotation matching professional technique.",
        "moderate": "Your shoulder rotation is slightly limited. Focus on turning your shoulders more during preparation.",
        "poor": "Insufficient shoulder rotation is limiting your power. Practice turning your hitting shoulder back further before swinging forward.",
    },
    "left_shoulder": {
        "body_part": "Non-Racket Shoulder",
        "good": "Good shoulder alignment and rotation through the stroke.",
        "moderate": "Your non-racket shoulder could lead the rotation more. Think about pulling your front shoulder back during preparation.",
        "poor": "Your shoulders are not rotating enough. The front shoulder should point toward the ball during preparation.",
    },
    "right_hip": {
        "body_part": "Right Hip",
        "good": "Strong hip rotation matching expert technique. This is where power comes from.",
        "moderate": "Your hip rotation is slightly limited. Focus on initiating the forward swing with your hips before your arm moves.",
        "poor": "Your hips are too static during the stroke. Professionals generate most of their power from hip rotation. Practice rotating your hips toward the target before your arm swings.",
    },
    "left_hip": {
        "body_part": "Left Hip",
        "good": "Good hip positioning and weight transfer.",
        "moderate": "Your hip positioning could be improved. Focus on transferring your weight from back foot to front foot during the swing.",
        "poor": "Your hips are not involved enough in the stroke. Practice stepping into the shot and rotating your hips toward the target.",
    },
    "right_knee": {
        "body_part": "Right Knee",
        "good": "Good knee bend and leg drive, similar to professional form.",
        "moderate": "You could benefit from slightly more knee bend. Staying lower gives you better balance.",
        "poor": "Your legs are too straight during the stroke. Bend your knees more and push up through the shot for power and stability.",
    },
    "left_knee": {
        "body_part": "Left Knee",
        "good": "Good knee positioning supporting your balance through the stroke.",
        "moderate": "Your front knee could be more engaged. Try bending it slightly as you step into the shot.",
        "poor": "Your front leg is too rigid. A slight bend in your front knee as you step forward helps absorb impact and keeps you balanced.",
    },
    "right_ankle": {
        "body_part": "Right Ankle",
        "good": "Good footwork and ankle positioning.",
        "moderate": "Your foot positioning is slightly off. Stay on the balls of your feet, not flat-footed.",
        "poor": "Your footwork needs attention. Stay on the balls of your feet and practice small adjustment steps before each shot.",
    },
    "left_ankle": {
        "body_part": "Left Ankle",
        "good": "Good foot placement and weight transfer through the front foot.",
        "moderate": "Your front foot placement could be more consistent. Step toward the target with your front foot.",
        "poor": "Your front foot is not stepping into the shot consistently. Practice stepping forward as you swing.",
    },
}


# ================================================================
# SIAMESE TRANSFORMER MODEL
# ================================================================
class PositionalEncoding(torch.nn.Module):
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

class StrokeTransformerEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = torch.nn.Linear(INPUT_DIM, EMBED_DIM)
        self.pos_encoder = PositionalEncoding(EMBED_DIM, max_len=MAX_SEQ_LENGTH + 10)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM * 4, dropout=DROPOUT, batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.output_projection = torch.nn.Sequential(
            torch.nn.Linear(EMBED_DIM, EMBEDDING_SIZE), torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT), torch.nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE),
        )
        self.attention_weights = None
        for layer in self.transformer_encoder.layers:
            layer.self_attn.need_weights = True
            layer.self_attn.average_attn_weights = True
            layer.self_attn.register_forward_hook(self._attn_hook)

    def _attn_hook(self, module, input, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            self.attention_weights = output[1].detach().cpu()

    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask) if mask is not None else self.transformer_encoder(x)
        if mask is not None:
            m = (~mask).unsqueeze(-1).float()
            x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return self.output_projection(x)

class SiameseTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StrokeTransformerEncoder()
    def forward(self, x1, x2, mask1=None, mask2=None):
        e1, e2 = self.encoder(x1, mask1), self.encoder(x2, mask2)
        return torch.sqrt(torch.sum((e1 - e2) ** 2, dim=1) + 1e-8), e1, e2
    def get_embedding(self, x, mask=None):
        return self.encoder(x, mask)


# ================================================================
# PROCESSING FUNCTIONS
# ================================================================
def process_uploaded_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
    all_keypoints = []
    detected = 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=2,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                all_keypoints.append([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])
                detected += 1
            else:
                all_keypoints.append([[0.0, 0.0]] * 33)
            out.write(frame)
    cap.release()
    out.release()

    if not all_keypoints:
        return None, None, None
    return np.array(all_keypoints), output_path, {
        "fps": fps, "total_frames": total_frames, "detected_frames": detected,
        "detection_rate": round(detected / total_frames * 100, 1) if total_frames > 0 else 0,
    }


def preprocess_keypoints(keypoints):
    sums = np.sum(np.abs(keypoints), axis=(1, 2))
    keypoints = keypoints[sums > 0]
    if len(keypoints) < 5:
        return None
    hip_mid = (keypoints[:, LEFT_HIP, :] + keypoints[:, RIGHT_HIP, :]) / 2.0
    keypoints = keypoints - hip_mid[:, np.newaxis, :]
    sh_mid = (keypoints[:, LEFT_SHOULDER, :] + keypoints[:, RIGHT_SHOULDER, :]) / 2.0
    hp_mid = (keypoints[:, LEFT_HIP, :] + keypoints[:, RIGHT_HIP, :]) / 2.0
    torso = np.mean(np.linalg.norm(sh_mid - hp_mid, axis=1))
    if torso < 0.01:
        return None
    keypoints = keypoints / torso
    keypoints = keypoints[:, SELECTED_JOINTS, :]
    smoothed = np.copy(keypoints).astype(float)
    half = SMOOTHING_WINDOW // 2
    for i in range(len(keypoints)):
        s, e = max(0, i - half), min(len(keypoints), i + half + 1)
        smoothed[i] = np.mean(keypoints[s:e], axis=0)
    return smoothed


def compare_against_experts(preprocessed, stroke_type):
    folder = PREPROCESSED_DIR / "thetis" / stroke_type
    if not folder.exists():
        return None, None
    files = sorted([f for f in folder.iterdir() if f.name.endswith("_preprocessed.npy")])
    if len(files) > 15:
        files = [files[i] for i in np.random.choice(len(files), 15, replace=False)]

    overall_dists = []
    per_joint = {j: [] for j in JOINT_NAMES}
    flat_p = preprocessed.reshape(len(preprocessed), -1)

    for ef in files:
        expert = np.load(str(ef))
        flat_e = expert.reshape(len(expert), -1)
        alignment = dtw(flat_p, flat_e, keep_internals=False)
        overall_dists.append(alignment.normalizedDistance)
        for j, jn in enumerate(JOINT_NAMES):
            ja = dtw(preprocessed[:, j, :], expert[:, j, :], keep_internals=False)
            per_joint[jn].append(ja.normalizedDistance)

    avg_overall = np.mean(overall_dists)
    avg_pj = {j: np.mean(d) for j, d in per_joint.items()}
    return {
        "overall_distance": round(float(avg_overall), 4),
        "similarity_score": round(100 * np.exp(-avg_overall), 1),
        "per_joint_distances": {k: round(float(v), 4) for k, v in avg_pj.items()},
    }, avg_pj


def pad_single(seq):
    flat = seq.reshape(len(seq), -1)
    sl = len(flat)
    if sl >= MAX_SEQ_LENGTH:
        return torch.FloatTensor(flat[:MAX_SEQ_LENGTH]).unsqueeze(0), torch.BoolTensor([False] * MAX_SEQ_LENGTH).unsqueeze(0)
    pad = np.zeros((MAX_SEQ_LENGTH - sl, flat.shape[1]))
    padded = np.vstack([flat, pad])
    mask = [False] * sl + [True] * (MAX_SEQ_LENGTH - sl)
    return torch.FloatTensor(padded).unsqueeze(0), torch.BoolTensor(mask).unsqueeze(0)


@st.cache_resource
def load_siamese_model():
    model_path = MODEL_DIR / "siamese_transformer_best.pth"
    if not model_path.exists():
        return None
    model = SiameseTransformer()
    model.load_state_dict(torch.load(str(model_path), map_location="cpu", weights_only=True))
    model.eval()
    return model


def siamese_compare(model, preprocessed, stroke_type):
    folder = PREPROCESSED_DIR / "thetis" / stroke_type
    if not folder.exists() or model is None:
        return None, None

    files = sorted([f for f in folder.iterdir() if f.name.endswith("_preprocessed.npy")])
    if len(files) > 10:
        files = [files[i] for i in np.random.choice(len(files), 10, replace=False)]

    distances = []
    p_pad, p_mask = pad_single(preprocessed)
    for ef in files:
        expert = np.load(str(ef))
        e_pad, e_mask = pad_single(expert)
        with torch.no_grad():
            dist, _, _ = model(p_pad, e_pad, p_mask, e_mask)
            distances.append(dist.item())

    avg_dist = np.mean(distances)
    similarity = round(100 * np.exp(-avg_dist), 1)

    attention = None
    with torch.no_grad():
        _ = model.get_embedding(p_pad, p_mask)
    attn = model.encoder.attention_weights
    if attn is not None:
        attn_avg = attn[0].mean(dim=0).numpy()
        sl = int((~p_mask[0]).sum())
        frame_imp = attn_avg[:sl, :sl].sum(axis=0)
        frame_imp = frame_imp / frame_imp.max()
        attention = frame_imp

    return {"siamese_similarity": similarity, "siamese_distance": round(float(avg_dist), 4)}, attention


def compute_joint_angle(a, b, c):
    ba, bc = a - b, c - b
    cos = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def extract_features(kp):
    features = [len(kp)]
    for j in range(12):
        jd = kp[:, j, :]
        features.extend([np.mean(jd[:,0]), np.mean(jd[:,1]), np.std(jd[:,0]), np.std(jd[:,1]),
                         np.ptp(jd[:,0]), np.ptp(jd[:,1]),
                         np.min(jd[:,0]), np.max(jd[:,0]), np.min(jd[:,1]), np.max(jd[:,1])])
    for j in range(12):
        jd = kp[:, j, :]
        if len(jd) > 1:
            vel = np.linalg.norm(np.diff(jd, axis=0), axis=1)
            features.extend([np.mean(vel), np.max(vel), np.std(vel)])
        else:
            features.extend([0, 0, 0])
    for a, b, c in [(0,2,4),(1,3,5),(6,8,10),(7,9,11),(2,0,6),(3,1,7)]:
        ang = compute_joint_angle(kp[:,a,:], kp[:,b,:], kp[:,c,:])
        features.extend([np.mean(ang), np.std(ang), np.min(ang), np.max(ang), np.ptp(ang)])
    if len(kp) > 1:
        upper = np.mean([np.mean(np.linalg.norm(np.diff(kp[:,j,:],axis=0),axis=1)) for j in range(6)])
        lower = np.mean([np.mean(np.linalg.norm(np.diff(kp[:,j,:],axis=0),axis=1)) for j in range(6,12)])
        features.append(upper / (lower + 1e-8))
        left = np.mean([np.mean(np.linalg.norm(np.diff(kp[:,j,:],axis=0),axis=1)) for j in [0,2,4,6,8,10]])
        right = np.mean([np.mean(np.linalg.norm(np.diff(kp[:,j,:],axis=0),axis=1)) for j in [1,3,5,7,9,11]])
        features.append(left / (right + 1e-8))
    else:
        features.extend([0, 0])
    return np.array(features)


@st.cache_resource
def train_ml_classifiers():
    expert_features, beginner_features = [], []
    for stroke in ["forehand", "backhand"]:
        for level, container in [("thetis", expert_features), ("thetis_beginners", beginner_features)]:
            folder = PREPROCESSED_DIR / level / stroke
            if not folder.exists():
                continue
            for f in folder.iterdir():
                if f.name.endswith("_preprocessed.npy"):
                    container.append(extract_features(np.load(str(f))))
    if not expert_features or not beginner_features:
        return None
    X = np.nan_to_num(np.array(expert_features + beginner_features))
    y = np.array(["expert"] * len(expert_features) + ["beginner"] * len(beginner_features))
    models = {}
    for name, clf in [
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")),
        ("SVM", SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42)),
        ("KNN", KNeighborsClassifier(n_neighbors=5, weights="distance")),
    ]:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X, y)
        models[name] = pipe
    return models


def ml_predict(models, preprocessed):
    if models is None:
        return None
    feats = np.nan_to_num(extract_features(preprocessed).reshape(1, -1))
    results = {}
    for name, pipe in models.items():
        pred = pipe.predict(feats)[0]
        prob = pipe.predict_proba(feats)[0]
        expert_idx = np.where(pipe.classes_ == "expert")[0][0]
        results[name] = {"prediction": pred, "expert_probability": round(float(prob[expert_idx]), 4)}
    return results


def generate_feedback(per_joint):
    sorted_j = sorted(per_joint.items(), key=lambda x: x[1], reverse=True)
    items = []
    for jn, dist in sorted_j:
        fb = JOINT_FEEDBACK.get(jn)
        if not fb:
            continue
        if dist < 0.2: level, icon = "good", "✅"
        elif dist < 0.4: level, icon = "moderate", "⚠️"
        else: level, icon = "poor", "🔴"
        items.append({"joint": jn, "body_part": fb["body_part"], "distance": dist,
                      "level": level, "icon": icon, "message": fb[level]})
    return items


# ================================================================
# DATA LOADING
# ================================================================
@st.cache_data
def load_json(filename):
    path = RESULTS_DIR / filename
    if path.exists():
        with open(str(path)) as f:
            return json.load(f)
    return None

def get_stroke_type(folder):
    return "Forehand" if "forehand" in folder else "Backhand" if "backhand" in folder else "Unknown"
def get_ball_type(folder):
    return "Shadow Swing" if "without_ball" in folder else "With Ball" if "with_ball" in folder else "Unknown"


# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(page_title="TennisForm", page_icon="🎾", layout="wide")

st.sidebar.title("🎾 TennisForm")
page = st.sidebar.radio("Navigate", [
    "Overview", "Analyse My Stroke", "DTW Analysis",
    "ML Classification", "Siamese Transformer", "Method Comparison", "Individual Stroke",
])
st.sidebar.markdown("---")
st.sidebar.markdown("Final Year Project — NTU")


# ================================================================
# OVERVIEW
# ================================================================
if page == "Overview":
    st.title("TennisForm — Tennis Stroke Analysis")
    st.markdown("Compare your tennis technique against professional players using pose estimation and machine learning.")

    dtw_data = load_json("dtw_comparison_results.json")
    siam_data = load_json("siamese_transformer_results.json")
    ml_data = load_json("ml_classification_results.json")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Videos Analysed", "81" if dtw_data else "N/A")
    with c2:
        if dtw_data: st.metric("Avg DTW Similarity", f"{np.mean([r['similarity_score'] for r in dtw_data]):.1f}%")
    with c3:
        if siam_data: st.metric("Avg Siamese Similarity", f"{np.mean([r['siamese_similarity'] for r in siam_data]):.1f}%")
    with c4:
        if ml_data: st.metric("Best ML Accuracy", f"{max(v['cv_accuracy_mean'] for v in ml_data.values()):.1%}")

    st.markdown("---")
    st.markdown("Go to **Analyse My Stroke** to upload a video and get personalised feedback.")


# ================================================================
# ANALYSE MY STROKE
# ================================================================
elif page == "Analyse My Stroke":
    st.title("Analyse My Stroke")
    st.markdown("Upload a video of your tennis stroke to get personalised coaching feedback.")

    col_upload, col_tips = st.columns([2, 1])

    with col_tips:
        stroke_type = st.radio("What stroke is this?", ["forehand", "backhand"])
        st.markdown("---")
        st.markdown("**Tips for best results:**")
        st.markdown("- Film from the front, facing the player")
        st.markdown("- Keep full body in frame head to feet")
        st.markdown("- Good lighting, clean background")
        st.markdown("- One stroke per video")
        st.markdown("- Match the THETIS camera angle (front-on)")

    with col_upload:
        uploaded_file = st.file_uploader("Upload your stroke video", type=["mov", "mp4", "avi"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.markdown("---")

        with st.spinner("Extracting pose data from your video..."):
            keypoints, overlay_path, video_info = process_uploaded_video(tmp_path)

        if keypoints is not None and overlay_path is not None:
            col_orig, col_skel = st.columns(2)
            with col_orig:
                st.subheader("Your Video")
                st.video(tmp_path)
            with col_skel:
                st.subheader("Skeleton Overlay")
                st.video(overlay_path)

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Frames", video_info["total_frames"])
            with c2: st.metric("Pose Detected", f"{video_info['detection_rate']}%")
            with c3: st.metric("FPS", f"{video_info['fps']:.0f}")

            st.markdown("---")

            preprocessed = preprocess_keypoints(keypoints)

            if preprocessed is not None:
                st.subheader(f"Stroke Type: {stroke_type.title()}")

                # === DTW ===
                with st.spinner("Running DTW comparison..."):
                    dtw_results, per_joint = compare_against_experts(preprocessed, stroke_type)

                # === SIAMESE ===
                siamese_model = load_siamese_model()
                siam_results, attention = None, None
                if siamese_model:
                    with st.spinner("Running Siamese Transformer..."):
                        siam_results, attention = siamese_compare(siamese_model, preprocessed, stroke_type)

                # === ML ===
                ml_models = train_ml_classifiers()
                ml_results = ml_predict(ml_models, preprocessed) if ml_models else None

                # ---- SCORES ----
                if dtw_results:
                    st.markdown("---")
                    st.subheader("Similarity Scores")

                    score_cols = st.columns(3)
                    with score_cols[0]:
                        st.metric("DTW Similarity", f"{dtw_results['similarity_score']}%")
                    with score_cols[1]:
                        if siam_results:
                            st.metric("Siamese Similarity", f"{siam_results['siamese_similarity']}%")
                        else:
                            st.metric("Siamese Similarity", "N/A")
                    with score_cols[2]:
                        if ml_results:
                            expert_votes = sum(1 for m in ml_results.values() if m["prediction"] == "expert")
                            st.metric("ML Verdict", f"{expert_votes}/3 say Expert")
                        else:
                            st.metric("ML Verdict", "N/A")

                    # Score bars
                    dtw_score = dtw_results["similarity_score"]
                    fig, ax = plt.subplots(figsize=(10, 1.8))
                    color_dtw = "#4CAF50" if dtw_score >= 40 else "#FF9800" if dtw_score >= 25 else "#f44336"
                    ax.barh(["DTW"], [dtw_score], color=color_dtw, height=0.4)
                    ax.barh(["DTW"], [100 - dtw_score], left=[dtw_score], color="#e0e0e0", height=0.4)
                    if siam_results:
                        ss = siam_results["siamese_similarity"]
                        color_s = "#4CAF50" if ss >= 40 else "#FF9800" if ss >= 25 else "#f44336"
                        ax.barh(["Siamese"], [ss], color=color_s, height=0.4)
                        ax.barh(["Siamese"], [100 - ss], left=[ss], color="#e0e0e0", height=0.4)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Similarity to Expert (%)")
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    # ML detail
                    if ml_results:
                        st.markdown("**ML Classification:**")
                        ml_rows = [{"Model": n, "Prediction": r["prediction"].title(),
                                   "Expert Probability": f"{r['expert_probability']:.1%}"}
                                  for n, r in ml_results.items()]
                        st.dataframe(pd.DataFrame(ml_rows), use_container_width=True, hide_index=True)

                    st.markdown("---")

                    # Per-joint chart
                    st.subheader("Body Part Breakdown")
                    sorted_joints = sorted(per_joint.items(), key=lambda x: x[1], reverse=True)
                    fig, ax = plt.subplots(figsize=(12, 5))
                    jn = [JOINT_DISPLAY[j[0]] for j in sorted_joints]
                    jv = [j[1] for j in sorted_joints]
                    jc = ["#f44336" if v > 0.4 else "#FF9800" if v > 0.3 else "#4CAF50" for v in jv]
                    bars = ax.barh(jn[::-1], jv[::-1], color=jc[::-1])
                    ax.set_xlabel("Deviation from Expert (lower is better)")
                    for bar, val in zip(bars, jv[::-1]):
                        lbl = "Close to expert" if val < 0.2 else "Needs work" if val < 0.4 else "Focus area"
                        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                               f"{val:.3f} — {lbl}", va="center", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    # Attention
                    if attention is not None:
                        st.subheader("Transformer Attention — Which Frames Matter Most")
                        fig, ax = plt.subplots(figsize=(14, 3))
                        ax.bar(range(len(attention)), attention, color="#2196F3", alpha=0.7)
                        top_frames = np.argsort(attention)[-5:][::-1]
                        for f in top_frames:
                            ax.bar(f, attention[f], color="#f44336")
                        ax.set_xlabel("Frame Number")
                        ax.set_ylabel("Attention Weight")
                        ax.set_title("Red bars = frames the model considers most important")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                    st.markdown("---")

                    # Feedback
                    st.subheader("Your Personalised Coaching Feedback")
                    feedback_items = generate_feedback(per_joint)

                    focus = [f for f in feedback_items if f["level"] == "poor"]
                    attn_items = [f for f in feedback_items if f["level"] == "moderate"]
                    good = [f for f in feedback_items if f["level"] == "good"]

                    sc1, sc2, sc3 = st.columns(3)
                    with sc1: st.metric("🔴 Focus Areas", len(focus))
                    with sc2: st.metric("⚠️ Needs Attention", len(attn_items))
                    with sc3: st.metric("✅ Doing Well", len(good))

                    if focus:
                        st.markdown("### 🔴 Priority Areas to Improve")
                        st.markdown("These body parts differ most from professional technique. Work on these first.")
                        for fb in focus:
                            with st.expander(f"🔴 {fb['body_part']} — deviation: {fb['distance']:.3f}"):
                                st.markdown(fb["message"])

                    if attn_items:
                        st.markdown("### ⚠️ Areas That Need Some Attention")
                        for fb in attn_items:
                            with st.expander(f"⚠️ {fb['body_part']} — deviation: {fb['distance']:.3f}"):
                                st.markdown(fb["message"])

                    if good:
                        st.markdown("### ✅ What You're Doing Well")
                        for fb in good:
                            with st.expander(f"✅ {fb['body_part']} — deviation: {fb['distance']:.3f}"):
                                st.markdown(fb["message"])

                    st.markdown("---")
                    st.subheader("Your Top 3 Tips")
                    for i, fb in enumerate(feedback_items[:3]):
                        st.markdown(f"**{i+1}. {fb['body_part']}:** {fb['message']}")

            try:
                os.unlink(tmp_path)
                if overlay_path: os.unlink(overlay_path)
            except:
                pass
        else:
            st.error("Could not process the video. Try a clip with better lighting.")


# ================================================================
# DTW ANALYSIS
# ================================================================
elif page == "DTW Analysis":
    st.title("DTW Analysis")
    dtw_data = load_json("dtw_comparison_results.json")
    if not dtw_data:
        st.error("DTW results not found.")
    else:
        df = pd.DataFrame(dtw_data)
        st.subheader("Similarity by Recording Type")
        fig, ax = plt.subplots(figsize=(12, 5))
        folders = sorted(df["folder"].unique())
        colors = ["#2196F3", "#64B5F6", "#FF9800", "#FFB74D"]
        bp = ax.boxplot([df[df["folder"]==f]["similarity_score"].values for f in folders],
                       labels=[f.replace("_tennis_","\n").replace("_"," ") for f in folders], patch_artist=True)
        for p, c in zip(bp["boxes"], colors): p.set_facecolor(c)
        ax.set_ylabel("Similarity (%)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Per-Joint Distances")
        ja = {j: np.mean([r["per_joint_distances"][j] for r in dtw_data]) for j in JOINT_NAMES}
        sj = sorted(ja.items(), key=lambda x: x[1], reverse=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        v = [j[1] for j in sj]
        bc = ["#f44336" if x > 0.4 else "#FF9800" if x > 0.3 else "#4CAF50" for x in v]
        ax.barh([JOINT_DISPLAY[j[0]] for j in sj][::-1], v[::-1], color=bc[::-1])
        ax.set_xlabel("Average DTW Distance")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ================================================================
# ML CLASSIFICATION
# ================================================================
elif page == "ML Classification":
    st.title("ML Classification")
    ml_data = load_json("ml_classification_results.json")
    fi = load_json("feature_importance.json")
    if not ml_data:
        st.error("ML results not found.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ms = list(ml_data.keys())
        ac = [ml_data[m]["cv_accuracy_mean"] for m in ms]
        mc = {"Random Forest": "#4CAF50", "SVM": "#2196F3", "KNN": "#FF9800"}
        bars = ax.bar(ms, ac, color=[mc[m] for m in ms], yerr=[ml_data[m]["cv_accuracy_std"] for m in ms], capsize=10)
        ax.set_ylim(0.8, 1.0); ax.set_ylabel("Accuracy")
        for b, a in zip(bars, ac):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{a:.1%}", ha="center", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (n, r) in zip(axes, ml_data.items()):
            sns.heatmap(np.array(r["confusion_matrix"]), annot=True, fmt="d", cmap="Blues",
                       xticklabels=["Expert","Beginner"], yticklabels=["Expert","Beginner"], ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(n)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if fi:
            st.subheader("Top 15 Features")
            t = fi[:15]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh([f["feature"].replace("_"," ").title() for f in t][::-1], [f["importance"] for f in t][::-1], color="#4CAF50")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ================================================================
# SIAMESE TRANSFORMER
# ================================================================
elif page == "Siamese Transformer":
    st.title("Siamese Transformer")
    sd = load_json("siamese_transformer_results.json")
    sh = load_json("siamese_training_history.json")
    mi = load_json("siamese_model_info.json")
    if not sd:
        st.error("Results not found.")
    else:
        if mi:
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("Layers", mi.get("num_layers"))
            with c2: st.metric("Heads", mi.get("num_heads"))
            with c3: st.metric("Embed Dim", mi.get("embed_dim"))
            with c4: st.metric("Parameters", f"{mi.get('total_parameters',0):,}")
        if sh:
            dh = pd.DataFrame(sh)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot(dh["epoch"], dh["train_loss"], label="Train", color="#2196F3")
            axes[0].plot(dh["epoch"], dh["val_loss"], label="Val", color="#f44336")
            axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
            axes[1].plot(dh["epoch"], dh["val_accuracy"], color="#4CAF50")
            axes[1].set_ylim(0.5,1.0); axes[1].grid(alpha=0.3); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        entries = [r for r in sd if "attention_weights" in r]
        if entries:
            st.subheader("Attention Viewer")
            opts = [f"{r['video']} ({r['folder']})" for r in entries[:20]]
            sel = st.selectbox("Select:", opts)
            if sel:
                e = entries[opts.index(sel)]
                a = np.array(e["attention_weights"])
                fig, ax = plt.subplots(figsize=(14, 3))
                ax.bar(range(len(a)), a, color="#2196F3", alpha=0.7)
                if "top_attention_frames" in e:
                    for f in e["top_attention_frames"]:
                        if f < len(a): ax.bar(f, a[f], color="#f44336")
                ax.set_xlabel("Frame"); ax.set_ylabel("Attention")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()


# ================================================================
# METHOD COMPARISON
# ================================================================
elif page == "Method Comparison":
    st.title("Method Comparison")
    dd = load_json("dtw_comparison_results.json")
    sd = load_json("siamese_transformer_results.json")
    if dd and sd:
        folders = sorted(set(r["folder"] for r in dd))
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(folders)); w = 0.35
        dv = [np.mean([r["similarity_score"] for r in dd if r["folder"]==f]) for f in folders]
        sv = [np.mean([r["siamese_similarity"] for r in sd if r["folder"]==f]) for f in folders]
        ax.bar(x-w/2, dv, w, label="DTW", color="#2196F3")
        ax.bar(x+w/2, sv, w, label="Siamese", color="#FF9800")
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace("_tennis_","\n").replace("_"," ") for f in folders], fontsize=8)
        ax.set_ylabel("Similarity (%)"); ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        rows = []
        for f in folders:
            d = np.mean([r["similarity_score"] for r in dd if r["folder"]==f])
            s = np.mean([r["siamese_similarity"] for r in sd if r["folder"]==f])
            rows.append({"Type": f.replace("_"," ").title(), "DTW": f"{d:.1f}%", "Siamese": f"{s:.1f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ================================================================
# INDIVIDUAL STROKE
# ================================================================
elif page == "Individual Stroke":
    st.title("Individual Stroke")
    dd = load_json("dtw_comparison_results.json")
    sd = load_json("siamese_transformer_results.json")
    if dd:
        videos = sorted(set(r["video"] for r in dd))
        sel = st.selectbox("Select recording:", videos)
        if sel:
            e = next((r for r in dd if r["video"]==sel), None)
            if e:
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("DTW Similarity", f"{e['similarity_score']}%")
                    st.markdown(f"**Stroke:** {get_stroke_type(e['folder'])}")
                with c2:
                    if sd:
                        se = next((r for r in sd if r["video"]==sel), None)
                        if se: st.metric("Siamese Similarity", f"{se['siamese_similarity']}%")

                if "per_joint_distances" in e:
                    pj = e["per_joint_distances"]
                    fb = generate_feedback(pj)
                    sj = sorted(pj.items(), key=lambda x: x[1], reverse=True)
                    fig, ax = plt.subplots(figsize=(12, 5))
                    v = [j[1] for j in sj]
                    bc = ["#f44336" if x>0.4 else "#FF9800" if x>0.3 else "#4CAF50" for x in v]
                    ax.barh([JOINT_DISPLAY[j[0]] for j in sj][::-1], v[::-1], color=bc[::-1])
                    ax.set_xlabel("Deviation from Expert")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.subheader("Coaching Feedback")
                    for f in [f for f in fb if f["level"]=="poor"]:
                        with st.expander(f"🔴 {f['body_part']}"): st.markdown(f["message"])
                    for f in [f for f in fb if f["level"]=="moderate"]:
                        with st.expander(f"⚠️ {f['body_part']}"): st.markdown(f["message"])
                    for f in [f for f in fb if f["level"]=="good"]:
                        with st.expander(f"✅ {f['body_part']}"): st.markdown(f["message"])
