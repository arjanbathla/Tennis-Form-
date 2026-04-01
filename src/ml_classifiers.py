"""
Tennis Stroke ML Classification
Trains RF, SVM, and KNN to distinguish expert from beginner technique,
then predicts skill level on personal recordings.
"""

import numpy as np
import json
import csv
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results"

JOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def compute_joint_angle(point_a, point_b, point_c):
    ba = point_a - point_b
    bc = point_c - point_b
    cos_angle = np.sum(ba * bc, axis=-1) / (
        np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    )
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def extract_features(keypoints):
    features = []
    feature_names = []

    features.append(len(keypoints))
    feature_names.append("sequence_length")

    for j, jn in enumerate(JOINT_NAMES):
        jd = keypoints[:, j, :]
        features.extend([np.mean(jd[:, 0]), np.mean(jd[:, 1]),
                         np.std(jd[:, 0]), np.std(jd[:, 1]),
                         np.ptp(jd[:, 0]), np.ptp(jd[:, 1]),
                         np.min(jd[:, 0]), np.max(jd[:, 0]),
                         np.min(jd[:, 1]), np.max(jd[:, 1])])
        feature_names.extend([f"{jn}_mean_x", f"{jn}_mean_y",
                               f"{jn}_std_x", f"{jn}_std_y",
                               f"{jn}_range_x", f"{jn}_range_y",
                               f"{jn}_min_x", f"{jn}_max_x",
                               f"{jn}_min_y", f"{jn}_max_y"])

    for j, jn in enumerate(JOINT_NAMES):
        jd = keypoints[:, j, :]
        if len(jd) > 1:
            v = np.linalg.norm(np.diff(jd, axis=0), axis=1)
            features.extend([np.mean(v), np.max(v), np.std(v)])
        else:
            features.extend([0.0, 0.0, 0.0])
        feature_names.extend([f"{jn}_vel_mean", f"{jn}_vel_max", f"{jn}_vel_std"])

    # shoulder-elbow-wrist, hip-knee-ankle, elbow-shoulder-hip (both sides)
    angle_defs = [
        ("left_elbow_angle", 0, 2, 4),
        ("right_elbow_angle", 1, 3, 5),
        ("left_knee_angle", 6, 8, 10),
        ("right_knee_angle", 7, 9, 11),
        ("left_shoulder_angle", 2, 0, 6),
        ("right_shoulder_angle", 3, 1, 7),
    ]
    for aname, a, b, c in angle_defs:
        angles = compute_joint_angle(keypoints[:, a, :], keypoints[:, b, :], keypoints[:, c, :])
        features.extend([np.mean(angles), np.std(angles), np.min(angles), np.max(angles), np.ptp(angles)])
        feature_names.extend([f"{aname}_mean", f"{aname}_std", f"{aname}_min", f"{aname}_max", f"{aname}_range"])

    if len(keypoints) > 1:
        def avg_vel(joints):
            return np.mean([np.mean(np.linalg.norm(np.diff(keypoints[:, j, :], axis=0), axis=1)) for j in joints])
        features.append(avg_vel([0,1,2,3,4,5]) / (avg_vel([6,7,8,9,10,11]) + 1e-8))
        feature_names.append("upper_lower_movement_ratio")
        features.append(avg_vel([0,2,4,6,8,10]) / (avg_vel([1,3,5,7,9,11]) + 1e-8))
        feature_names.append("left_right_symmetry")
    else:
        features.extend([0.0, 0.0])
        feature_names.extend(["upper_lower_movement_ratio", "left_right_symmetry"])

    return np.array(features), feature_names


def load_dataset(folder_path, label):
    folder = Path(folder_path)
    if not folder.exists():
        return [], [], [], []
    features_list, labels, names = [], [], []
    for f in sorted(folder.iterdir()):
        if f.name.endswith("_preprocessed.npy"):
            kp = np.load(str(f))
            feats, feat_names = extract_features(kp)
            features_list.append(feats)
            labels.append(label)
            names.append(f.stem)
    return features_list, labels, names, feat_names if features_list else []


def main():
    print("=" * 60)
    print("TENNIS STROKE ML CLASSIFICATION")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")
    all_features, all_labels, all_names, all_sources = [], [], [], []
    feat_names = None

    for stroke in ["forehand", "backhand"]:
        feats, labels, names, fn = load_dataset(PREPROCESSED_DIR / "thetis" / stroke, "expert")
        if feats:
            all_features.extend(feats); all_labels.extend(labels)
            all_names.extend(names)
            all_sources.extend([f"thetis_expert_{stroke}"] * len(feats))
            if feat_names is None:
                feat_names = fn

    expert_count = len(all_features)
    print(f"  THETIS expert samples: {expert_count}")

    for stroke in ["forehand", "backhand"]:
        feats, labels, names, _ = load_dataset(PREPROCESSED_DIR / "thetis_beginners" / stroke, "beginner")
        if feats:
            all_features.extend(feats); all_labels.extend(labels)
            all_names.extend(names)
            all_sources.extend([f"thetis_beginner_{stroke}"] * len(feats))

    beginner_count = len(all_features) - expert_count
    print(f"  THETIS beginner samples: {beginner_count}")

    if expert_count == 0 or beginner_count == 0:
        print("\nERROR: Need both expert and beginner data. Run extract_thetis_beginners.py and preprocess_thetis_beginners.py first.")
        return

    personal_folders = [
        "forehand_tennis_with_ball",
        "forehand_tennis_without_ball",
        "backhand_tennis_with_ball",
        "backhand_tennis_without_ball",
    ]
    personal_features, personal_names, personal_sources = [], [], []
    for folder_name in personal_folders:
        feats, _, names, _ = load_dataset(PREPROCESSED_DIR / folder_name, "personal")
        if feats:
            personal_features.extend(feats)
            personal_names.extend(names)
            personal_sources.extend([folder_name] * len(feats))

    print(f"  Personal samples (for prediction): {len(personal_features)}")

    X = np.nan_to_num(np.array(all_features), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(all_labels)

    print(f"\nTraining data: {X.shape}, expert={np.sum(y == 'expert')}, beginner={np.sum(y == 'beginner')}")

    print(f"\n{'=' * 50}")
    print("CLASSIFIER EVALUATION (5-fold cross-validation)")
    print(f"{'=' * 50}")
    
    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="euclidean",
        ),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_summary = {}
    trained_models = {}
    
    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")

        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        print(f"  CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
        print(f"  Per-fold: {[f'{s:.3f}' for s in scores]}")

        pipeline.fit(X, y)
        trained_models[name] = pipeline

        y_pred = pipeline.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        print(f"  Training accuracy: {report['accuracy']:.3f}")
        print(f"  Expert p/r: {report['expert']['precision']:.3f} / {report['expert']['recall']:.3f}")
        print(f"  Beginner p/r: {report['beginner']['precision']:.3f} / {report['beginner']['recall']:.3f}")

        cm = confusion_matrix(y, y_pred, labels=["expert", "beginner"])
        print(f"  Confusion matrix:")
        print(f"                 Predicted")
        print(f"                 Expert  Beginner")
        print(f"    Actual Expert   {cm[0][0]:>4}    {cm[0][1]:>4}")
        print(f"    Actual Beginner {cm[1][0]:>4}    {cm[1][1]:>4}")
        
        results_summary[name] = {
            "cv_accuracy_mean": round(float(scores.mean()), 4),
            "cv_accuracy_std": round(float(scores.std()), 4),
            "cv_scores": [round(float(s), 4) for s in scores],
            "training_accuracy": round(float(report["accuracy"]), 4),
            "confusion_matrix": cm.tolist(),
        }
    
    print(f"\n{'=' * 50}")
    print("TOP 15 MOST IMPORTANT FEATURES (Random Forest)")
    print(f"{'=' * 50}")

    rf_model = trained_models["Random Forest"].named_steps["classifier"]
    importances = rf_model.feature_importances_
    importance_indices = np.argsort(importances)[::-1]

    for rank, idx in enumerate(importance_indices[:15]):
        bar = "█" * int(importances[idx] * 200)
        print(f"  {rank+1:>2}. {feat_names[idx]:<35} {importances[idx]:.4f}  {bar}")
    
    if personal_features:
        print(f"\n{'=' * 50}")
        print("PREDICTIONS ON PERSONAL RECORDINGS")
        print(f"{'=' * 50}")

        X_personal = np.nan_to_num(np.array(personal_features), nan=0.0, posinf=0.0, neginf=0.0)
        personal_results = []

        for model_name, pipeline in trained_models.items():
            preds = pipeline.predict(X_personal)
            probs = pipeline.predict_proba(X_personal)
            print(f"\n--- {model_name} ---")
            print(f"  Expert: {np.sum(preds == 'expert')}/{len(preds)}, Beginner: {np.sum(preds == 'beginner')}/{len(preds)}")

            for folder_name in personal_folders:
                mask = [s == folder_name for s in personal_sources]
                fp = preds[mask]
                if len(fp) > 0:
                    pct = np.mean(fp == "expert") * 100
                    print(f"    {folder_name}: {pct:.0f}% expert")

            expert_idx = np.where(pipeline.classes_ == "expert")[0][0]
            for i, (name, source, pred) in enumerate(zip(personal_names, personal_sources, preds)):
                personal_results.append({
                    "video": name, "folder": source, "model": model_name,
                    "prediction": pred,
                    "expert_probability": round(float(probs[i][expert_idx]), 4),
                })

        pred_path = RESULTS_DIR / "ml_personal_predictions.json"
        with open(str(pred_path), "w") as f:
            json.dump(personal_results, f, indent=2)
        print(f"\nPersonal predictions saved to: {pred_path}")

    results_path = RESULTS_DIR / "ml_classification_results.json"
    with open(str(results_path), "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nClassification results saved to: {results_path}")

    imp_path = RESULTS_DIR / "feature_importance.json"
    with open(str(imp_path), "w") as f:
        json.dump([{"feature": feat_names[i], "importance": round(float(importances[i]), 6)}
                   for i in importance_indices], f, indent=2)
    print(f"Feature importance saved to: {imp_path}")

    print(f"\n{'=' * 60}")
    print("ML CLASSIFICATION COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
