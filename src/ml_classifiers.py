"""
Tennis Stroke ML Classification
Trains Random Forest, SVM, and KNN classifiers to distinguish
expert from beginner technique using features extracted from
preprocessed keypoint sequences.

Then applies trained models to personal recordings to predict
skill level.
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

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Joint names matching preprocessing output
JOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def compute_joint_angle(point_a, point_b, point_c):
    """
    Compute angle at point_b formed by points a-b-c.
    Returns angle in degrees.
    """
    ba = point_a - point_b
    bc = point_c - point_b
    
    cos_angle = np.sum(ba * bc, axis=-1) / (
        np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def extract_features(keypoints):
    """
    Extract a fixed-length feature vector from a preprocessed keypoint sequence.
    
    Args:
        keypoints: numpy array of shape (frames, 12, 2)
    
    Returns:
        features: 1D numpy array of features
        feature_names: list of feature names
    """
    features = []
    feature_names = []
    
    num_frames = len(keypoints)
    
    # Feature 1: Sequence length (normalised)
    features.append(num_frames)
    feature_names.append("sequence_length")
    
    # Feature group 2: Per-joint position statistics
    for j, joint_name in enumerate(JOINT_NAMES):
        joint_data = keypoints[:, j, :]  # (frames, 2)
        
        # Mean position
        features.extend([np.mean(joint_data[:, 0]), np.mean(joint_data[:, 1])])
        feature_names.extend([f"{joint_name}_mean_x", f"{joint_name}_mean_y"])
        
        # Std of position
        features.extend([np.std(joint_data[:, 0]), np.std(joint_data[:, 1])])
        feature_names.extend([f"{joint_name}_std_x", f"{joint_name}_std_y"])
        
        # Range of motion
        features.extend([
            np.ptp(joint_data[:, 0]),  # max - min for x
            np.ptp(joint_data[:, 1]),  # max - min for y
        ])
        feature_names.extend([f"{joint_name}_range_x", f"{joint_name}_range_y"])
        
        # Min and max positions
        features.extend([
            np.min(joint_data[:, 0]), np.max(joint_data[:, 0]),
            np.min(joint_data[:, 1]), np.max(joint_data[:, 1]),
        ])
        feature_names.extend([
            f"{joint_name}_min_x", f"{joint_name}_max_x",
            f"{joint_name}_min_y", f"{joint_name}_max_y",
        ])
    
    # Feature group 3: Per-joint velocity (frame-to-frame displacement)
    for j, joint_name in enumerate(JOINT_NAMES):
        joint_data = keypoints[:, j, :]
        
        if len(joint_data) > 1:
            velocities = np.linalg.norm(np.diff(joint_data, axis=0), axis=1)
            features.extend([
                np.mean(velocities),
                np.max(velocities),
                np.std(velocities),
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        feature_names.extend([
            f"{joint_name}_vel_mean",
            f"{joint_name}_vel_max",
            f"{joint_name}_vel_std",
        ])
    
    # Feature group 4: Joint angles
    # Indices in our 12-joint selection:
    # 0=left_shoulder, 1=right_shoulder, 2=left_elbow, 3=right_elbow
    # 4=left_wrist, 5=right_wrist, 6=left_hip, 7=right_hip
    # 8=left_knee, 9=right_knee, 10=left_ankle, 11=right_ankle
    
    angle_definitions = [
        ("left_elbow_angle", 0, 2, 4),     # shoulder-elbow-wrist (left)
        ("right_elbow_angle", 1, 3, 5),     # shoulder-elbow-wrist (right)
        ("left_knee_angle", 6, 8, 10),      # hip-knee-ankle (left)
        ("right_knee_angle", 7, 9, 11),     # hip-knee-ankle (right)
        ("left_shoulder_angle", 2, 0, 6),   # elbow-shoulder-hip (left)
        ("right_shoulder_angle", 3, 1, 7),  # elbow-shoulder-hip (right)
    ]
    
    for angle_name, idx_a, idx_b, idx_c in angle_definitions:
        angles = compute_joint_angle(
            keypoints[:, idx_a, :],
            keypoints[:, idx_b, :],
            keypoints[:, idx_c, :],
        )
        
        features.extend([
            np.mean(angles),
            np.std(angles),
            np.min(angles),
            np.max(angles),
            np.ptp(angles),  # range of angle
        ])
        feature_names.extend([
            f"{angle_name}_mean",
            f"{angle_name}_std",
            f"{angle_name}_min",
            f"{angle_name}_max",
            f"{angle_name}_range",
        ])
    
    # Feature group 5: Coordination features
    # Ratio of upper body movement to lower body movement
    upper_joints = [0, 1, 2, 3, 4, 5]  # shoulders, elbows, wrists
    lower_joints = [6, 7, 8, 9, 10, 11]  # hips, knees, ankles
    
    if len(keypoints) > 1:
        upper_movement = np.mean([
            np.mean(np.linalg.norm(np.diff(keypoints[:, j, :], axis=0), axis=1))
            for j in upper_joints
        ])
        lower_movement = np.mean([
            np.mean(np.linalg.norm(np.diff(keypoints[:, j, :], axis=0), axis=1))
            for j in lower_joints
        ])
        
        movement_ratio = upper_movement / (lower_movement + 1e-8)
        features.append(movement_ratio)
        feature_names.append("upper_lower_movement_ratio")
        
        # Symmetry: difference between left and right side movements
        left_joints = [0, 2, 4, 6, 8, 10]
        right_joints = [1, 3, 5, 7, 9, 11]
        
        left_movement = np.mean([
            np.mean(np.linalg.norm(np.diff(keypoints[:, j, :], axis=0), axis=1))
            for j in left_joints
        ])
        right_movement = np.mean([
            np.mean(np.linalg.norm(np.diff(keypoints[:, j, :], axis=0), axis=1))
            for j in right_joints
        ])
        
        symmetry_ratio = left_movement / (right_movement + 1e-8)
        features.append(symmetry_ratio)
        feature_names.append("left_right_symmetry")
    else:
        features.extend([0.0, 0.0])
        feature_names.extend(["upper_lower_movement_ratio", "left_right_symmetry"])
    
    return np.array(features), feature_names


def load_dataset(folder_path, label):
    """Load all preprocessed files from a folder and extract features."""
    folder = Path(folder_path)
    if not folder.exists():
        return [], []
    
    features_list = []
    labels = []
    names = []
    
    for f in sorted(folder.iterdir()):
        if f.name.endswith("_preprocessed.npy"):
            keypoints = np.load(str(f))
            feats, feat_names = extract_features(keypoints)
            features_list.append(feats)
            labels.append(label)
            names.append(f.stem)
    
    return features_list, labels, names, feat_names if features_list else []


def main():
    print("=" * 60)
    print("TENNIS STROKE ML CLASSIFICATION")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # LOAD DATA
    # ================================================================
    print("\nLoading data...")
    
    all_features = []
    all_labels = []
    all_names = []
    all_sources = []
    feat_names = None
    
    # THETIS Expert data (label = "expert")
    for stroke in ["forehand", "backhand"]:
        folder = PREPROCESSED_DIR / "thetis" / stroke
        feats, labels, names, fn = load_dataset(folder, "expert")
        if feats:
            all_features.extend(feats)
            all_labels.extend(labels)
            all_names.extend(names)
            all_sources.extend([f"thetis_expert_{stroke}"] * len(feats))
            if feat_names is None:
                feat_names = fn
    
    expert_count = len(all_features)
    print(f"  THETIS expert samples: {expert_count}")
    
    # THETIS Beginner data (label = "beginner")
    for stroke in ["forehand", "backhand"]:
        folder = PREPROCESSED_DIR / "thetis_beginners" / stroke
        feats, labels, names, fn = load_dataset(folder, "beginner")
        if feats:
            all_features.extend(feats)
            all_labels.extend(labels)
            all_names.extend(names)
            all_sources.extend([f"thetis_beginner_{stroke}"] * len(feats))
    
    beginner_count = len(all_features) - expert_count
    print(f"  THETIS beginner samples: {beginner_count}")
    
    if expert_count == 0 or beginner_count == 0:
        print("\nERROR: Need both expert and beginner data for classification.")
        print("Make sure you have run:")
        print("  1. python src/extract_thetis_beginners.py")
        print("  2. python src/preprocess_thetis_beginners.py")
        return
    
    # Personal recordings (for prediction later, not training)
    personal_features = []
    personal_names = []
    personal_sources = []
    
    personal_folders = [
        "forehand_tennis_with_ball",
        "forehand_tennis_without_ball",
        "backhand_tennis_with_ball",
        "backhand_tennis_without_ball",
    ]
    
    for folder_name in personal_folders:
        folder = PREPROCESSED_DIR / folder_name
        feats, _, names, _ = load_dataset(folder, "personal")
        if feats:
            personal_features.extend(feats)
            personal_names.extend(names)
            personal_sources.extend([folder_name] * len(feats))
    
    print(f"  Personal samples (for prediction): {len(personal_features)}")
    
    # ================================================================
    # PREPARE TRAINING DATA
    # ================================================================
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\nTraining data shape: {X.shape}")
    print(f"  Expert: {np.sum(y == 'expert')}")
    print(f"  Beginner: {np.sum(y == 'beginner')}")
    
    # Replace any NaN or inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ================================================================
    # TRAIN AND EVALUATE CLASSIFIERS
    # ================================================================
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
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])
        
        # Cross-validation
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        
        print(f"  Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
        print(f"  Per-fold scores: {[f'{s:.3f}' for s in scores]}")
        
        # Train on full dataset for prediction
        pipeline.fit(X, y)
        trained_models[name] = pipeline
        
        # Full training set classification report
        y_pred = pipeline.predict(X)
        print(f"\n  Training set classification report:")
        report = classification_report(y, y_pred, output_dict=True)
        print(f"    Accuracy: {report['accuracy']:.3f}")
        print(f"    Expert precision: {report['expert']['precision']:.3f}")
        print(f"    Expert recall: {report['expert']['recall']:.3f}")
        print(f"    Beginner precision: {report['beginner']['precision']:.3f}")
        print(f"    Beginner recall: {report['beginner']['recall']:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred, labels=["expert", "beginner"])
        print(f"\n  Confusion matrix:")
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
    
    # ================================================================
    # FEATURE IMPORTANCE (Random Forest)
    # ================================================================
    print(f"\n{'=' * 50}")
    print("TOP 15 MOST IMPORTANT FEATURES (Random Forest)")
    print(f"{'=' * 50}")
    
    rf_pipeline = trained_models["Random Forest"]
    rf_model = rf_pipeline.named_steps["classifier"]
    importances = rf_model.feature_importances_
    
    # Sort by importance
    importance_indices = np.argsort(importances)[::-1]
    
    for rank, idx in enumerate(importance_indices[:15]):
        bar = "█" * int(importances[idx] * 200)
        print(f"  {rank+1:>2}. {feat_names[idx]:<35} {importances[idx]:.4f}  {bar}")
    
    # ================================================================
    # PREDICT ON PERSONAL RECORDINGS
    # ================================================================
    if personal_features:
        print(f"\n{'=' * 50}")
        print("PREDICTIONS ON PERSONAL RECORDINGS")
        print(f"{'=' * 50}")
        
        X_personal = np.array(personal_features)
        X_personal = np.nan_to_num(X_personal, nan=0.0, posinf=0.0, neginf=0.0)
        
        personal_results = []
        
        for model_name, pipeline in trained_models.items():
            predictions = pipeline.predict(X_personal)
            probabilities = pipeline.predict_proba(X_personal)
            
            expert_count_pred = np.sum(predictions == "expert")
            beginner_count_pred = np.sum(predictions == "beginner")
            
            print(f"\n--- {model_name} ---")
            print(f"  Expert predictions: {expert_count_pred}/{len(predictions)}")
            print(f"  Beginner predictions: {beginner_count_pred}/{len(predictions)}")
            
            # Per-folder breakdown
            print(f"\n  Per-folder breakdown:")
            for folder_name in personal_folders:
                folder_mask = [s == folder_name for s in personal_sources]
                folder_preds = predictions[folder_mask]
                if len(folder_preds) > 0:
                    expert_pct = np.mean(folder_preds == "expert") * 100
                    print(f"    {folder_name}: {expert_pct:.0f}% expert, {100-expert_pct:.0f}% beginner")
            
            # Store detailed results
            for i, (name, source, pred) in enumerate(zip(personal_names, personal_sources, predictions)):
                # Get probability for expert class
                class_labels = pipeline.classes_
                expert_idx = np.where(class_labels == "expert")[0][0]
                expert_prob = probabilities[i][expert_idx]
                
                personal_results.append({
                    "video": name,
                    "folder": source,
                    "model": model_name,
                    "prediction": pred,
                    "expert_probability": round(float(expert_prob), 4),
                })
        
        # Save personal predictions
        pred_path = RESULTS_DIR / "ml_personal_predictions.json"
        with open(str(pred_path), "w") as f:
            json.dump(personal_results, f, indent=2)
        print(f"\nPersonal predictions saved to: {pred_path}")
    
    # ================================================================
    # SAVE RESULTS
    # ================================================================
    results_path = RESULTS_DIR / "ml_classification_results.json"
    with open(str(results_path), "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nClassification results saved to: {results_path}")
    
    # Save feature importance
    importance_data = [
        {"feature": feat_names[i], "importance": round(float(importances[i]), 6)}
        for i in importance_indices
    ]
    imp_path = RESULTS_DIR / "feature_importance.json"
    with open(str(imp_path), "w") as f:
        json.dump(importance_data, f, indent=2)
    print(f"Feature importance saved to: {imp_path}")
    
    print(f"\n{'=' * 60}")
    print("ML CLASSIFICATION COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
