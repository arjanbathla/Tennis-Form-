"""
Tennis Stroke ML Classification
Trains RF, SVM, and KNN to distinguish expert from beginner technique,
then predicts skill level on personal recordings.
"""

import numpy as np
import json
import csv
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
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

    # 75/25 stratified split, seed=42 — same as unified_evaluation in app.py
    idx_tr, idx_te = train_test_split(list(range(len(X))), test_size=0.25,
                                       stratify=y, random_state=42)
    X_train, X_test = X[idx_tr], X[idx_te]
    y_train, y_test = y[idx_tr], y[idx_te]

    print(f"\nTotal: {X.shape[0]}, Train: {len(idx_tr)}, Test: {len(idx_te)}")
    print(f"  expert={np.sum(y == 'expert')}, beginner={np.sum(y == 'beginner')}")

    print(f"\n{'=' * 50}")
    print("HYPERPARAMETER TUNING (GridSearchCV on training set)")
    print(f"{'=' * 50}")

    param_grids = {
        "Random Forest": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [5, 10, 20, None],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__class_weight": ["balanced"],
        },
        "SVM": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__kernel": ["rbf", "linear"],
            "classifier__gamma": ["scale", "auto"],
            "classifier__class_weight": ["balanced"],
        },
        "KNN": {
            "classifier__n_neighbors": [3, 5, 7, 9],
            "classifier__weights": ["uniform", "distance"],
            "classifier__metric": ["euclidean", "manhattan"],
        },
    }

    base_clfs = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        # algorithm='ball_tree' avoids the sklearn ArgKminClassMode fast path,
        # which crashes on string labels in sklearn >=1.5.
        "KNN": KNeighborsClassifier(algorithm="ball_tree"),
    }

    results_summary = {}
    trained_models = {}

    for name, clf in base_clfs.items():
        print(f"\n--- {name} ---")

        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
        gs = GridSearchCV(pipeline, param_grids[name], cv=5, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train, y_train)

        best_pipe = gs.best_estimator_
        trained_models[name] = best_pipe
        print(f"  Best params: {gs.best_params_}")
        print(f"  GridSearchCV best CV accuracy: {gs.best_score_:.3f}")

        y_pred_train = best_pipe.predict(X_train)
        y_pred_test = best_pipe.predict(X_test)
        report_train = classification_report(y_train, y_pred_train, output_dict=True)
        report_test = classification_report(y_test, y_pred_test, output_dict=True)
        print(f"  Training accuracy: {report_train['accuracy']:.3f}")
        print(f"  Test accuracy: {report_test['accuracy']:.3f}")
        print(f"  Test Expert p/r: {report_test['expert']['precision']:.3f} / {report_test['expert']['recall']:.3f}")
        print(f"  Test Beginner p/r: {report_test['beginner']['precision']:.3f} / {report_test['beginner']['recall']:.3f}")

        cm = confusion_matrix(y_test, y_pred_test, labels=["expert", "beginner"])
        print(f"  Test confusion matrix:")
        print(f"                 Predicted")
        print(f"                 Expert  Beginner")
        print(f"    Actual Expert   {cm[0][0]:>4}    {cm[0][1]:>4}")
        print(f"    Actual Beginner {cm[1][0]:>4}    {cm[1][1]:>4}")

        results_summary[name] = {
            "cv_accuracy_mean": round(float(gs.best_score_), 4),
            "cv_accuracy_std": round(float(gs.cv_results_["std_test_score"][gs.best_index_]), 4),
            "cv_scores": [round(float(gs.best_score_), 4)],
            "training_accuracy": round(float(report_train["accuracy"]), 4),
            "confusion_matrix": cm.tolist(),
            "best_params": {k: str(v) for k, v in gs.best_params_.items()},
        }

    base_estimators = [
        ("rf", trained_models["Random Forest"]),
        ("svm", trained_models["SVM"]),
        ("knn", trained_models["KNN"]),
    ]

    ensembles = {
        "Soft Voting Ensemble": VotingClassifier(estimators=base_estimators, voting="soft"),
        "Stacking Ensemble": StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, ens in ensembles.items():
        print(f"\n--- {name} ---")
        scores = cross_val_score(ens, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"  CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

        ens.fit(X_train, y_train)
        trained_models[name] = ens

        y_pred_train = ens.predict(X_train)
        y_pred_test = ens.predict(X_test)
        report_train = classification_report(y_train, y_pred_train, output_dict=True)
        report_test = classification_report(y_test, y_pred_test, output_dict=True)
        print(f"  Training accuracy: {report_train['accuracy']:.3f}")
        print(f"  Test accuracy: {report_test['accuracy']:.3f}")
        print(f"  Test Expert p/r: {report_test['expert']['precision']:.3f} / {report_test['expert']['recall']:.3f}")
        print(f"  Test Beginner p/r: {report_test['beginner']['precision']:.3f} / {report_test['beginner']['recall']:.3f}")

        cm = confusion_matrix(y_test, y_pred_test, labels=["expert", "beginner"])
        print(f"  Test confusion matrix:")
        print(f"                 Predicted")
        print(f"                 Expert  Beginner")
        print(f"    Actual Expert   {cm[0][0]:>4}    {cm[0][1]:>4}")
        print(f"    Actual Beginner {cm[1][0]:>4}    {cm[1][1]:>4}")

        results_summary[name] = {
            "cv_accuracy_mean": round(float(scores.mean()), 4),
            "cv_accuracy_std": round(float(scores.std()), 4),
            "cv_scores": [round(float(s), 4) for s in scores],
            "training_accuracy": round(float(report_train["accuracy"]), 4),
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
