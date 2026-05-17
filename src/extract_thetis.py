"""
Extract keypoints for THETIS expert players (p32-p55) only,
using MediaPipe Pose. Beginners are handled by extract_thetis_beginners.py.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path

mp_pose = mp.solutions.pose

PROJECT_ROOT = Path(__file__).parent.parent
THETIS_RGB_DIR = PROJECT_ROOT / "data" / "thetis" / "dataset-main" / "VIDEO_RGB"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "thetis"

STROKE_FOLDERS = {
    "forehand_flat": "forehand",
    "backhand": "backhand",
}

EXPERT_RANGE = range(32, 56)  # p32-p55


def is_expert(filename):
    try:
        return int(Path(filename).stem.split("_")[0][1:]) in EXPERT_RANGE
    except (ValueError, IndexError):
        return False


def extract_keypoints(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  could not open {video_path}")
        return None, None, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kps, detected = [], 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=2,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            r = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if r.pose_landmarks:
                kps.append([[lm.x, lm.y] for lm in r.pose_landmarks.landmark])
                detected += 1
            else:
                kps.append([[0.0, 0.0]] * 33)

    cap.release()
    if not kps:
        return None, fps, total, 0
    return np.array(kps), fps, total, detected


def process_thetis_experts():
    print("thetis expert extraction (p32-p55)")
    summary = []
    total_videos = 0
    successful = 0
    skipped = 0

    for thetis_folder, label in STROKE_FOLDERS.items():
        src = THETIS_RGB_DIR / thetis_folder
        out = PROCESSED_DIR / label
        out.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            print(f"  missing: {src}")
            continue

        videos = sorted([f for f in src.iterdir()
                         if f.suffix.lower() in ['.avi', '.mp4', '.mov']])
        experts = [f for f in videos if is_expert(f.name)]
        skipped += len(videos) - len(experts)
        print(f"\n{thetis_folder}: {len(experts)} expert / {len(videos)} total")

        for v in experts:
            total_videos += 1
            kps, fps, total, detected = extract_keypoints(v)
            if kps is None or detected == 0:
                print(f"  {v.name}: failed")
                continue

            rate = (detected / total) * 100 if total > 0 else 0
            np.save(str(out / (v.stem + "_keypoints.npy")), kps)
            meta = {
                "source_video": v.name,
                "thetis_folder": thetis_folder,
                "stroke_type": label,
                "player": v.stem.split("_")[0],
                "skill_level": "expert",
                "fps": fps,
                "total_frames": total,
                "detected_frames": detected,
                "detection_rate": round(rate, 1),
                "keypoints_shape": list(kps.shape),
            }
            with open(str(out / (v.stem + "_metadata.json")), "w") as f:
                json.dump(meta, f, indent=2)
            summary.append(meta)
            successful += 1
            print(f"  {v.stem}: {detected}/{total} ({rate:.1f}%)")

    print(f"\ndone: {successful}/{total_videos} successful, {skipped} beginners skipped")
    if summary:
        print(f"avg detection: {np.mean([s['detection_rate'] for s in summary]):.1f}%")
        for _, label in STROKE_FOLDERS.items():
            sr = [s for s in summary if s["stroke_type"] == label]
            if sr:
                print(f"  {label}: {len(sr)} videos, {np.mean([s['detection_rate'] for s in sr]):.1f}%")
        with open(str(PROCESSED_DIR / "thetis_extraction_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    process_thetis_experts()
