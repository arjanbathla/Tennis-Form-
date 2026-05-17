"""
Same MediaPipe extraction, but for THETIS beginner players (p1-p31).
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path

mp_pose = mp.solutions.pose

PROJECT_ROOT = Path(__file__).parent.parent
THETIS_RGB_DIR = PROJECT_ROOT / "data" / "thetis" / "dataset-main" / "VIDEO_RGB"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "thetis_beginners"

STROKE_FOLDERS = {
    "forehand_flat": "forehand",
    "backhand": "backhand",
}

BEGINNER_RANGE = range(1, 32)


def is_beginner(filename):
    try:
        return int(Path(filename).stem.split("_")[0][1:]) in BEGINNER_RANGE
    except (ValueError, IndexError):
        return False


def extract_keypoints(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
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


def main():
    print("thetis beginner extraction (p1-p31)")
    summary = []
    total_count = 0
    successful = 0

    for thetis_folder, label in STROKE_FOLDERS.items():
        src = THETIS_RGB_DIR / thetis_folder
        out = PROCESSED_DIR / label
        out.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            print(f"  missing: {src}")
            continue

        videos = sorted([f for f in src.iterdir()
                         if f.suffix.lower() in ['.avi', '.mp4', '.mov']])
        beginners = [f for f in videos if is_beginner(f.name)]
        print(f"\n{thetis_folder}: {len(beginners)} beginners")

        for v in beginners:
            total_count += 1
            kps, fps, total, detected = extract_keypoints(v)
            if kps is None or detected == 0:
                print(f"  {v.stem}: failed")
                continue

            rate = (detected / total) * 100 if total > 0 else 0
            np.save(str(out / (v.stem + "_keypoints.npy")), kps)
            meta = {
                "source_video": v.name,
                "thetis_folder": thetis_folder,
                "stroke_type": label,
                "player": v.stem.split("_")[0],
                "skill_level": "beginner",
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

    print(f"\ndone: {successful}/{total_count}")
    if summary:
        print(f"avg detection: {np.mean([s['detection_rate'] for s in summary]):.1f}%")
        with open(str(PROCESSED_DIR / "beginner_extraction_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
