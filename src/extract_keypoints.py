"""
Runs the recorded stroke videos through MediaPipe Pose
and saves the keypoints as .npy.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path

mp_pose = mp.solutions.pose

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FOLDERS = [
    "forehand_tennis_with_ball",
    "forehand_tennis_without_ball",
    "backhand_tennis_with_ball",
    "backhand_tennis_without_ball",
]


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


def process_all_videos():
    for f in FOLDERS:
        (PROCESSED_DIR / f).mkdir(parents=True, exist_ok=True)

    print("keypoint extraction")
    summary = []
    total_videos = 0
    successful = 0

    for folder in FOLDERS:
        raw = RAW_DATA_DIR / folder
        out = PROCESSED_DIR / folder
        if not raw.exists():
            print(f"  missing: {raw}")
            continue

        videos = sorted([f for f in raw.iterdir()
                         if f.suffix.lower() in ['.mov', '.mp4', '.avi']])
        print(f"\n{folder}: {len(videos)} videos")

        for v in videos:
            total_videos += 1
            kps, fps, total, detected = extract_keypoints(v)

            if kps is None or detected == 0:
                print(f"  {v.name}: failed")
                continue

            rate = (detected / total) * 100 if total > 0 else 0
            np.save(str(out / (v.stem + "_keypoints.npy")), kps)
            meta = {
                "source_video": v.name, "folder": folder, "fps": fps,
                "total_frames": total, "detected_frames": detected,
                "detection_rate": round(rate, 1), "keypoints_shape": list(kps.shape),
            }
            with open(str(out / (v.stem + "_metadata.json")), "w") as f:
                json.dump(meta, f, indent=2)
            summary.append(meta)
            successful += 1
            print(f"  {v.name}: {detected}/{total} ({rate:.1f}%)")

    print(f"\ndone: {successful}/{total_videos} successful")
    if summary:
        avg = np.mean([s["detection_rate"] for s in summary])
        print(f"avg detection rate: {avg:.1f}%")
        with open(str(PROCESSED_DIR / "extraction_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        for folder in FOLDERS:
            fr = [s for s in summary if s["folder"] == folder]
            if fr:
                print(f"  {folder}: {len(fr)} videos, {np.mean([s['detection_rate'] for s in fr]):.1f}%")


if __name__ == "__main__":
    process_all_videos()
