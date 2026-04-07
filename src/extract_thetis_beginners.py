"""
Extracts keypoints from THETIS beginner players (p1-p31)
via MediaPipe Pose.
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
    name = Path(filename).stem
    player_part = name.split("_")[0]
    try:
        player_num = int(player_part[1:])
        return player_num in BEGINNER_RANGE
    except (ValueError, IndexError):
        return False


def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_keypoints = []
    detected_frames = 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=2,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                all_keypoints.append([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])
                detected_frames += 1
            else:
                all_keypoints.append([[0.0, 0.0]] * 33)

    cap.release()
    if not all_keypoints:
        return None, fps, total_frames, 0
    return np.array(all_keypoints), fps, total_frames, detected_frames


def main():
    print("=" * 60)
    print("THETIS BEGINNER KEYPOINT EXTRACTION")
    print("=" * 60)
    print("Processing beginner players (p1-p31)")
    
    summary = []
    total = 0
    successful = 0
    
    for thetis_folder, our_label in STROKE_FOLDERS.items():
        source_dir = THETIS_RGB_DIR / thetis_folder
        output_dir = PROCESSED_DIR / our_label
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not source_dir.exists():
            print(f"\nWARNING: {source_dir} not found")
            continue
        
        video_files = sorted([
            f for f in source_dir.iterdir()
            if f.suffix.lower() in ['.avi', '.mp4', '.mov']
        ])
        
        beginner_videos = [f for f in video_files if is_beginner(f.name)]
        
        print(f"\n--- {thetis_folder} ---")
        print(f"  Beginner videos: {len(beginner_videos)}")
        
        for video_file in beginner_videos:
            total += 1
            print(f"\n  Processing: {video_file.name}")
            
            keypoints, fps, total_frames, detected_frames = extract_keypoints_from_video(video_file)
            
            if keypoints is not None and detected_frames > 0:
                detection_rate = (detected_frames / total_frames) * 100
                
                output_name = video_file.stem + "_keypoints.npy"
                np.save(str(output_dir / output_name), keypoints)
                
                metadata = {
                    "source_video": video_file.name,
                    "thetis_folder": thetis_folder,
                    "stroke_type": our_label,
                    "player": video_file.stem.split("_")[0],
                    "skill_level": "beginner",
                    "fps": fps,
                    "total_frames": total_frames,
                    "detected_frames": detected_frames,
                    "detection_rate": round(detection_rate, 1),
                    "keypoints_shape": list(keypoints.shape),
                }
                meta_path = output_dir / (video_file.stem + "_metadata.json")
                with open(str(meta_path), "w") as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"    Detected: {detected_frames}/{total_frames} ({detection_rate:.1f}%)")
                summary.append(metadata)
                successful += 1
            else:
                print(f"    FAILED")
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {successful}/{total} successful")
    if summary:
        avg = np.mean([s["detection_rate"] for s in summary])
        print(f"Average detection rate: {avg:.1f}%")
    
    summary_path = PROCESSED_DIR / "beginner_extraction_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
