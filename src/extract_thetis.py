"""
Extracts keypoints from THETIS expert players (p32-p55)
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


def extract_keypoints_from_video(video_path, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Could not open {video_path}")
        return None, None, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_keypoints = []
    detected_frames = 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=2,
                      min_detection_confidence=min_detection_confidence,
                      min_tracking_confidence=min_tracking_confidence) as pose:
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


def process_thetis_experts():
    print("=" * 60)
    print("THETIS EXPERT KEYPOINT EXTRACTION")
    print("=" * 60)
    print("Processing expert players (p32-p55) only")
    print(f"Stroke types: {list(STROKE_FOLDERS.keys())}")
    
    summary = []
    total_videos = 0
    successful = 0
    skipped_beginners = 0
    
    for thetis_folder, our_label in STROKE_FOLDERS.items():
        source_dir = THETIS_RGB_DIR / thetis_folder
        output_dir = PROCESSED_DIR / our_label
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not source_dir.exists():
            print(f"\nWARNING: Folder not found: {source_dir}")
            continue
        
        video_files = sorted([
            f for f in source_dir.iterdir()
            if f.suffix.lower() in ['.avi', '.mp4', '.mov']
        ])
        
        expert_videos = [f for f in video_files if is_expert(f.name)]
        beginner_videos = len(video_files) - len(expert_videos)
        skipped_beginners += beginner_videos
        
        print(f"\n--- {thetis_folder} ---")
        print(f"  Total videos: {len(video_files)}")
        print(f"  Expert videos: {len(expert_videos)}")
        print(f"  Beginners skipped: {beginner_videos}")
        
        for video_file in expert_videos:
            total_videos += 1
            print(f"\n  Processing: {video_file.name}")
            
            keypoints, fps, total_frames, detected_frames = extract_keypoints_from_video(video_file)
            
            if keypoints is not None and detected_frames > 0:
                detection_rate = (detected_frames / total_frames) * 100 if total_frames > 0 else 0

                output_name = video_file.stem + "_keypoints.npy"
                np.save(str(output_dir / output_name), keypoints)

                metadata = {
                    "source_video": video_file.name,
                    "thetis_folder": thetis_folder,
                    "stroke_type": our_label,
                    "player": video_file.stem.split("_")[0],
                    "skill_level": "expert",
                    "fps": fps,
                    "total_frames": total_frames,
                    "detected_frames": detected_frames,
                    "detection_rate": round(detection_rate, 1),
                    "keypoints_shape": list(keypoints.shape),
                }
                meta_path = output_dir / (video_file.stem + "_metadata.json")
                with open(str(meta_path), "w") as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"    FPS: {fps}")
                print(f"    Frames: {total_frames}")
                print(f"    Pose detected: {detected_frames}/{total_frames} ({detection_rate:.1f}%)")
                print(f"    Saved: {output_name}")
                
                summary.append(metadata)
                successful += 1
            else:
                print(f"    FAILED: No keypoints extracted")
    
    print("\n" + "=" * 60)
    print("THETIS EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Expert videos processed: {total_videos}")
    print(f"Successful: {successful}")
    print(f"Failed: {total_videos - successful}")
    print(f"Beginner videos skipped: {skipped_beginners}")
    
    if summary:
        print(f"Average detection rate: {np.mean([s['detection_rate'] for s in summary]):.1f}%")

        print("\nPer-stroke breakdown:")
        for _, label in STROKE_FOLDERS.items():
            sr = [s for s in summary if s["stroke_type"] == label]
            if sr:
                print(f"  {label}: {len(sr)} videos, avg detection {np.mean([s['detection_rate'] for s in sr]):.1f}%")

        summary_path = PROCESSED_DIR / "thetis_extraction_summary.json"
        with open(str(summary_path), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    process_thetis_experts()
