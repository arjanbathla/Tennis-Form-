"""
Tennis Stroke Keypoint Extraction
Processes all recorded tennis stroke videos through MediaPipe Pose
and saves the extracted keypoint data as .npy files.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FOLDERS = [
    "forehand_tennis_with_ball",
    "forehand_tennis_without_ball",
    "backhand_tennis_with_ball",
    "backhand_tennis_without_ball",
]


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


def process_all_videos():
    for folder in FOLDERS:
        (PROCESSED_DIR / folder).mkdir(parents=True, exist_ok=True)

    summary = []
    
    print("=" * 60)
    print("TENNIS STROKE KEYPOINT EXTRACTION")
    print("=" * 60)
    
    total_videos = 0
    successful = 0
    
    for folder in FOLDERS:
        raw_folder = RAW_DATA_DIR / folder
        processed_folder = PROCESSED_DIR / folder
        
        if not raw_folder.exists():
            print(f"\nWARNING: Folder not found: {raw_folder}")
            continue
        
        # Get all video files
        video_files = sorted([
            f for f in raw_folder.iterdir()
            if f.suffix.lower() in ['.mov', '.mp4', '.avi']
        ])
        
        print(f"\n--- {folder} ({len(video_files)} videos) ---")
        
        for video_file in video_files:
            total_videos += 1
            print(f"\n  Processing: {video_file.name}")
            
            # Extract keypoints
            keypoints, fps, total_frames, detected_frames = extract_keypoints_from_video(video_file)
            
            if keypoints is not None and detected_frames > 0:
                detection_rate = (detected_frames / total_frames) * 100 if total_frames > 0 else 0

                output_name = video_file.stem + "_keypoints.npy"
                np.save(str(processed_folder / output_name), keypoints)

                metadata = {
                    "source_video": video_file.name,
                    "folder": folder,
                    "fps": fps,
                    "total_frames": total_frames,
                    "detected_frames": detected_frames,
                    "detection_rate": round(detection_rate, 1),
                    "keypoints_shape": list(keypoints.shape),
                }
                meta_path = processed_folder / (video_file.stem + "_metadata.json")
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
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total videos processed: {total_videos}")
    print(f"Successful: {successful}")
    print(f"Failed: {total_videos - successful}")
    
    if summary:
        print(f"Average detection rate: {np.mean([s['detection_rate'] for s in summary]):.1f}%")
        summary_path = PROCESSED_DIR / "extraction_summary.json"
        with open(str(summary_path), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")

    print("\nPer-folder breakdown:")
    for folder in FOLDERS:
        folder_results = [s for s in summary if s["folder"] == folder]
        if folder_results:
            avg_rate = np.mean([s["detection_rate"] for s in folder_results])
            print(f"  {folder}: {len(folder_results)} videos, avg detection {avg_rate:.1f}%")


if __name__ == "__main__":
    process_all_videos()
