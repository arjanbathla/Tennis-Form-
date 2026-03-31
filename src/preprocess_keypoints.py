"""
Tennis Stroke Keypoint Preprocessing
Cleans and normalises extracted keypoint data from both personal
recordings and THETIS expert videos so they are directly comparable.

Steps:
1. Remove frames where pose detection failed (zero-filled frames)
2. Select biomechanically relevant joints (remove face/hand landmarks)
3. Centre skeleton on hip midpoint (removes position-in-frame effects)
4. Scale by torso length (removes body size differences)
5. Smooth keypoints with a moving average (reduces jitter)
6. Save preprocessed data
"""

import numpy as np
import json
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

# ----------------------------------------------------------------
# MediaPipe Pose landmark indices (33 total)
# We select the 12 most biomechanically relevant joints for tennis
# ----------------------------------------------------------------
SELECTED_JOINTS = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

JOINT_INDICES = sorted(SELECTED_JOINTS.keys())
JOINT_NAMES = [SELECTED_JOINTS[i] for i in JOINT_INDICES]

# Hip indices for centering
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24

# Shoulder indices for torso length calculation
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

# Smoothing window size (number of frames for moving average)
SMOOTHING_WINDOW = 3

# Minimum detection rate to keep a video
MIN_DETECTION_RATE = 0.7


def remove_failed_frames(keypoints):
    """
    Remove frames where pose detection failed.
    Failed frames are identified as rows where all values are zero.
    
    Args:
        keypoints: numpy array of shape (frames, 33, 2)
    
    Returns:
        cleaned: numpy array with failed frames removed
        kept_count: number of frames kept
        removed_count: number of frames removed
    """
    # A frame is failed if all keypoint values are zero
    frame_sums = np.sum(np.abs(keypoints), axis=(1, 2))
    valid_mask = frame_sums > 0
    
    cleaned = keypoints[valid_mask]
    kept_count = int(np.sum(valid_mask))
    removed_count = len(keypoints) - kept_count
    
    return cleaned, kept_count, removed_count


def select_relevant_joints(keypoints):
    """
    Select only the biomechanically relevant joints.
    Removes face landmarks, finger landmarks, and toe landmarks
    that add noise without helping stroke comparison.
    
    Args:
        keypoints: numpy array of shape (frames, 33, 2)
    
    Returns:
        selected: numpy array of shape (frames, 12, 2)
    """
    return keypoints[:, JOINT_INDICES, :]


def centre_on_hips(keypoints):
    """
    Centre the skeleton on the midpoint of the two hips.
    This removes the effect of where the player stands in the frame.
    
    After this step, the hip midpoint is at (0, 0) for every frame,
    and all other joints are relative to the hips.
    
    Args:
        keypoints: numpy array of shape (frames, 33, 2)
        Note: must be called BEFORE select_relevant_joints
    
    Returns:
        centred: numpy array of same shape, centred on hip midpoint
    """
    left_hip = keypoints[:, LEFT_HIP_IDX, :]
    right_hip = keypoints[:, RIGHT_HIP_IDX, :]
    hip_midpoint = (left_hip + right_hip) / 2.0
    
    # Subtract hip midpoint from all joints
    centred = keypoints - hip_midpoint[:, np.newaxis, :]
    
    return centred


def scale_by_torso(keypoints):
    """
    Scale keypoints by torso length to normalise for body size differences.
    Torso length is defined as the distance from shoulder midpoint to hip midpoint.
    
    After centering on hips, the hip midpoint is at (0,0), so torso length
    is just the distance from shoulder midpoint to origin.
    
    Args:
        keypoints: numpy array of shape (frames, 33, 2)
        Note: must be called AFTER centre_on_hips
    
    Returns:
        scaled: numpy array of same shape
        avg_torso_length: the average torso length used for scaling
    """
    left_shoulder = keypoints[:, LEFT_SHOULDER_IDX, :]
    right_shoulder = keypoints[:, RIGHT_SHOULDER_IDX, :]
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
    
    left_hip = keypoints[:, LEFT_HIP_IDX, :]
    right_hip = keypoints[:, RIGHT_HIP_IDX, :]
    hip_midpoint = (left_hip + right_hip) / 2.0
    
    # Torso length per frame
    torso_lengths = np.linalg.norm(shoulder_midpoint - hip_midpoint, axis=1)
    
    # Use average torso length across all frames to avoid frame-by-frame noise
    avg_torso_length = np.mean(torso_lengths)
    
    if avg_torso_length < 0.01:
        # Avoid division by near-zero
        print("    WARNING: Very small torso length detected, skipping scaling")
        return keypoints, avg_torso_length
    
    scaled = keypoints / avg_torso_length
    
    return scaled, avg_torso_length


def smooth_keypoints(keypoints, window_size=SMOOTHING_WINDOW):
    """
    Apply a simple moving average to smooth keypoint trajectories.
    Reduces frame-to-frame jitter from the pose estimator.
    
    Args:
        keypoints: numpy array of shape (frames, joints, 2)
        window_size: number of frames for moving average
    
    Returns:
        smoothed: numpy array of same shape
    """
    if len(keypoints) < window_size:
        return keypoints
    
    smoothed = np.copy(keypoints).astype(float)
    half_window = window_size // 2
    
    for i in range(len(keypoints)):
        start = max(0, i - half_window)
        end = min(len(keypoints), i + half_window + 1)
        smoothed[i] = np.mean(keypoints[start:end], axis=0)
    
    return smoothed


def preprocess_single_video(keypoints_path, metadata_path=None):
    """
    Run the full preprocessing pipeline on a single video's keypoints.
    
    Args:
        keypoints_path: path to the .npy keypoints file
        metadata_path: path to the .json metadata file (optional)
    
    Returns:
        preprocessed: numpy array of shape (frames, 12, 2) or None if failed
        info: dictionary with preprocessing statistics
    """
    # Load raw keypoints
    keypoints = np.load(str(keypoints_path))
    
    info = {
        "source_file": keypoints_path.name,
        "original_frames": len(keypoints),
        "original_shape": list(keypoints.shape),
    }
    
    # Step 1: Remove failed frames
    keypoints, kept, removed = remove_failed_frames(keypoints)
    info["frames_after_cleaning"] = kept
    info["frames_removed"] = removed
    detection_rate = kept / info["original_frames"] if info["original_frames"] > 0 else 0
    info["detection_rate"] = round(detection_rate, 3)
    
    if kept < 5:
        info["status"] = "FAILED: Too few valid frames"
        return None, info
    
    if detection_rate < MIN_DETECTION_RATE:
        info["status"] = f"FAILED: Detection rate {detection_rate:.1%} below threshold {MIN_DETECTION_RATE:.0%}"
        return None, info
    
    # Step 2: Centre on hip midpoint
    keypoints = centre_on_hips(keypoints)
    
    # Step 3: Scale by torso length
    keypoints, avg_torso = scale_by_torso(keypoints)
    info["avg_torso_length"] = round(float(avg_torso), 4)
    
    # Step 4: Select relevant joints (after centering and scaling)
    keypoints = select_relevant_joints(keypoints)
    
    # Step 5: Smooth keypoints
    keypoints = smooth_keypoints(keypoints)
    
    info["final_frames"] = len(keypoints)
    info["final_shape"] = list(keypoints.shape)
    info["joints_selected"] = JOINT_NAMES
    info["status"] = "SUCCESS"
    
    return keypoints, info


def process_folder(input_dir, output_dir, label):
    """Process all keypoint files in a folder."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all keypoint files
    npy_files = sorted([f for f in input_dir.iterdir() if f.name.endswith("_keypoints.npy")])
    
    if not npy_files:
        print(f"  No keypoint files found in {input_dir}")
        return []
    
    print(f"\n--- {label} ({len(npy_files)} files) ---")
    
    results = []
    successful = 0
    
    for npy_file in npy_files:
        meta_file = npy_file.parent / npy_file.name.replace("_keypoints.npy", "_metadata.json")
        
        preprocessed, info = preprocess_single_video(npy_file, meta_file)
        info["folder"] = label
        
        if preprocessed is not None:
            # Save preprocessed keypoints
            output_name = npy_file.stem.replace("_keypoints", "_preprocessed") + ".npy"
            output_path = output_dir / output_name
            np.save(str(output_path), preprocessed)
            
            info["output_file"] = output_name
            successful += 1
            
            print(f"  {npy_file.stem}: {info['original_frames']} -> {info['final_frames']} frames, "
                  f"torso={info['avg_torso_length']:.3f}")
        else:
            print(f"  {npy_file.stem}: {info['status']}")
        
        results.append(info)
    
    print(f"  Successful: {successful}/{len(npy_files)}")
    return results


def main():
    """Run preprocessing on all extracted keypoints."""
    
    print("=" * 60)
    print("KEYPOINT PREPROCESSING")
    print("=" * 60)
    print(f"Selected joints ({len(JOINT_INDICES)}): {JOINT_NAMES}")
    print(f"Smoothing window: {SMOOTHING_WINDOW} frames")
    print(f"Min detection rate: {MIN_DETECTION_RATE:.0%}")
    
    all_results = []
    
    # ---- Process personal recordings ----
    print("\n" + "=" * 40)
    print("PERSONAL RECORDINGS")
    print("=" * 40)
    
    personal_folders = [
        "forehand_tennis_with_ball",
        "forehand_tennis_without_ball",
        "backhand_tennis_with_ball",
        "backhand_tennis_without_ball",
    ]
    
    for folder in personal_folders:
        input_dir = PROCESSED_DIR / folder
        output_dir = PREPROCESSED_DIR / folder
        if input_dir.exists():
            results = process_folder(input_dir, output_dir, folder)
            all_results.extend(results)
    
    # ---- Process THETIS expert recordings ----
    print("\n" + "=" * 40)
    print("THETIS EXPERT RECORDINGS")
    print("=" * 40)
    
    thetis_folders = ["forehand", "backhand"]
    
    for folder in thetis_folders:
        input_dir = PROCESSED_DIR / "thetis" / folder
        output_dir = PREPROCESSED_DIR / "thetis" / folder
        if input_dir.exists():
            results = process_folder(input_dir, output_dir, f"thetis_{folder}")
            all_results.extend(results)
    
    # ---- Print overall summary ----
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    successful = [r for r in all_results if r["status"] == "SUCCESS"]
    failed = [r for r in all_results if r["status"] != "SUCCESS"]
    
    print(f"Total files: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  {f['source_file']}: {f['status']}")
    
    if successful:
        avg_original = np.mean([r["original_frames"] for r in successful])
        avg_final = np.mean([r["final_frames"] for r in successful])
        avg_torso = np.mean([r["avg_torso_length"] for r in successful])
        
        print(f"\nAverage original frames: {avg_original:.0f}")
        print(f"Average final frames: {avg_final:.0f}")
        print(f"Average torso length: {avg_torso:.3f}")
    
    # Save full summary
    summary_path = PREPROCESSED_DIR / "preprocessing_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(summary_path), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
