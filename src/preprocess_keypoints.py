"""
Preprocesses raw keypoint files: removes failed frames, centres on hips,
scales by torso length, selects 12 biomechanical joints, and smooths.
"""

import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

# 12 biomechanically relevant joints (excludes face/hands/feet)
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

LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

SMOOTHING_WINDOW = 3
MIN_DETECTION_RATE = 0.7


def remove_failed_frames(keypoints):
    valid = np.sum(np.abs(keypoints), axis=(1, 2)) > 0
    cleaned = keypoints[valid]
    kept = int(valid.sum())
    return cleaned, kept, len(keypoints) - kept


def select_relevant_joints(keypoints):
    return keypoints[:, JOINT_INDICES, :]


def centre_on_hips(keypoints):
    hip_mid = (keypoints[:, LEFT_HIP_IDX, :] + keypoints[:, RIGHT_HIP_IDX, :]) / 2.0
    return keypoints - hip_mid[:, np.newaxis, :]


def scale_by_torso(keypoints):
    shoulder_mid = (keypoints[:, LEFT_SHOULDER_IDX, :] + keypoints[:, RIGHT_SHOULDER_IDX, :]) / 2.0
    hip_mid = (keypoints[:, LEFT_HIP_IDX, :] + keypoints[:, RIGHT_HIP_IDX, :]) / 2.0
    avg_torso = np.mean(np.linalg.norm(shoulder_mid - hip_mid, axis=1))
    if avg_torso < 0.01:
        print("    WARNING: Very small torso length detected, skipping scaling")
        return keypoints, avg_torso
    return keypoints / avg_torso, avg_torso


def smooth_keypoints(keypoints, window_size=SMOOTHING_WINDOW):
    if len(keypoints) < window_size:
        return keypoints
    smoothed = np.copy(keypoints).astype(float)
    half = window_size // 2
    for i in range(len(keypoints)):
        smoothed[i] = np.mean(keypoints[max(0, i-half):min(len(keypoints), i+half+1)], axis=0)
    return smoothed


def preprocess_single_video(keypoints_path):
    keypoints = np.load(str(keypoints_path))
    info = {
        "source_file": keypoints_path.name,
        "original_frames": len(keypoints),
        "original_shape": list(keypoints.shape),
    }

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

    keypoints = centre_on_hips(keypoints)
    keypoints, avg_torso = scale_by_torso(keypoints)
    info["avg_torso_length"] = round(float(avg_torso), 4)
    keypoints = select_relevant_joints(keypoints)
    keypoints = smooth_keypoints(keypoints)

    info["final_frames"] = len(keypoints)
    info["final_shape"] = list(keypoints.shape)
    info["joints_selected"] = JOINT_NAMES
    info["status"] = "SUCCESS"
    return keypoints, info


def process_folder(input_dir, output_dir, label):
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_files = sorted([f for f in input_dir.iterdir() if f.name.endswith("_keypoints.npy")])
    
    if not npy_files:
        print(f"  No keypoint files found in {input_dir}")
        return []
    
    print(f"\n--- {label} ({len(npy_files)} files) ---")
    
    results = []
    successful = 0
    
    for npy_file in npy_files:
        preprocessed, info = preprocess_single_video(npy_file)
        info["folder"] = label
        
        if preprocessed is not None:
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
    print("=" * 60)
    print("KEYPOINT PREPROCESSING")
    print("=" * 60)
    print(f"Joints: {len(JOINT_INDICES)}, smoothing: {SMOOTHING_WINDOW} frames, min detection: {MIN_DETECTION_RATE:.0%}")

    all_results = []

    print("\n--- Personal Recordings ---")
    for folder in ["forehand_tennis_with_ball", "forehand_tennis_without_ball",
                   "backhand_tennis_with_ball", "backhand_tennis_without_ball"]:
        if (PROCESSED_DIR / folder).exists():
            all_results.extend(process_folder(PROCESSED_DIR / folder, PREPROCESSED_DIR / folder, folder))

    print("\n--- THETIS Expert Recordings ---")
    for folder in ["forehand", "backhand"]:
        if (PROCESSED_DIR / "thetis" / folder).exists():
            all_results.extend(process_folder(PROCESSED_DIR / "thetis" / folder,
                                              PREPROCESSED_DIR / "thetis" / folder, f"thetis_{folder}"))

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
    
    summary_path = PREPROCESSED_DIR / "preprocessing_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(summary_path), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
