"""
Preprocess THETIS Beginner Keypoints
Same preprocessing pipeline as the expert and personal data:
hip-centering, torso scaling, joint selection, smoothing.
"""

import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "thetis_beginners"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed" / "thetis_beginners"

SELECTED_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
SMOOTHING_WINDOW = 3


def preprocess(keypoints):
    """Full preprocessing pipeline on raw keypoints."""
    # Remove failed frames
    frame_sums = np.sum(np.abs(keypoints), axis=(1, 2))
    keypoints = keypoints[frame_sums > 0]
    
    if len(keypoints) < 5:
        return None
    
    # Centre on hips
    hip_mid = (keypoints[:, LEFT_HIP, :] + keypoints[:, RIGHT_HIP, :]) / 2.0
    keypoints = keypoints - hip_mid[:, np.newaxis, :]
    
    # Scale by torso length
    shoulder_mid = (keypoints[:, LEFT_SHOULDER, :] + keypoints[:, RIGHT_SHOULDER, :]) / 2.0
    hip_mid2 = (keypoints[:, LEFT_HIP, :] + keypoints[:, RIGHT_HIP, :]) / 2.0
    torso = np.mean(np.linalg.norm(shoulder_mid - hip_mid2, axis=1))
    
    if torso < 0.01:
        return None
    
    keypoints = keypoints / torso
    
    # Select relevant joints
    keypoints = keypoints[:, SELECTED_JOINTS, :]
    
    # Smooth
    smoothed = np.copy(keypoints).astype(float)
    half = SMOOTHING_WINDOW // 2
    for i in range(len(keypoints)):
        start = max(0, i - half)
        end = min(len(keypoints), i + half + 1)
        smoothed[i] = np.mean(keypoints[start:end], axis=0)
    
    return smoothed


def main():
    print("=" * 60)
    print("PREPROCESSING THETIS BEGINNERS")
    print("=" * 60)
    
    total = 0
    successful = 0
    
    for folder in ["forehand", "backhand"]:
        input_dir = PROCESSED_DIR / folder
        output_dir = PREPROCESSED_DIR / folder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_dir.exists():
            print(f"  {input_dir} not found, skipping")
            continue
        
        npy_files = sorted([f for f in input_dir.iterdir() if f.name.endswith("_keypoints.npy")])
        print(f"\n--- {folder} ({len(npy_files)} files) ---")
        
        for npy_file in npy_files:
            total += 1
            keypoints = np.load(str(npy_file))
            result = preprocess(keypoints)
            
            if result is not None:
                output_name = npy_file.stem.replace("_keypoints", "_preprocessed") + ".npy"
                np.save(str(output_dir / output_name), result)
                successful += 1
                print(f"  {npy_file.stem}: {len(keypoints)} -> {len(result)} frames")
            else:
                print(f"  {npy_file.stem}: FAILED")
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {successful}/{total} successful")


if __name__ == "__main__":
    main()
