"""
Same pipeline as preprocess_keypoints.py, for THETIS beginners.
Kept separate because the folder layout differs.
"""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "thetis_beginners"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed" / "thetis_beginners"

SELECTED_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
SMOOTHING_WINDOW = 3


def preprocess(kp):
    kp = kp[np.sum(np.abs(kp), axis=(1, 2)) > 0]
    if len(kp) < 5:
        return None

    hip = (kp[:, LEFT_HIP, :] + kp[:, RIGHT_HIP, :]) / 2.0
    kp = kp - hip[:, np.newaxis, :]

    sh = (kp[:, LEFT_SHOULDER, :] + kp[:, RIGHT_SHOULDER, :]) / 2.0
    hp = (kp[:, LEFT_HIP, :] + kp[:, RIGHT_HIP, :]) / 2.0
    torso = np.mean(np.linalg.norm(sh - hp, axis=1))
    if torso < 0.01:
        return None

    kp = kp / torso
    kp = kp[:, SELECTED_JOINTS, :]

    smoothed = np.copy(kp).astype(float)
    h = SMOOTHING_WINDOW // 2
    for i in range(len(kp)):
        smoothed[i] = np.mean(kp[max(0, i - h):min(len(kp), i + h + 1)], axis=0)
    return smoothed


def main():
    print("preprocessing THETIS beginners")
    total = 0
    successful = 0

    for folder in ["forehand", "backhand"]:
        src = PROCESSED_DIR / folder
        out = PREPROCESSED_DIR / folder
        out.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            print(f"  skip: {src}")
            continue

        files = sorted([f for f in src.iterdir() if f.name.endswith("_keypoints.npy")])
        print(f"\n{folder}: {len(files)} files")

        for f in files:
            total += 1
            kp = np.load(str(f))
            result = preprocess(kp)
            if result is not None:
                np.save(str(out / (f.stem.replace("_keypoints", "_preprocessed") + ".npy")), result)
                successful += 1
                print(f"  {f.stem}: {len(kp)} -> {len(result)} frames")
            else:
                print(f"  {f.stem}: failed")

    print(f"\ndone: {successful}/{total}")


if __name__ == "__main__":
    main()
