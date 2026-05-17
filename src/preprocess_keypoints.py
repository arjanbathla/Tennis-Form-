"""
Raw keypoint files -> drop failed frames, hip-centre, torso-scale,
keep 12 joints, 3-frame smooth.
"""

import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

# shoulders, elbows, wrists, hips, knees, ankles — skipping face/hands/feet
SELECTED_JOINTS = {
    11: "left_shoulder",  12: "right_shoulder",
    13: "left_elbow",     14: "right_elbow",
    15: "left_wrist",     16: "right_wrist",
    23: "left_hip",       24: "right_hip",
    25: "left_knee",      26: "right_knee",
    27: "left_ankle",     28: "right_ankle",
}

JOINT_INDICES = sorted(SELECTED_JOINTS.keys())
JOINT_NAMES = [SELECTED_JOINTS[i] for i in JOINT_INDICES]

LEFT_HIP_IDX, RIGHT_HIP_IDX = 23, 24
LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX = 11, 12

SMOOTHING_WINDOW = 3
MIN_DETECTION_RATE = 0.7


def remove_failed_frames(kp):
    valid = np.sum(np.abs(kp), axis=(1, 2)) > 0
    cleaned = kp[valid]
    kept = int(valid.sum())
    return cleaned, kept, len(kp) - kept


def select_relevant_joints(kp):
    return kp[:, JOINT_INDICES, :]


def centre_on_hips(kp):
    hip = (kp[:, LEFT_HIP_IDX, :] + kp[:, RIGHT_HIP_IDX, :]) / 2.0
    return kp - hip[:, np.newaxis, :]


def scale_by_torso(kp):
    sh = (kp[:, LEFT_SHOULDER_IDX, :] + kp[:, RIGHT_SHOULDER_IDX, :]) / 2.0
    hp = (kp[:, LEFT_HIP_IDX, :] + kp[:, RIGHT_HIP_IDX, :]) / 2.0
    torso = np.mean(np.linalg.norm(sh - hp, axis=1))
    if torso < 0.01:
        print("    tiny torso, skipping scale")
        return kp, torso
    return kp / torso, torso


def smooth_keypoints(kp, window=SMOOTHING_WINDOW):
    if len(kp) < window:
        return kp
    smoothed = np.copy(kp).astype(float)
    h = window // 2
    for i in range(len(kp)):
        smoothed[i] = np.mean(kp[max(0, i-h):min(len(kp), i+h+1)], axis=0)
    return smoothed


def preprocess_single_video(path):
    kp = np.load(str(path))
    info = {
        "source_file": path.name,
        "original_frames": len(kp),
        "original_shape": list(kp.shape),
    }

    kp, kept, removed = remove_failed_frames(kp)
    info["frames_after_cleaning"] = kept
    info["frames_removed"] = removed
    rate = kept / info["original_frames"] if info["original_frames"] > 0 else 0
    info["detection_rate"] = round(rate, 3)

    if kept < 5:
        info["status"] = "too few valid frames"
        return None, info
    if rate < MIN_DETECTION_RATE:
        info["status"] = f"detection rate {rate:.1%} below {MIN_DETECTION_RATE:.0%}"
        return None, info

    kp = centre_on_hips(kp)
    kp, torso = scale_by_torso(kp)
    info["avg_torso_length"] = round(float(torso), 4)
    kp = select_relevant_joints(kp)
    kp = smooth_keypoints(kp)

    info["final_frames"] = len(kp)
    info["final_shape"] = list(kp.shape)
    info["joints_selected"] = JOINT_NAMES
    info["status"] = "ok"
    return kp, info


def process_folder(input_dir, output_dir, label):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in input_dir.iterdir() if f.name.endswith("_keypoints.npy")])

    if not files:
        print(f"  no .npy in {input_dir}")
        return []

    print(f"\n{label}: {len(files)} files")

    results = []
    successful = 0

    for f in files:
        prep, info = preprocess_single_video(f)
        info["folder"] = label

        if prep is not None:
            out_name = f.stem.replace("_keypoints", "_preprocessed") + ".npy"
            np.save(str(output_dir / out_name), prep)
            info["output_file"] = out_name
            successful += 1
            print(f"  {f.stem}: {info['original_frames']} -> {info['final_frames']} "
                  f"frames, torso={info['avg_torso_length']:.3f}")
        else:
            print(f"  {f.stem}: {info['status']}")

        results.append(info)

    print(f"  {successful}/{len(files)} ok")
    return results


def main():
    print(f"preprocessing — joints={len(JOINT_INDICES)}, smooth={SMOOTHING_WINDOW}, "
          f"min_detect={MIN_DETECTION_RATE:.0%}")

    all_results = []

    print("\npersonal recordings")
    for folder in ["forehand_tennis_with_ball", "forehand_tennis_without_ball",
                   "backhand_tennis_with_ball", "backhand_tennis_without_ball"]:
        if (PROCESSED_DIR / folder).exists():
            all_results.extend(process_folder(PROCESSED_DIR / folder,
                                              PREPROCESSED_DIR / folder, folder))

    print("\nTHETIS experts")
    for folder in ["forehand", "backhand"]:
        if (PROCESSED_DIR / "thetis" / folder).exists():
            all_results.extend(process_folder(PROCESSED_DIR / "thetis" / folder,
                                              PREPROCESSED_DIR / "thetis" / folder,
                                              f"thetis_{folder}"))

    ok = [r for r in all_results if r["status"] == "ok"]
    failed = [r for r in all_results if r["status"] != "ok"]

    print(f"\n{len(ok)}/{len(all_results)} ok, {len(failed)} failed")

    if failed:
        for f in failed:
            print(f"  {f['source_file']}: {f['status']}")

    if ok:
        ao = np.mean([r["original_frames"] for r in ok])
        af = np.mean([r["final_frames"] for r in ok])
        at = np.mean([r["avg_torso_length"] for r in ok])
        print(f"avg frames: {ao:.0f} -> {af:.0f}, avg torso: {at:.3f}")

    summary_path = PREPROCESSED_DIR / "preprocessing_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(summary_path), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
