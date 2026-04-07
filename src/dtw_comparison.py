"""
Compares personal stroke keypoints against THETIS expert references
using Dynamic Time Warping (overall + per-joint).
"""

import numpy as np
import json
import csv
from pathlib import Path
from dtw import dtw

PROJECT_ROOT = Path(__file__).parent.parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results"

JOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def load_preprocessed_files(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return {}
    return {f.stem: np.load(str(f)) for f in sorted(folder.iterdir())
            if f.name.endswith("_preprocessed.npy")}


def compute_dtw_distance(sequence_a, sequence_b):
    flat_a = sequence_a.reshape(len(sequence_a), -1)
    flat_b = sequence_b.reshape(len(sequence_b), -1)
    overall = dtw(flat_a, flat_b, keep_internals=False).normalizedDistance
    per_joint = {jn: dtw(sequence_a[:, j, :], sequence_b[:, j, :], keep_internals=False).normalizedDistance
                 for j, jn in enumerate(JOINT_NAMES)}
    return overall, per_joint


def compute_average_reference(reference_files):
    names = list(reference_files.keys())
    if len(names) <= 1:
        return names[0], reference_files[names[0]]

    if len(names) > 20:
        names = [names[i] for i in np.random.choice(len(names), 20, replace=False)]

    total_dists = {}
    for i, na in enumerate(names):
        flat_a = reference_files[na].reshape(len(reference_files[na]), -1)
        total_dists[na] = sum(
            dtw(flat_a, reference_files[nb].reshape(len(reference_files[nb]), -1), keep_internals=False).normalizedDistance
            for j, nb in enumerate(names) if i != j
        )

    best = min(total_dists, key=total_dists.get)
    return best, reference_files[best]


def compare_stroke_type(personal_folders, thetis_folder, stroke_label):
    print(f"\n{'=' * 50}")
    print(f"COMPARING: {stroke_label.upper()}")
    print(f"{'=' * 50}")
    
    thetis_files = load_preprocessed_files(thetis_folder)
    if not thetis_files:
        print(f"  ERROR: No THETIS references found in {thetis_folder}")
        return []

    print(f"  {len(thetis_files)} expert references loaded")
    print(f"  Finding median reference...")
    median_name, median_ref = compute_average_reference(thetis_files)
    print(f"  Median: {median_name}")

    ref_names = list(thetis_files.keys())
    sample_refs = np.random.choice(ref_names, min(5, len(ref_names)), replace=False)
    results = []

    for personal_folder in personal_folders:
        personal_files = load_preprocessed_files(personal_folder)
        folder_name = Path(personal_folder).name
        
        if not personal_files:
            continue
        
        print(f"\n  --- {folder_name} ({len(personal_files)} files) ---")
        
        for video_name, personal_keypoints in personal_files.items():
            overall_dist, per_joint = compute_dtw_distance(personal_keypoints, median_ref)
            multi_dists = [compute_dtw_distance(personal_keypoints, thetis_files[r])[0] for r in sample_refs]
            similarity_score = round(100 * np.exp(-overall_dist), 1)

            result = {
                "video": video_name,
                "folder": folder_name,
                "stroke_type": stroke_label,
                "dtw_distance_median_ref": round(float(overall_dist), 4),
                "dtw_distance_multi_ref_avg": round(float(np.mean(multi_dists)), 4),
                "similarity_score": similarity_score,
                "median_reference": median_name,
                "per_joint_distances": {k: round(float(v), 4) for k, v in per_joint.items()},
            }
            results.append(result)

            best_joint = min(per_joint, key=per_joint.get)
            worst_joint = max(per_joint, key=per_joint.get)
            
            print(f"    {video_name}")
            print(f"      DTW distance: {overall_dist:.4f}  |  Similarity: {similarity_score}%")
            print(f"      Best joint:  {best_joint} ({per_joint[best_joint]:.4f})")
            print(f"      Worst joint: {worst_joint} ({per_joint[worst_joint]:.4f})")
    
    return results


def print_summary_statistics(all_results):
    print(f"\n{'=' * 60}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 60}")
    
    folders = set(r["folder"] for r in all_results)
    
    print(f"\n{'Folder':<40} {'Count':>6} {'Avg DTW':>10} {'Avg Sim':>10}")
    print("-" * 70)
    
    for folder in sorted(folders):
        folder_results = [r for r in all_results if r["folder"] == folder]
        avg_dtw = np.mean([r["dtw_distance_median_ref"] for r in folder_results])
        avg_sim = np.mean([r["similarity_score"] for r in folder_results])
        print(f"  {folder:<38} {len(folder_results):>6} {avg_dtw:>10.4f} {avg_sim:>9.1f}%")
    
    print(f"\n{'Stroke Type':<40} {'Count':>6} {'Avg DTW':>10} {'Avg Sim':>10}")
    print("-" * 70)
    
    for stroke in ["forehand", "backhand"]:
        stroke_results = [r for r in all_results if r["stroke_type"] == stroke]
        if stroke_results:
            avg_dtw = np.mean([r["dtw_distance_median_ref"] for r in stroke_results])
            avg_sim = np.mean([r["similarity_score"] for r in stroke_results])
            print(f"  {stroke:<38} {len(stroke_results):>6} {avg_dtw:>10.4f} {avg_sim:>9.1f}%")
    
    print(f"\nPER-JOINT AVERAGE DTW DISTANCES (all strokes combined)")
    print("-" * 50)
    
    joint_averages = {}
    for joint in JOINT_NAMES:
        distances = [r["per_joint_distances"][joint] for r in all_results]
        joint_averages[joint] = np.mean(distances)
    
    sorted_joints = sorted(joint_averages.items(), key=lambda x: x[1], reverse=True)
    
    for joint_name, avg_dist in sorted_joints:
        bar = "█" * int(avg_dist * 50)
        print(f"  {joint_name:<20} {avg_dist:.4f}  {bar}")


def main():
    print("=" * 60)
    print("TENNIS STROKE DTW COMPARISON")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    all_results.extend(compare_stroke_type(
        [PREPROCESSED_DIR / "forehand_tennis_with_ball", PREPROCESSED_DIR / "forehand_tennis_without_ball"],
        PREPROCESSED_DIR / "thetis" / "forehand", "forehand"
    ))
    all_results.extend(compare_stroke_type(
        [PREPROCESSED_DIR / "backhand_tennis_with_ball", PREPROCESSED_DIR / "backhand_tennis_without_ball"],
        PREPROCESSED_DIR / "thetis" / "backhand", "backhand"
    ))

    if all_results:
        print_summary_statistics(all_results)

    json_path = RESULTS_DIR / "dtw_comparison_results.json"
    with open(str(json_path), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    csv_path = RESULTS_DIR / "dtw_comparison_results.csv"
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "folder", "stroke_type", "dtw_distance", "similarity_score"]
                        + [f"dtw_{j}" for j in JOINT_NAMES])
        for r in all_results:
            row = [
                r["video"],
                r["folder"],
                r["stroke_type"],
                r["dtw_distance_median_ref"],
                r["similarity_score"],
            ]
            row.extend([r["per_joint_distances"][joint] for joint in JOINT_NAMES])
            writer.writerow(row)
    
    print(f"CSV results saved to: {csv_path}")
    
    print(f"\n{'=' * 60}")
    print("DTW COMPARISON COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
