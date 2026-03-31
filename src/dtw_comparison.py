"""
Tennis Stroke DTW Comparison
Compares preprocessed personal stroke keypoints against THETIS expert
references using Dynamic Time Warping.

Produces:
- Overall DTW distance per video (lower = more similar to expert)
- Per-joint DTW distances (which body parts differ most)
- Summary statistics grouped by stroke type and quality level
"""

import numpy as np
import json
import csv
from pathlib import Path
from dtw import dtw

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Joint names matching the preprocessing output (12 joints)
JOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def load_preprocessed_files(folder_path):
    """Load all preprocessed .npy files from a folder."""
    files = {}
    folder = Path(folder_path)
    if not folder.exists():
        return files
    
    for f in sorted(folder.iterdir()):
        if f.name.endswith("_preprocessed.npy"):
            keypoints = np.load(str(f))
            files[f.stem] = keypoints
    
    return files


def compute_dtw_distance(sequence_a, sequence_b):
    """
    Compute DTW distance between two keypoint sequences.
    
    Args:
        sequence_a: numpy array of shape (frames_a, 12, 2)
        sequence_b: numpy array of shape (frames_b, 12, 2)
    
    Returns:
        overall_distance: normalised DTW distance across all joints
        per_joint_distances: dict mapping joint name to its DTW distance
    """
    # Flatten each frame from (12, 2) to (24,) for overall DTW
    flat_a = sequence_a.reshape(len(sequence_a), -1)
    flat_b = sequence_b.reshape(len(sequence_b), -1)
    
    # Overall DTW distance
    alignment = dtw(flat_a, flat_b, keep_internals=False)
    overall_distance = alignment.normalizedDistance
    
    # Per-joint DTW distances
    per_joint_distances = {}
    for j, joint_name in enumerate(JOINT_NAMES):
        joint_a = sequence_a[:, j, :]  # shape (frames_a, 2)
        joint_b = sequence_b[:, j, :]  # shape (frames_b, 2)
        
        joint_alignment = dtw(joint_a, joint_b, keep_internals=False)
        per_joint_distances[joint_name] = joint_alignment.normalizedDistance
    
    return overall_distance, per_joint_distances


def compute_average_reference(reference_files):
    """
    Compute the average DTW distance when comparing a stroke against
    ALL expert references, not just one. Returns the median reference
    sequence (the one closest to all others) for use as a representative.
    
    Args:
        reference_files: dict of {name: keypoints_array}
    
    Returns:
        median_name: name of the most representative expert sequence
        median_keypoints: the keypoints of the median reference
    """
    names = list(reference_files.keys())
    
    if len(names) <= 1:
        name = names[0]
        return name, reference_files[name]
    
    # Sample up to 20 references for speed
    if len(names) > 20:
        sample_indices = np.random.choice(len(names), 20, replace=False)
        sampled_names = [names[i] for i in sample_indices]
    else:
        sampled_names = names
    
    # Find the median reference (lowest total distance to all others)
    total_distances = {}
    for i, name_a in enumerate(sampled_names):
        total_dist = 0
        flat_a = reference_files[name_a].reshape(len(reference_files[name_a]), -1)
        for j, name_b in enumerate(sampled_names):
            if i != j:
                flat_b = reference_files[name_b].reshape(len(reference_files[name_b]), -1)
                alignment = dtw(flat_a, flat_b, keep_internals=False)
                total_dist += alignment.normalizedDistance
        total_distances[name_a] = total_dist
    
    median_name = min(total_distances, key=total_distances.get)
    return median_name, reference_files[median_name]


def compare_stroke_type(personal_folders, thetis_folder, stroke_label):
    """
    Compare all personal recordings of a stroke type against THETIS experts.
    
    Args:
        personal_folders: list of folder paths containing personal recordings
        thetis_folder: path to THETIS expert preprocessed files
        stroke_label: "forehand" or "backhand"
    
    Returns:
        results: list of dictionaries with comparison results
    """
    print(f"\n{'=' * 50}")
    print(f"COMPARING: {stroke_label.upper()}")
    print(f"{'=' * 50}")
    
    # Load THETIS expert references
    thetis_files = load_preprocessed_files(thetis_folder)
    if not thetis_files:
        print(f"  ERROR: No THETIS references found in {thetis_folder}")
        return []
    
    print(f"  THETIS expert references loaded: {len(thetis_files)}")
    
    # Find the median (most representative) expert reference
    print(f"  Finding median expert reference...")
    median_name, median_ref = compute_average_reference(thetis_files)
    print(f"  Median reference: {median_name}")
    
    # Also select 5 random references for multi-reference comparison
    ref_names = list(thetis_files.keys())
    num_refs = min(5, len(ref_names))
    sample_refs = np.random.choice(ref_names, num_refs, replace=False)
    
    results = []
    
    # Process each personal folder
    for personal_folder in personal_folders:
        personal_files = load_preprocessed_files(personal_folder)
        folder_name = Path(personal_folder).name
        
        if not personal_files:
            continue
        
        print(f"\n  --- {folder_name} ({len(personal_files)} files) ---")
        
        for video_name, personal_keypoints in personal_files.items():
            # Compare against median reference
            overall_dist, per_joint = compute_dtw_distance(personal_keypoints, median_ref)
            
            # Compare against multiple references and average
            multi_ref_distances = []
            for ref_name in sample_refs:
                ref_kp = thetis_files[ref_name]
                dist, _ = compute_dtw_distance(personal_keypoints, ref_kp)
                multi_ref_distances.append(dist)
            
            avg_multi_ref_dist = np.mean(multi_ref_distances)
            
            # Convert overall distance to a similarity score (0-100)
            # Using exponential decay: similarity = 100 * exp(-distance)
            similarity_score = round(100 * np.exp(-overall_dist), 1)
            
            result = {
                "video": video_name,
                "folder": folder_name,
                "stroke_type": stroke_label,
                "dtw_distance_median_ref": round(float(overall_dist), 4),
                "dtw_distance_multi_ref_avg": round(float(avg_multi_ref_dist), 4),
                "similarity_score": similarity_score,
                "median_reference": median_name,
                "per_joint_distances": {k: round(float(v), 4) for k, v in per_joint.items()},
            }
            results.append(result)
            
            # Find best and worst joints
            best_joint = min(per_joint, key=per_joint.get)
            worst_joint = max(per_joint, key=per_joint.get)
            
            print(f"    {video_name}")
            print(f"      DTW distance: {overall_dist:.4f}  |  Similarity: {similarity_score}%")
            print(f"      Best joint:  {best_joint} ({per_joint[best_joint]:.4f})")
            print(f"      Worst joint: {worst_joint} ({per_joint[worst_joint]:.4f})")
    
    return results


def print_summary_statistics(all_results):
    """Print summary tables of the DTW comparison results."""
    
    print(f"\n{'=' * 60}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 60}")
    
    # Group by folder
    folders = set(r["folder"] for r in all_results)
    
    print(f"\n{'Folder':<40} {'Count':>6} {'Avg DTW':>10} {'Avg Sim':>10}")
    print("-" * 70)
    
    for folder in sorted(folders):
        folder_results = [r for r in all_results if r["folder"] == folder]
        avg_dtw = np.mean([r["dtw_distance_median_ref"] for r in folder_results])
        avg_sim = np.mean([r["similarity_score"] for r in folder_results])
        print(f"  {folder:<38} {len(folder_results):>6} {avg_dtw:>10.4f} {avg_sim:>9.1f}%")
    
    # Group by stroke type
    print(f"\n{'Stroke Type':<40} {'Count':>6} {'Avg DTW':>10} {'Avg Sim':>10}")
    print("-" * 70)
    
    for stroke in ["forehand", "backhand"]:
        stroke_results = [r for r in all_results if r["stroke_type"] == stroke]
        if stroke_results:
            avg_dtw = np.mean([r["dtw_distance_median_ref"] for r in stroke_results])
            avg_sim = np.mean([r["similarity_score"] for r in stroke_results])
            print(f"  {stroke:<38} {len(stroke_results):>6} {avg_dtw:>10.4f} {avg_sim:>9.1f}%")
    
    # Per-joint analysis across all results
    print(f"\nPER-JOINT AVERAGE DTW DISTANCES (all strokes combined)")
    print("-" * 50)
    
    joint_averages = {}
    for joint in JOINT_NAMES:
        distances = [r["per_joint_distances"][joint] for r in all_results]
        joint_averages[joint] = np.mean(distances)
    
    # Sort by distance (worst first)
    sorted_joints = sorted(joint_averages.items(), key=lambda x: x[1], reverse=True)
    
    for joint_name, avg_dist in sorted_joints:
        bar = "█" * int(avg_dist * 50)
        print(f"  {joint_name:<20} {avg_dist:.4f}  {bar}")


def main():
    """Run DTW comparison for all stroke types."""
    
    print("=" * 60)
    print("TENNIS STROKE DTW COMPARISON")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # ---- Forehand comparison ----
    forehand_personal = [
        PREPROCESSED_DIR / "forehand_tennis_with_ball",
        PREPROCESSED_DIR / "forehand_tennis_without_ball",
    ]
    forehand_thetis = PREPROCESSED_DIR / "thetis" / "forehand"
    
    forehand_results = compare_stroke_type(forehand_personal, forehand_thetis, "forehand")
    all_results.extend(forehand_results)
    
    # ---- Backhand comparison ----
    backhand_personal = [
        PREPROCESSED_DIR / "backhand_tennis_with_ball",
        PREPROCESSED_DIR / "backhand_tennis_without_ball",
    ]
    backhand_thetis = PREPROCESSED_DIR / "thetis" / "backhand"
    
    backhand_results = compare_stroke_type(backhand_personal, backhand_thetis, "backhand")
    all_results.extend(backhand_results)
    
    # ---- Print summary ----
    if all_results:
        print_summary_statistics(all_results)
    
    # ---- Save results ----
    # Save as JSON
    json_path = RESULTS_DIR / "dtw_comparison_results.json"
    with open(str(json_path), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {json_path}")
    
    # Save as CSV for easy analysis
    csv_path = RESULTS_DIR / "dtw_comparison_results.csv"
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header
        header = ["video", "folder", "stroke_type", "dtw_distance", "similarity_score"]
        header.extend([f"dtw_{joint}" for joint in JOINT_NAMES])
        writer.writerow(header)
        
        # Data rows
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
    np.random.seed(42)  # For reproducibility
    main()
