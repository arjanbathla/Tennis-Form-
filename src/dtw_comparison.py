"""
DTW comparison: personal stroke keypoints vs THETIS expert references.
Produces overall + per-joint distances and dumps a csv/json summary.
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


def load_preprocessed(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return {}
    return {f.stem: np.load(str(f)) for f in sorted(folder.iterdir())
            if f.name.endswith("_preprocessed.npy")}


def compute_dtw(a, b):
    fa, fb = a.reshape(len(a), -1), b.reshape(len(b), -1)
    overall = dtw(fa, fb, keep_internals=False).normalizedDistance
    per_joint = {jn: dtw(a[:, j, :], b[:, j, :], keep_internals=False).normalizedDistance
                 for j, jn in enumerate(JOINT_NAMES)}
    return overall, per_joint


def pick_median_reference(refs):
    # pick the ref with the smallest summed DTW distance to all others;
    # cap at 20 to keep the pairwise loop bounded
    names = list(refs.keys())
    if len(names) <= 1:
        return names[0], refs[names[0]]

    if len(names) > 20:
        names = [names[i] for i in np.random.choice(len(names), 20, replace=False)]

    totals = {}
    for i, na in enumerate(names):
        fa = refs[na].reshape(len(refs[na]), -1)
        totals[na] = sum(
            dtw(fa, refs[nb].reshape(len(refs[nb]), -1), keep_internals=False).normalizedDistance
            for j, nb in enumerate(names) if i != j
        )

    best = min(totals, key=totals.get)
    return best, refs[best]


def compare_stroke_type(personal_folders, thetis_folder, stroke_label):
    print(f"\n{stroke_label}")
    thetis = load_preprocessed(thetis_folder)
    if not thetis:
        print(f"  no expert refs in {thetis_folder}")
        return []

    print(f"  {len(thetis)} expert refs, finding median...")
    median_name, median_ref = pick_median_reference(thetis)
    print(f"  median: {median_name}")

    sample_refs = np.random.choice(list(thetis.keys()), min(5, len(thetis)), replace=False)
    results = []

    for personal_folder in personal_folders:
        personal = load_preprocessed(personal_folder)
        folder_name = Path(personal_folder).name
        if not personal:
            continue

        print(f"\n  {folder_name} ({len(personal)} files)")
        for name, kp in personal.items():
            overall, per_joint = compute_dtw(kp, median_ref)
            multi = [compute_dtw(kp, thetis[r])[0] for r in sample_refs]
            sim = round(100 * np.exp(-overall), 1)

            results.append({
                "video": name,
                "folder": folder_name,
                "stroke_type": stroke_label,
                "dtw_distance_median_ref": round(float(overall), 4),
                "dtw_distance_multi_ref_avg": round(float(np.mean(multi)), 4),
                "similarity_score": sim,
                "median_reference": median_name,
                "per_joint_distances": {k: round(float(v), 4) for k, v in per_joint.items()},
            })

            best = min(per_joint, key=per_joint.get)
            worst = max(per_joint, key=per_joint.get)
            print(f"    {name}: dtw={overall:.4f} sim={sim}% "
                  f"best={best}({per_joint[best]:.4f}) worst={worst}({per_joint[worst]:.4f})")

    return results


def print_summary(results):
    print("\nsummary")
    print(f"\n{'folder':<40} {'n':>4} {'avg dtw':>10} {'avg sim':>8}")
    for folder in sorted({r["folder"] for r in results}):
        fr = [r for r in results if r["folder"] == folder]
        adtw = np.mean([r["dtw_distance_median_ref"] for r in fr])
        asim = np.mean([r["similarity_score"] for r in fr])
        print(f"  {folder:<38} {len(fr):>4} {adtw:>10.4f} {asim:>7.1f}%")

    print(f"\n{'stroke':<40} {'n':>4} {'avg dtw':>10} {'avg sim':>8}")
    for stroke in ["forehand", "backhand"]:
        sr = [r for r in results if r["stroke_type"] == stroke]
        if sr:
            adtw = np.mean([r["dtw_distance_median_ref"] for r in sr])
            asim = np.mean([r["similarity_score"] for r in sr])
            print(f"  {stroke:<38} {len(sr):>4} {adtw:>10.4f} {asim:>7.1f}%")

    print("\nper-joint avg dtw")
    joint_avgs = {j: np.mean([r["per_joint_distances"][j] for r in results]) for j in JOINT_NAMES}
    for j, d in sorted(joint_avgs.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(d * 50)
        print(f"  {j:<20} {d:.4f}  {bar}")


def main():
    print("DTW comparison")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    all_results.extend(compare_stroke_type(
        [PREPROCESSED_DIR / "forehand_tennis_with_ball",
         PREPROCESSED_DIR / "forehand_tennis_without_ball"],
        PREPROCESSED_DIR / "thetis" / "forehand", "forehand"))
    all_results.extend(compare_stroke_type(
        [PREPROCESSED_DIR / "backhand_tennis_with_ball",
         PREPROCESSED_DIR / "backhand_tennis_without_ball"],
        PREPROCESSED_DIR / "thetis" / "backhand", "backhand"))

    if all_results:
        print_summary(all_results)

    json_path = RESULTS_DIR / "dtw_comparison_results.json"
    with open(str(json_path), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\njson -> {json_path}")

    csv_path = RESULTS_DIR / "dtw_comparison_results.csv"
    with open(str(csv_path), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "folder", "stroke_type", "dtw_distance", "similarity_score"]
                   + [f"dtw_{j}" for j in JOINT_NAMES])
        for r in all_results:
            w.writerow([r["video"], r["folder"], r["stroke_type"],
                        r["dtw_distance_median_ref"], r["similarity_score"]]
                       + [r["per_joint_distances"][j] for j in JOINT_NAMES])
    print(f"csv -> {csv_path}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
