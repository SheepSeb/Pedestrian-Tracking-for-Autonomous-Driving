from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import motmetrics as mm
import numpy as np
import pandas as pd

import deep_sort_app

# NumPy 2.0 removed np.asfarray; motmetrics still calls it. Provide a shim.
if not hasattr(np, "asfarray"):
    def _asfarray(a, dtype=None):
        return np.asarray(a, dtype=float if dtype is None else dtype)
    np.asfarray = _asfarray  # type: ignore[attr-defined]


def _run_sequence(
    sequence: str,
    mot_dir: Path,
    detection_dir: Path,
    output_dir: Path,
    min_confidence: float,
    nms_max_overlap: float,
    min_detection_height: int,
    max_cosine_distance: float,
    nn_budget: int | None,
    use_bloom_filter: bool,
    expected_tracks: int,
    bloom_false_positive_rate: float,
) -> float:
    """Run Deep SORT on a single sequence and return runtime in seconds."""
    sequence_dir = mot_dir / sequence
    detection_file = detection_dir / f"{sequence}.npy"
    output_file = output_dir / f"{sequence}.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    deep_sort_app.run(
        os.fspath(sequence_dir),
        os.fspath(detection_file),
        os.fspath(output_file),
        min_confidence,
        nms_max_overlap,
        min_detection_height,
        max_cosine_distance,
        nn_budget,
        display=False,
        use_bloom_filter=use_bloom_filter,
        expected_tracks=expected_tracks,
        bloom_false_positive_rate=bloom_false_positive_rate,
    )
    return time.perf_counter() - start


def _load_gt(path: Path) -> pd.DataFrame:
    """Load and filter MOT16 ground-truth annotations (people only, conf > 0)."""
    gt = mm.io.load_motchallenge(os.fspath(path))
    if "Conf" in gt.columns:
        gt = gt[gt["Conf"] > 0]
    if "ClassId" in gt.columns:
        gt = gt[gt["ClassId"] == 1]
    return gt


def _load_hyp(path: Path) -> pd.DataFrame:
    """Load tracker hypotheses."""
    hyp = mm.io.load_motchallenge(os.fspath(path))
    if "Conf" in hyp.columns:
        hyp = hyp[hyp["Conf"] > 0]
    return hyp


def _evaluate_sequences(
    mot_dir: Path, tracks_dir: Path, sequences: List[str], iou_threshold: float, runtimes: Dict[str, float]
) -> pd.DataFrame:
    accs: List[mm.MOTAccumulator] = []
    names: List[str] = []
    for seq in sequences:
        gt_path = mot_dir / seq / "gt" / "gt.txt"
        hyp_path = tracks_dir / f"{seq}.txt"
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing ground truth: {gt_path}")
        if not hyp_path.exists():
            raise FileNotFoundError(f"Missing tracker output: {hyp_path}")
        gt = _load_gt(gt_path)
        hyp = _load_hyp(hyp_path)
        acc = mm.utils.compare_to_groundtruth(gt, hyp, "iou", distth=iou_threshold)
        accs.append(acc)
        names.append(seq)

    mh = mm.metrics.create()
    metrics = [
        "mota",
        "motp",
        "num_switches",
        "num_fragmentations",
        "num_false_positives",
        "num_misses",
        "mostly_tracked",
        "mostly_lost",
        "num_unique_objects",
    ]
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)

    # Convert to a more familiar reporting style:
    summary = summary.rename(
        columns={
            "num_switches": "ID",
            "num_fragmentations": "FM",
            "num_false_positives": "FP",
            "num_misses": "FN",
            "motp": "MOTP_raw",
        }
    )

    # MOTP from motmetrics is an average distance (1 - IoU). Report as IoU-style.
    summary["MOTP"] = 1.0 - summary["MOTP_raw"]
    summary["MT"] = summary["mostly_tracked"] * summary["num_unique_objects"]
    summary["ML"] = summary["mostly_lost"] * summary["num_unique_objects"]

    runtime_series = pd.Series(runtimes, name="RuntimeSeconds")
    summary = summary.join(runtime_series, how="left")

    desired_cols = [
        "mota",
        "MOTP",
        "MT",
        "ML",
        "ID",
        "FM",
        "FP",
        "FN",
        "RuntimeSeconds",
    ]
    for col in desired_cols:
        if col not in summary.columns:
            summary[col] = np.nan
    return summary[desired_cols]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Deep SORT on MOT16 and report MOT metrics (MOTA, MOTP, MT, ML, ID, FM, FP, FN, Runtime)."
    )
    parser.add_argument("--mot_dir", type=Path, default=Path("MOT16") / "train", help="Path to MOT16 train root.")
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional list of sequence names (e.g., MOT16-09) to evaluate. Defaults to all sequences under mot_dir.",
    )
    parser.add_argument(
        "--detection_dir",
        type=Path,
        required=True,
        help="Directory containing detection .npy files (one per sequence).",
    )
    parser.add_argument(
        "--tracks_dir",
        type=Path,
        default=Path("experiments") / "mot16_tracks",
        help="Where to store tracker outputs.",
    )
    parser.add_argument(
        "--metrics_out",
        type=Path,
        default=Path("experiments") / "mot16_metrics.csv",
        help="CSV file to store the metrics summary.",
    )
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for matching.")
    parser.add_argument("--min_confidence", type=float, default=0.3, help="Detection confidence threshold.")
    parser.add_argument("--min_detection_height", type=int, default=0, help="Minimum detection height.")
    parser.add_argument("--nms_max_overlap", type=float, default=1.0, help="NMS IoU threshold.")
    parser.add_argument("--max_cosine_distance", type=float, default=0.2, help="Appearance gating threshold.")
    parser.add_argument("--nn_budget", type=int, default=100, help="Appearance gallery size; use 0 for unlimited.")
    parser.add_argument("--no_bloom_filter", action="store_true", help="Disable Bloom filter optimization.")
    parser.add_argument("--expected_tracks", type=int, default=1000, help="Bloom filter expected track count.")
    parser.add_argument("--bloom_false_positive_rate", type=float, default=0.01, help="Bloom filter FPR.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mot_dir = args.mot_dir
    detection_dir = args.detection_dir
    tracks_dir = args.tracks_dir
    tracks_dir.mkdir(parents=True, exist_ok=True)

    sequences = (
        sorted([d for d in os.listdir(mot_dir) if (mot_dir / d).is_dir()])
        if args.sequences is None
        else args.sequences
    )

    runtimes: Dict[str, float] = {}
    for seq in sequences:
        print(f"[tracking] {seq}")
        runtime = _run_sequence(
            seq,
            mot_dir,
            detection_dir,
            tracks_dir,
            args.min_confidence,
            args.nms_max_overlap,
            args.min_detection_height,
            args.max_cosine_distance,
            args.nn_budget if args.nn_budget > 0 else None,
            use_bloom_filter=not args.no_bloom_filter,
            expected_tracks=args.expected_tracks,
            bloom_false_positive_rate=args.bloom_false_positive_rate,
        )
        runtimes[seq] = runtime
        print(f"[done] {seq} runtime: {runtime:.2f}s")

    summary = _evaluate_sequences(mot_dir, tracks_dir, sequences, args.iou_threshold, runtimes)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.metrics_out)

    print("\n=== MOT16 Metrics (higher MOTA/MOTP better; lower ID/FM/FP/FN better) ===")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    overall = summary.loc["OVERALL"]
    print("\n[overall]")
    for key in ["mota", "MOTP", "MT", "ML", "ID", "FM", "FP", "FN", "RuntimeSeconds"]:
        val = overall.get(key, np.nan)
        print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")


if __name__ == "__main__":
    main()
