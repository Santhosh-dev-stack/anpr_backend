"""Diff two run_pipeline_offline.py CSVs (e.g. local CPU run vs. Colab GPU
run on the same video + same models) to check whether the set of validated
plate readings actually matches, not just inference speed.

Track IDs are assigned independently in each run and aren't comparable
directly, so this compares by the *set of validated plate strings* each run
produced overall, plus a per-track count/confidence sanity check.

Usage:
    python compare_pipeline_runs.py local_results.csv colab_results.csv
"""

import argparse
import csv


def load_plates(path: str) -> dict[str, list[float]]:
    """plate text -> list of confidences it was read at (one run can read
    the same plate on more than one track, e.g. if tracking loses it briefly)."""
    plates: dict[str, list[float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for text in row["all_readings"].split(";"):
                text = text.strip()
                if text:
                    plates.setdefault(text, []).append(float(row["best_confidence"]))
    return plates


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two pipeline result CSVs")
    parser.add_argument("run_a", help="e.g. local_results.csv")
    parser.add_argument("run_b", help="e.g. colab_results.csv")
    args = parser.parse_args()

    a = load_plates(args.run_a)
    b = load_plates(args.run_b)

    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    both = sorted(set(a) & set(b))

    print(f"{args.run_a}: {len(a)} distinct validated plates")
    print(f"{args.run_b}: {len(b)} distinct validated plates")
    print(f"Agree on: {len(both)} plates\n")

    if both:
        print("=== Matched plates (confidence comparison) ===")
        for plate in both:
            conf_a = max(a[plate])
            conf_b = max(b[plate])
            print(f"  {plate:12} {args.run_a}={conf_a:.3f}  {args.run_b}={conf_b:.3f}  Δ={conf_b - conf_a:+.3f}")

    if only_a:
        print(f"\n=== Only in {args.run_a} ({len(only_a)}) ===")
        for plate in only_a:
            print(f"  {plate}  (conf={max(a[plate]):.3f})")

    if only_b:
        print(f"\n=== Only in {args.run_b} ({len(only_b)}) ===")
        for plate in only_b:
            print(f"  {plate}  (conf={max(b[plate]):.3f})")

    total = len(set(a) | set(b))
    agreement = len(both) / total if total else 1.0
    print(f"\nOverall plate-set agreement: {agreement:.1%}")


if __name__ == "__main__":
    main()
