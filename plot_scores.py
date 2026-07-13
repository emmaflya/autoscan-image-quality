"""
Analyse USAF MTF scores CSV and produce plots:

1. Histogram: average number of images that resolve each frequency
   (resolved = both H_MTF >= 0.16 and V_MTF >= 0.16), averaged over all targets.

2. Per-frequency distribution: two subplots (H_MTF left, V_MTF right)
   showing the distribution of scores across all images/targets.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MTF_THRESHOLD = 0.16 # Threshold for considering a frequency resolved (both H and V MTF must exceed this value)


def plot_resolution_histogram(df, output_dir):
    """
    Bar chart: for each frequency, average number of images resolving it.
    'Resolved' means both h_mtf >= threshold AND v_mtf >= threshold.
    Averaged over all targets.
    """
    df = df.copy()
    df["resolved"] = (
        df["h_mtf"].notna()
        & df["v_mtf"].notna()
        & (df["h_mtf"] >= MTF_THRESHOLD)
        & (df["v_mtf"] >= MTF_THRESHOLD)
    )

    grouped = (
        df.groupby(["target", "frequency"])["resolved"]
        .sum()
        .reset_index()
        .rename(columns={"resolved": "num_resolved"})
    )

    avg = grouped.groupby("frequency")["num_resolved"].mean()
    avg = avg.sort_index(ascending=False)  # largest frequency first

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(avg.index.astype(str), avg.values, color="#3b82f6", edgecolor="black")
    ax.set_xlabel("Spatial frequency (lp/mm)")
    ax.set_ylabel("Avg. number of images resolving target")
    ax.set_title(f"Resolution count per frequency (threshold = {MTF_THRESHOLD})")
    ax.set_ylim(0, avg.values.max() * 1.2 if avg.values.max() > 0 else 1)

    for bar, val in zip(bars, avg.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02 * ax.get_ylim()[1],
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    out_path = output_dir / "resolution_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_mtf_distributions(df, output_dir):
    """
    For each frequency: 2 subplots — H_MTF distribution (left), V_MTF distribution (right).
    """
    frequencies = sorted(df["frequency"].unique(), reverse=True)

    for freq in frequencies:
        sub = df[df["frequency"] == freq]

        h_scores = sub["h_mtf"].dropna()
        v_scores = sub["v_mtf"].dropna()
        h_scores = h_scores[h_scores >= 0]
        v_scores = v_scores[v_scores >= 0]

        fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

        # Common bin range
        all_vals = pd.concat([h_scores, v_scores])
        if len(all_vals) == 0:
            plt.close(fig)
            continue
        bin_max = min(1.0, all_vals.max() + 0.05)
        bins = np.linspace(0, bin_max, 25)

        # H_MTF
        ax_h.hist(h_scores, bins=bins, color="#ef4444", edgecolor="black", alpha=0.8)
        ax_h.axvline(MTF_THRESHOLD, color="black", linestyle="--", linewidth=1, label=f"Threshold ({MTF_THRESHOLD})")
        ax_h.set_xlabel("H_MTF")
        ax_h.set_ylabel("Count")
        ax_h.set_title(f"H-lines MTF — freq {freq}")
        ax_h.legend(fontsize=8)

        # V_MTF
        ax_v.hist(v_scores, bins=bins, color="#3b82f6", edgecolor="black", alpha=0.8)
        ax_v.axvline(MTF_THRESHOLD, color="black", linestyle="--", linewidth=1, label=f"Threshold ({MTF_THRESHOLD})")
        ax_v.set_xlabel("V_MTF")
        ax_v.set_ylabel("Count")
        ax_v.set_title(f"V-lines MTF — freq {freq}")
        ax_v.legend(fontsize=8)

        plt.tight_layout()
        out_path = output_dir / f"mtf_distribution_freq_{freq}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot USAF MTF score analysis")
    parser.add_argument(
        "csv_file",
        help="Path to usaf_scores.csv",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory for plots (default: same as CSV)",
        default=None,
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    output_dir = Path(args.output) if args.output else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Ensure numeric types
    df["h_mtf"] = pd.to_numeric(df["h_mtf"], errors="coerce")
    df["v_mtf"] = pd.to_numeric(df["v_mtf"], errors="coerce")
    df["frequency"] = df["frequency"].astype(str)

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Frequencies: {sorted(df['frequency'].unique(), reverse=True)}")
    print(f"Targets: {sorted(df['target'].unique())}")
    print(f"Images: {df['image'].nunique()}")
    print()

    plot_resolution_histogram(df, output_dir)
    plot_mtf_distributions(df, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
