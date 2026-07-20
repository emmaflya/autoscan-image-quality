"""
Analyse USAF MTF scores CSV(s) and produce plots:

1. Resolution histogram (bar chart): percentage resolved per frequency.
2. Per-frequency MTF distribution (line): H and V subplots.
3. Per-frequency radius plot (line): resolved % vs distance to center.
4. Per-frequency POI plot (line): distribution of per-image resolved %.

Supports multiple CSV files for comparison — one line/bar per CSV with
consistent color coding across all plots.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MTF_THRESHOLD = 0.16  # Threshold for considering a frequency resolved

# Color palette for comparing multiple CSVs (up to 10)
COLORS = [
    "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
    "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
]


def _get_label(csv_path):
    """Derive a short label from the CSV path (parent folder name)."""
    return csv_path.parent.name


def plot_resolution_histogram(datasets, output_dir):
    """
    Bar chart: for each frequency, percentage of detected targets that resolve it.
    One group of bars per frequency, one bar per dataset.
    """
    frequencies = None
    all_pcts = []

    for df, label in datasets:
        df = df.copy()
        df["resolved"] = (
            df["h_mtf"].notna()
            & df["v_mtf"].notna()
            & (df["h_mtf"] >= MTF_THRESHOLD)
            & (df["v_mtf"] >= MTF_THRESHOLD)
        )

        # For each individual target at each frequency:
        # d = number of images that detect it, r = number that resolve it
        # percentage = r/d * 100. Then average across all targets.
        target_pct = (
            df.groupby(["target", "frequency"])
            .agg(d=("resolved", "count"), r=("resolved", "sum"))
            .reset_index()
        )
        target_pct["pct"] = target_pct["r"] / target_pct["d"] * 100.0

        pct = target_pct.groupby("frequency")["pct"].mean()
        pct = pct.sort_index(ascending=False)
        all_pcts.append(pct)

        if frequencies is None:
            frequencies = pct.index.tolist()

    n_datasets = len(datasets)
    n_freq = len(frequencies)
    x = np.arange(n_freq)
    width = 0.8 / n_datasets

    fig, ax = plt.subplots(figsize=(6 + n_datasets, 4))

    for i, ((_, label), pct) in enumerate(zip(datasets, all_pcts)):
        offset = (i - (n_datasets - 1) / 2) * width
        vals = [pct.get(f, 0.0) for f in frequencies]
        bars = ax.bar(x + offset, vals, width, color=COLORS[i % len(COLORS)],
                      edgecolor="black", label=label, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.0f}%",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(float(f)/2) for f in frequencies])
    ax.set_xlabel("GSD target (mm/pix)")
    ax.set_ylabel("Resolved (%)")
    ax.set_title(f"Resolution rate per GSD target (threshold = {MTF_THRESHOLD}), (= all targets)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "resolution_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_mtf_distributions(datasets, output_dir):
    """
    Per-frequency MTF distribution: 2 subplots (H left, V right), one line per dataset.
    """
    all_freqs = set()
    for df, _ in datasets:
        all_freqs.update(df["frequency"].unique())
    frequencies = sorted(all_freqs, reverse=True)

    for freq in frequencies:
        fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

        bin_max = 1.0
        bins = np.linspace(0, bin_max, 25)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        for i, (df, label) in enumerate(datasets):
            sub = df[df["frequency"] == freq]
            color = COLORS[i % len(COLORS)]

            h_scores = sub["h_mtf"].dropna()
            h_scores = h_scores[h_scores >= 0]
            if len(h_scores) > 0:
                counts_h, _ = np.histogram(h_scores, bins=bins)
                pct_h = counts_h / len(h_scores) * 100.0
                ax_h.plot(bin_centers, pct_h, color=color, linewidth=1.5,
                          marker="o", markersize=3, label=label)

            v_scores = sub["v_mtf"].dropna()
            v_scores = v_scores[v_scores >= 0]
            if len(v_scores) > 0:
                counts_v, _ = np.histogram(v_scores, bins=bins)
                pct_v = counts_v / len(v_scores) * 100.0
                ax_v.plot(bin_centers, pct_v, color=color, linewidth=1.5,
                          marker="o", markersize=3, label=label)

        ax_h.axvline(MTF_THRESHOLD, color="black", linestyle="--", linewidth=1,
                     label=f"Threshold ({MTF_THRESHOLD})")
        ax_h.set_xlabel("Horizontal lines MTF score")
        ax_h.set_ylabel("Percentage of resolved points (%)")
        ax_h.set_title(f"H-lines MTF")
        ax_h.legend(fontsize=7)

        ax_v.axvline(MTF_THRESHOLD, color="black", linestyle="--", linewidth=1,
                     label=f"Threshold ({MTF_THRESHOLD})")
        ax_v.set_xlabel("Vertical lines MTF score")
        ax_v.set_title(f"V-lines MTF")
        ax_v.legend(fontsize=7)
        
        fig.suptitle(f"Quality score distribution — GSD {float(freq)/2:.2f} mm/pix, total of {len(sub)} points (= all images, all targets)", fontsize=12)

        plt.tight_layout()
        out_path = output_dir / f"mtf_distribution_gsd_{float(freq)/2:.2f}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_radius_histogram(datasets, output_dir):
    """
    Per-frequency radius plot: resolved % vs distance to center, one line per dataset.
    """
    has_radius = any("radius" in df.columns for df, _ in datasets)
    if not has_radius:
        print("Skipping radius histogram: no dataset has 'radius' column.")
        return

    # Global bin range across all datasets
    all_radii = []
    for df, _ in datasets:
        if "radius" in df.columns:
            r = pd.to_numeric(df["radius"], errors="coerce").dropna()
            all_radii.append(r)
    if not all_radii:
        return
    all_radii = pd.concat(all_radii)
    if all_radii.empty:
        return
    bin_max = all_radii.max() * 1.05
    bins = np.linspace(0, bin_max, 15)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    all_freqs = set()
    for df, _ in datasets:
        all_freqs.update(df["frequency"].unique())
    frequencies = sorted(all_freqs, reverse=True)

    for freq in frequencies:
        fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        for i, (df, label) in enumerate(datasets):
            if "radius" not in df.columns:
                continue
            color = COLORS[i % len(COLORS)]
            df_c = df.copy()
            df_c["radius"] = pd.to_numeric(df_c["radius"], errors="coerce")
            df_c["h_resolved"] = df_c["h_mtf"].notna() & (df_c["h_mtf"] >= MTF_THRESHOLD)
            df_c["v_resolved"] = df_c["v_mtf"].notna() & (df_c["v_mtf"] >= MTF_THRESHOLD)

            sub = df_c[df_c["frequency"] == freq]

            h_radii_all = sub["radius"].dropna()
            h_radii_resolved = sub.loc[sub["h_resolved"], "radius"].dropna()
            v_radii_all = sub["radius"].dropna()
            v_radii_resolved = sub.loc[sub["v_resolved"], "radius"].dropna()

            counts_all_h, _ = np.histogram(h_radii_all, bins=bins)
            counts_res_h, _ = np.histogram(h_radii_resolved, bins=bins)
            pct_h = np.where(counts_all_h > 0, counts_res_h / counts_all_h * 100.0, 0.0)

            counts_all_v, _ = np.histogram(v_radii_all, bins=bins)
            counts_res_v, _ = np.histogram(v_radii_resolved, bins=bins)
            pct_v = np.where(counts_all_v > 0, counts_res_v / counts_all_v * 100.0, 0.0)

            ax_h.plot(bin_centers, pct_h, color=color, linewidth=1.5,
                      marker="o", markersize=4, label=label)
            ax_v.plot(bin_centers, pct_v, color=color, linewidth=1.5,
                      marker="o", markersize=4, label=label)

        ax_h.set_xlabel("Distance to image center (pixels)")
        ax_h.set_ylabel("Resolved (%)")
        ax_h.set_title(f"H-lines")
        ax_h.set_ylim(0, 105)
        ax_h.legend(fontsize=7)

        ax_v.set_xlabel("Distance to image center (pixels)")
        ax_v.set_title(f"V-lines")
        ax_v.set_ylim(0, 105)
        ax_v.legend(fontsize=7)
        
        fig.suptitle(f"Percentage of resolved targets per distance — GSD {float(freq)/2:.2f} mm/pix, total of {len(sub)} points (= all images, all targets)", fontsize=12)

        plt.tight_layout()
        out_path = output_dir / f"radius_histogram_gsd_{float(freq)/2:.2f}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_poi_histogram(datasets, output_dir):
    """
    Per-frequency POI plot: distribution of per-image resolved %, one line per dataset.
    """
    all_freqs = set()
    for df, _ in datasets:
        all_freqs.update(df["frequency"].unique())
    frequencies = sorted(all_freqs, reverse=True)

    bins = np.linspace(0, 100, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    for freq in frequencies:
        fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        for i, (df, label) in enumerate(datasets):
            color = COLORS[i % len(COLORS)]
            df_c = df.copy()
            df_c["h_resolved"] = df_c["h_mtf"].notna() & (df_c["h_mtf"] >= MTF_THRESHOLD)
            df_c["v_resolved"] = df_c["v_mtf"].notna() & (df_c["v_mtf"] >= MTF_THRESHOLD)

            sub = df_c[df_c["frequency"] == freq]
            if sub.empty:
                continue

            img_stats = sub.groupby("image").agg(
                h_total=("h_resolved", "count"),
                h_res=("h_resolved", "sum"),
                v_total=("v_resolved", "count"),
                v_res=("v_resolved", "sum"),
            )
            img_stats["h_pct"] = img_stats["h_res"] / img_stats["h_total"] * 100.0
            img_stats["v_pct"] = img_stats["v_res"] / img_stats["v_total"] * 100.0

            counts_h, _ = np.histogram(img_stats["h_pct"], bins=bins)
            counts_v, _ = np.histogram(img_stats["v_pct"], bins=bins)

            ax_h.plot(bin_centers, counts_h, color=color, linewidth=1.5,
                      marker="o", markersize=4, label=label)
            ax_v.plot(bin_centers, counts_v, color=color, linewidth=1.5,
                      marker="o", markersize=4, label=label)

        ax_h.set_xlabel("Resolved (%)")
        ax_h.set_ylabel("Number of Images")
        ax_h.set_title(f"H-lines")
        ax_h.set_xlim(0, 100)
        ax_h.legend(fontsize=7)

        ax_v.set_xlabel("Resolved (%)")
        ax_v.set_ylabel("Number of Images")
        ax_v.set_title(f"V-lines")
        ax_v.set_xlim(0, 100)
        ax_v.legend(fontsize=7)
        fig.suptitle(f"Distribution of per-image resolved % — GSD {float(freq)/2:.2f} mm/pix (=all images)", fontsize=12)
        plt.tight_layout()
        out_path = output_dir / f"image_histogram_gsd_{float(freq)/2:.2f}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_target_resolution_count(datasets, output_dir):
    """
    Per-frequency bar chart: x-axis is number of images that resolve a target
    (including 0), y-axis is number of targets with that count.
    A target is resolved in an image at a given frequency if both H and V
    lines score >= threshold.
    """
    all_freqs = set()
    for df, _ in datasets:
        all_freqs.update(df["frequency"].unique())
    frequencies = sorted(all_freqs, reverse=True)

    for freq in frequencies:
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, (df, label) in enumerate(datasets):
            sub = df[df["frequency"] == freq].copy()
            sub["resolved"] = (
                sub["h_mtf"].notna()
                & sub["v_mtf"].notna()
                & (sub["h_mtf"] >= MTF_THRESHOLD)
                & (sub["v_mtf"] >= MTF_THRESHOLD)
            )

            # For each unique target, count number of images that resolve it
            resolve_counts = (
                sub.groupby("target")["resolved"]
                .sum()
                .astype(int)
            )

            # Count how many targets have each resolve count (including 0)
            max_count = resolve_counts.max() if len(resolve_counts) > 0 else 0
            x_values = np.arange(0, max_count + 1)
            y_values = [(resolve_counts == c).sum() for c in x_values]

            color = COLORS[i % len(COLORS)]
            offset = (i - (len(datasets) - 1) / 2) * (0.8 / len(datasets))
            width = 0.8 / len(datasets)
            bars = ax.bar(x_values + offset, y_values, width, color=color,
                          edgecolor="black", label=label, alpha=0.85)
            for bar, val in zip(bars, y_values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.2,
                        str(val),
                        ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xlabel("Number of images resolving the target")
        ax.set_ylabel("Number of targets")
        ax.set_title(f"Target resolution count — GSD {float(freq)/2:.2f} mm/pix (threshold = {MTF_THRESHOLD})")
        ax.legend(fontsize=8)
        ax.yaxis.get_major_locator().set_params(integer=True)

        plt.tight_layout()
        out_path = output_dir / f"target_resolution_count_gsd_{float(freq)/2:.2f}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot USAF MTF score analysis (supports multiple CSVs)")
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Path(s) to usaf_scores.csv file(s)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory for plots (default: same as first CSV)",
        default=None,
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv_files]
    output_dir = Path(args.output) if args.output else csv_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df["h_mtf"] = pd.to_numeric(df["h_mtf"], errors="coerce")
        df["v_mtf"] = pd.to_numeric(df["v_mtf"], errors="coerce")
        df["frequency"] = df["frequency"].astype(str)
        label = _get_label(csv_path)
        datasets.append((df, label))
        print(f"Loaded {len(df)} rows from {csv_path} (label: {label})")

    print(f"\nComparing {len(datasets)} dataset(s)")
    print()

    plot_resolution_histogram(datasets, output_dir)
    plot_mtf_distributions(datasets, output_dir)
    plot_radius_histogram(datasets, output_dir)
    plot_poi_histogram(datasets, output_dir)
    plot_target_resolution_count(datasets, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
