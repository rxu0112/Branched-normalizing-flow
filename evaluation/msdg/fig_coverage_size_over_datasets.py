"""Save the original four-panel figure with results pooled across datasets."""

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")  # Save figures without displaying them.

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["bike","bio", "traffic", "states", "hospital"]
VERSIONS = range(1, 11)

# The original figure uses the first alpha column and reference lines at 0.9.
ALPHA_INDEX = 0
REFERENCE_COVERAGE = 0.9

RESULT_DIR = Path("result")
OUTPUT_DIR = Path("figure/msdg")
OUTPUT_FILENAME = "coverage_size_over_datasets_alpha_0.1.pdf"

# Filename prefix, legend label, and color, in the original plotting order.
METHODS = [
    ("scp", "SCP", "peachpuff"),
    ("iwcp", "IW-CP", "lightgreen"),
    ("wccp", "WC-CP", "lightcoral"),
    ("wrcp", "WR-CP", "sandybrown"),
    ("cqr", "CQR", "lightpink"),
    ("bnfcqr_aug", "Augmented BNF+CQR", "lightblue"),
    ("bnfcqr_aug_cond", "Augment-conditioned BNF+CQR", "violet"),
]

# Saved metric key and y-axis label, in the original panel order.
METRICS = [
    ("marginal coverage", "marginal"),
    ("wsc", "WSC"),
    ("wscg", "WSCG"),
    ("size", "set size"),
]


def load_results():
    """Collect test environments in version order, retaining all alpha columns."""
    data = {}

    for dataset in DATASETS:
        collected = {
            method: {metric: [] for metric, _ in METRICS}
            for method, _, _ in METHODS
        }

        for version in VERSIONS:
            for method, _, _ in METHODS:
                path = (
                    RESULT_DIR / dataset / "msdg"
                    / f"{method}_v{version}_msdg"
                )
                with path.open("rb") as stream:
                    results = pickle.load(stream)

                for result in results:
                    if result.get("data") != "test":
                        continue
                    if method == "wrcp" and result["beta"] != 1:
                        continue

                    for metric, _ in METRICS:
                        collected[method][metric].append(result[metric])

        data[dataset] = {
            method: {
                metric: np.stack(values)
                for metric, values in method_values.items()
            }
            for method, method_values in collected.items()
        }

    return data


def pool_datasets(data):
    """Pool observations in dataset order without averaging environments.

    Convert to float64 to match the original np.fromiter(..., dtype=float),
    including when the saved metric arrays contain float32 values.
    """
    return {
        metric: [
            np.concatenate([
                np.asarray(
                    data[dataset][method][metric][:, ALPHA_INDEX],
                    dtype=float,
                )
                for dataset in DATASETS
            ])
            for method, _, _ in METHODS
        ]
        for metric, _ in METRICS
    }


def create_figure(pooled):
    """Apply the original layout, boxplots, mean-size bars, and legend."""
    colors = [color for _, _, color in METHODS]
    fig, axs = plt.subplots(1, 4, figsize=(13.65, 1.84))
    fig.subplots_adjust(
        left=0.04,
        bottom=0.283,
        right=0.998,
        top=0.98,
        hspace=0.3,
        wspace=0.37,
    )

    for panel, (metric, ylabel) in enumerate(METRICS):
        ax = axs[panel]
        values = pooled[metric]

        if metric == "size":
            means = [np.mean(observations) for observations in values]
            ax.bar(range(1, len(values) + 1), means, color=colors)
        else:
            box = ax.boxplot(values, showfliers=False, patch_artist=True)
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
            ax.title.set_fontsize(13)

        ax.set_ylabel(ylabel, fontsize=14, labelpad=0)
        if panel < 2:
            ax.axhline(y=REFERENCE_COVERAGE, color="r", linestyle="--")
        if panel == 0:
            ax.set_ylim(0.7, 1.01)
        ax.set_xticks([])
        ax.set_xticklabels([])

    line_legend = mlines.Line2D(
        [], [], color="red", linestyle="--", label=r"$1-\alpha$"
    )
    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for _, label, color in METHODS
    ]
    fig.legend(
        handles=legend_patches + [line_legend],
        fontsize=12,
        loc="lower center",
        ncol=8,
    )
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()
    pooled = pool_datasets(data)
    fig = create_figure(pooled)

    try:
        fig.savefig(
            OUTPUT_DIR / OUTPUT_FILENAME,
            dpi=300,
            bbox_inches="tight",
        )
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
