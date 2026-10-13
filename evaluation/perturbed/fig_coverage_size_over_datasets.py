"""Save the original four-panel figure pooled over perturbed datasets."""

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")  # Save figures without displaying them.

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["bike", "bio", "traffic", "states", "hospital"]
VERSIONS = range(1, 11)
ALPHA_INDEX = 0  # The original figure uses the first saved alpha column.

RESULT_DIR = Path("result")
OUTPUT_DIR = Path("figure/perturbed")
OUTPUT_FILENAME = "coverage_size_over_datasets_alpha_0.1.pdf"

# Filename prefix, legend label, and color, in the original plotting order.
METHODS = [
    ("scp", "SCP", "peachpuff"),
    ("iwcp", "IW-CP", "lightgreen"),
    ("wccp", "WC-CP", "lightcoral"),
    ("wrcp", "WR-CP", "sandybrown"),
    ("cqr", "CQR", "lightpink"),
    ("bnfcqr_aug", "Augmented BNF+CQR", "lightblue"),
]

# Saved metric key and y-axis label, in the original panel order.
METRICS = [
    ("marginal coverage", "marginal"),
    ("wsc", "WSC"),
    ("wscg", "WSCG"),
    ("size", "set size"),
]


def load_results():
    """Collect test environments in version order, retaining WRCP beta=1."""
    data = {}

    for dataset in DATASETS:
        collected = {
            method: {metric: [] for metric, _ in METRICS}
            for method, _, _ in METHODS
        }

        for version in VERSIONS:
            for method, _, _ in METHODS:
                path = (
                    RESULT_DIR / dataset / "perturbed"
                    / f"{method}_v{version}_perturbed"
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


def pool_results(data, alpha_index):
    """Preserve the original dataset order and float64 pooling behavior."""
    pooled = {}

    for method, _, _ in METHODS:
        # Match the original fixed-width np.fromiter operation exactly.
        width = len(data[DATASETS[0]][method]["marginal coverage"][:, alpha_index])
        pooled[method] = {
            metric: np.fromiter(
                (data[dataset][method][metric][:, alpha_index] for dataset in DATASETS),
                dtype=np.dtype((float, width)),
            ).flatten()
            for metric, _ in METRICS
        }

    return pooled


def create_figure(pooled):
    """Apply the original layout, box plots, mean-size bars, and legend."""
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

    for ax, (metric, ylabel) in zip(axs, METRICS):
        values = [pooled[method][metric] for method, _, _ in METHODS]

        if metric == "size":
            means = [np.mean(observations) for observations in values]
            ax.bar(range(1, len(values) + 1), means, color=colors)
        else:
            box = ax.boxplot(values, showfliers=False, patch_artist=True)
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
            ax.title.set_fontsize(13)

        ax.set_ylabel(ylabel, fontsize=14, labelpad=0)
        if metric in ("marginal coverage", "wsc"):
            ax.axhline(y=0.9, color="r", linestyle="--")
        ax.set_xticks([])
        ax.set_xticklabels([])

    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for _, label, color in METHODS
    ]
    line_legend = mlines.Line2D(
        [], [], color="red", linestyle="--", label=r"$1-\alpha$"
    )
    fig.legend(
        handles=legend_patches + [line_legend],
        loc="lower center",
        ncol=7,
        prop={"size": 12},
    )
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()
    pooled = pool_results(data, ALPHA_INDEX)
    fig = create_figure(pooled)
    try:
        fig.savefig(OUTPUT_DIR / OUTPUT_FILENAME, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
