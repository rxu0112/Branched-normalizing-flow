"""Save the original coverage, WSC, WSCG, and size figure for perturbed data."""

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")  # Save figures without displaying them.

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


# Dataset order also determines the figure's column order.
DATASETS = [
    ("bike", "Bike Sharing"),
    ("bio", "PTS"),
    ("traffic", "Traffic"),
    ("states", "U.S. ILI"),
    ("hospital", "Fair Med"),
]
VERSIONS = range(1, 11)
ALPHAS = [0.1]

RESULT_DIR = Path("result")
OUTPUT_DIR = Path("figure/perturbed")

# Filename prefix, legend label, and color, in the original plotting order.
METHODS = [
    ("scp", "SCP", "peachpuff"),
    ("iwcp", "IW-CP", "lightgreen"),
    ("wccp", "WC-CP", "lightcoral"),
    ("wrcp", "WR-CP", "sandybrown"),
    ("cqr", "CQR", "lightpink"),
    ("bnfcqr_aug", "Augmented BNF+CQR", "lightblue"),
]

# Saved metric key and y-axis label, in the original row order.
METRICS = [
    ("marginal coverage", "marginal\ncoverage"),
    ("wsc", "WSC"),
    ("wscg", "WSCG"),
    ("size", "set size"),
]


def load_results():
    """Pool test environments in version order, preserving metric dtypes."""
    data = {}

    for dataset, _ in DATASETS:
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


def create_figure(data, alpha_index, alpha):
    """Preserve the original layout, styling, axis limits, and mean-size bars."""
    colors = [color for _, _, color in METHODS]
    fig, axs = plt.subplots(4, 5, figsize=(13.8, 5.0))
    fig.subplots_adjust(
        left=0.067,
        bottom=0.152,
        right=0.995,
        top=0.936,
        hspace=0.083,
        wspace=0.31,
    )

    for row, (metric, ylabel) in enumerate(METRICS):
        for col, (dataset, title) in enumerate(DATASETS):
            ax = axs[row, col]
            values = [
                data[dataset][method][metric][:, alpha_index]
                for method, _, _ in METHODS
            ]

            if metric == "size":
                means = [np.mean(observations) for observations in values]
                ax.bar(range(1, len(values) + 1), means, color=colors)
            else:
                box = ax.boxplot(values, showfliers=False, patch_artist=True)
                for patch, color in zip(box["boxes"], colors):
                    patch.set_facecolor(color)

            if metric in ("marginal coverage", "wsc"):
                ax.axhline(y=1 - alpha, color="r", linestyle="--")
            if row == 0:
                ax.title.set_text(title)
            if metric in ("wsc", "wscg"):
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=14, labelpad=10)

            ax.set_xticks([])
            ax.set_xticklabels([])

            # Preserve the original WSC limits for the fourth column (U.S. ILI).
            if metric == "wsc" and col == 3:
                ax.set_ylim([0.0, 1.01])

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
        ncol=4,
        prop={"size": 12},
    )
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()

    for alpha_index, alpha in enumerate(ALPHAS):
        fig = create_figure(data, alpha_index, alpha)
        try:
            fig.savefig(
                OUTPUT_DIR / f"coverage_size_alpha_{alpha}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        finally:
            plt.close(fig)


if __name__ == "__main__":
    main()
