"""Load MSDG results and save the original three-row coverage figure."""

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")  # Save figures without displaying them.

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# Dataset order also determines the figure's column order.
DATASETS = [
    ("bike", "Bike Rental"),
    ("bio", "PTS"),
    ("traffic", "Traffic"),
    ("states", "U.S. ILI"),
    ("hospital", "Fair Med"),
]
VERSIONS = range(1, 11)
ALPHAS = [0.1]

RESULT_DIR = Path("result")
OUTPUT_DIR = Path("figure/msdg")

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
    ("marginal coverage", "marginal"),
    ("wsc", "WSC"),
    ("wscg", "WSCG"),
]


def load_results():
    """Pool test environments in version order, retaining all alpha columns."""
    data = {}

    for dataset, _ in DATASETS:
        pooled = {
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
                        pooled[method][metric].append(result[metric])

        data[dataset] = {
            method: {
                metric: np.stack(values)
                for metric, values in method_values.items()
            }
            for method, method_values in pooled.items()
        }

    return data


def create_figure(data, alpha_index, alpha):
    """Apply the original dimensions, layout, styling, limits, and legend."""
    fig, axs = plt.subplots(3, 5, figsize=(13.8, 4))
    fig.subplots_adjust(
        left=0.05,
        bottom=0.11,
        right=0.995,
        top=0.936,
        hspace=0.11,
        wspace=0.31,
    )

    for row, (metric, ylabel) in enumerate(METRICS):
        for col, (dataset, title) in enumerate(DATASETS):
            ax = axs[row, col]
            values = [
                data[dataset][method][metric][:, alpha_index]
                for method, _, _ in METHODS
            ]
            box = ax.boxplot(values, showfliers=False, patch_artist=True)
            for patch, (_, _, color) in zip(box["boxes"], METHODS):
                patch.set_facecolor(color)

            if row < 2:
                ax.axhline(y=1 - alpha, color="r", linestyle="--")
            if row == 0:
                ax.title.set_text(title)
                ax.title.set_fontsize(13)
                ax.set_ylim([0.7, 1.01])
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=14, labelpad=10)

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
        fontsize=12,
        loc="lower center",
        ncol=7,
    )
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()

    for alpha_index, alpha in enumerate(ALPHAS):
        fig = create_figure(data, alpha_index, alpha)
        try:
            fig.savefig(
                OUTPUT_DIR / f"coverage_alpha_{alpha:g}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        finally:
            plt.close(fig)


if __name__ == "__main__":
    main()
