"""Save the original marginal-coverage figure with its training-gap bounds."""

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
OUTPUT_FILENAME = "marginal_lower_bound.pdf"

# Filename prefix, legend label, and color, in the original plotting order.
METHODS = [
    ("scp", "SCP", "peachpuff"),
    ("iwcp", "IW-CP", "lightgreen"),
    ("wccp", "WC-CP", "lightcoral"),
    ("wrcp", "WR-CP", "sandybrown"),
    ("cqr", "CQR", "lightpink"),
    ("bnfcqr_aug", "Augmented BNF+CQR", "lightblue"),
]


def load_results():
    """Collect test coverage and per-version training-gap maxima by alpha."""
    coverage = {}
    bounds = {}

    for dataset, _ in DATASETS:
        test_coverage = {method: [] for method, _, _ in METHODS}
        version_bounds = []

        for version in VERSIONS:
            for method, _, _ in METHODS:
                path = (
                    RESULT_DIR / dataset / "msdg"
                    / f"{method}_v{version}_msdg"
                )
                with path.open("rb") as stream:
                    results = pickle.load(stream)

                training_gaps = []
                for result in results:
                    split = result.get("data")
                    if split == "test":
                        if method == "wrcp" and result["beta"] != 1:
                            continue
                        test_coverage[method].append(result["marginal coverage"])
                    elif method == "bnfcqr_aug" and split == "train":
                        training_gaps.append(result["marginal abs gap"])

                if method == "bnfcqr_aug":
                    # Preserve the original column-wise max for each version.
                    version_bounds.append([
                        max(column) for column in zip(*training_gaps)
                    ])

        coverage[dataset] = {
            method: np.stack(values)
            for method, values in test_coverage.items()
        }
        bounds[dataset] = np.stack(version_bounds)

    return coverage, bounds


def create_figure(coverage, bounds, alpha_index, alpha):
    """Apply the original boxplots, reference lines, layout, and legend."""
    fig, axs = plt.subplots(1, 5, figsize=(13.8, 2.65))
    fig.subplots_adjust(
        left=0.07,
        bottom=0.2,
        right=0.995,
        top=0.876,
        hspace=0.11,
        wspace=0.31,
    )

    for col, (dataset, title) in enumerate(DATASETS):
        ax = axs[col]
        values = [
            coverage[dataset][method][:, alpha_index]
            for method, _, _ in METHODS
        ]
        box = ax.boxplot(values, showfliers=False, patch_artist=True)
        for patch, (_, _, color) in zip(box["boxes"], METHODS):
            patch.set_facecolor(color)

        ax.axhline(y=1 - alpha, color="r", linestyle="--")
        lower_bound = 1 - alpha - np.max(bounds[dataset][:, alpha_index])
        ax.axhline(y=lower_bound, color="b", linestyle="--")

        ax.title.set_text(title)
        ax.title.set_fontsize(13)
        if col == 0:
            ax.set_ylabel("marginal\ncoverage", fontsize=14, labelpad=10)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylim([0.7, 1.01])

    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for _, label, color in METHODS
    ]
    line_legend = mlines.Line2D(
        [], [], color="red", linestyle="--", label=r"$1-\alpha$"
    )
    bound_legend = mlines.Line2D(
        [], [], color="b", linestyle="--", label="lower bound"
    )
    fig.legend(
        handles=legend_patches + [line_legend, bound_legend],
        fontsize=12,
        loc="lower center",
        ncol=8,
    )
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage, bounds = load_results()

    for alpha_index, alpha in enumerate(ALPHAS):
        fig = create_figure(coverage, bounds, alpha_index, alpha)
        try:
            # Keep the original fixed filename for every alpha iteration.
            fig.savefig(
                OUTPUT_DIR / OUTPUT_FILENAME,
                dpi=300,
                bbox_inches="tight",
            )
        finally:
            plt.close(fig)


if __name__ == "__main__":
    main()
