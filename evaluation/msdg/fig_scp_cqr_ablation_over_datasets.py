"""Plot the SCP/CQR ablation with results pooled across datasets."""

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")  # Save figures without displaying them.

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


DATASETS = ["bike", "bio", "traffic", "states", "hospital"]
VERSIONS = range(1, 11)

# Entries correspond to the alpha columns in the saved results.
ALPHAS = [0.1]

RESULT_DIR = Path("result")
OUTPUT_DIR = Path("figure/msdg")

# Filename prefix, legend label, and color.
METHODS = [
    ("scp", "SCP", "peachpuff"),
    ("cqr", "CQR", "lightpink"),
    ("bnfscp_aug", "Augmented BNF+SCP", "lightgreen"),
    ("bnfcqr_aug", "Augmented BNF+CQR", "lightblue"),
]

# Saved metric key and axis label.
METRICS = [
    ("marginal coverage", "marginal"),
    ("wsc", "WSC"),
    ("wscg", "WSCG"),
]


def load_results():
    """Pool individual test evaluations across datasets and versions."""
    pooled = {
        method: {metric: [] for metric, _ in METRICS}
        for method, _, _ in METHODS
    }

    for dataset in DATASETS:
        for version in VERSIONS:
            for method, _, _ in METHODS:
                path = (
                    RESULT_DIR
                    / dataset
                    / "msdg"
                    / f"{method}_v{version}_msdg"
                )

                with path.open("rb") as stream:
                    results = pickle.load(stream)

                for result in results:
                    if result.get("data") != "test":
                        continue

                    for metric, _ in METRICS:
                        values = np.atleast_1d(
                            np.asarray(result[metric], dtype=float)
                        )
                        pooled[method][metric].append(values)

    data = {}
    for method, _, _ in METHODS:
        data[method] = {}
        for metric, _ in METRICS:
            values = pooled[method][metric]
            if not values:
                raise ValueError(
                    f"No test results found for {method}: {metric}"
                )

            # Shape: (pooled test evaluations, alpha columns).
            data[method][metric] = np.stack(values)

    return data


def create_figure(data, alpha_index, alpha):
    """Create three panels using the original compact layout."""
    fig, axs = plt.subplots(1, 3, figsize=(6.89, 2.38))
    fig.subplots_adjust(
        left=0.074,
        bottom=0.436,
        right=0.98,
        top=0.98,
        hspace=0.3,
        wspace=0.37,
    )

    for index, (metric, ylabel) in enumerate(METRICS):
        ax = axs[index]

        values = [
            data[method][metric][:, alpha_index]
            for method, _, _ in METHODS
        ]

        box = ax.boxplot(
            values,
            showfliers=False,
            patch_artist=True,
        )

        for patch, (_, _, color) in zip(box["boxes"], METHODS):
            patch.set_facecolor(color)

        ax.set_ylabel(ylabel, fontsize=14, labelpad=0)

        if index < 2:
            ax.axhline(
                y=1 - alpha,
                color="red",
                linestyle="--",
            )

        if index == 0:
            ax.set_ylim(0.7, 1.01)

        ax.set_xticks([])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for _, label, color in METHODS
    ]
    line_legend = mlines.Line2D(
        [],
        [],
        color="red",
        linestyle="--",
        label=r"$1-\alpha$",
    )

    fig.legend(
        handles=legend_patches + [line_legend],
        fontsize=12,
        loc="lower center",
        ncol=2,
    )

    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()

    for alpha_index, alpha in enumerate(ALPHAS):
        fig = create_figure(data, alpha_index, alpha)
        output_path = (
            OUTPUT_DIR
            / f"scp_ablation_alpha_{alpha:g}_over_datasets.pdf"
        )

        try:
            fig.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
            )
        finally:
            plt.close(fig)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()