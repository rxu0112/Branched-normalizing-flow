"""Save the original mean-WSCG figure across confidence levels."""

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")  # Save figures without displaying them.

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


DATASETS = ["bike", "bio", "traffic", "states", "hospital"]
VERSIONS = range(1, 11)
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Keep explicit confidence labels to preserve the original tick text.
CONFIDENCE_LABELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

RESULT_DIR = Path("result")
OUTPUT_DIR = Path("figure/msdg")

# Filename prefix, legend label, and color, in the original plotting order.
METHODS = [
    ("scp", "SCP", "peachpuff"),
    ("iwcp", "IW-CP", "lightgreen"),
    ("wccp", "WC-CP", "lightcoral"),
    ("wrcp", "WR-CP", "sandybrown"),
    ("cqr", "CQR", "lightpink"),
    ("bnfcqr_aug", "Augmented BNF \n +CQR", "lightblue"),
]


def load_wscg():
    """Collect test-environment WSCG in version order, retaining every alpha."""
    data = {}

    for dataset in DATASETS:
        collected = {method: [] for method, _, _ in METHODS}

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
                    collected[method].append(result["wscg"])

        data[dataset] = {
            method: np.stack(values)
            for method, values in collected.items()
        }

    return data


def pool_by_alpha(data):
    """Pool datasets separately for each method and alpha.

    Keep the original fixed-length float64 conversion before calculating means.
    As in the original script, SCP in the first dataset determines that length.
    """
    length = len(data[DATASETS[0]]["scp"])
    pooled_dtype = np.dtype((float, length))

    return {
        method: [
            np.fromiter(
                (data[dataset][method][:, alpha_index] for dataset in DATASETS),
                dtype=pooled_dtype,
            ).flatten()
            for alpha_index in range(len(ALPHAS))
        ]
        for method, _, _ in METHODS
    }


def create_figure(pooled):
    """Apply the original line styles, reversed alpha order, axes, and legend."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 3))
    fig.subplots_adjust(
        left=0.126,
        bottom=0.1,
        right=0.73,
        top=0.974,
        hspace=0.2,
        wspace=0.2,
    )

    for method, label, color in METHODS:
        means = [np.mean(values) for values in pooled[method]]
        ax.plot(
            list(reversed(means)),
            label=label,
            color=color,
            linestyle="--",
            marker="o",
        )

    ax.set_xticks(range(len(CONFIDENCE_LABELS)))
    ax.set_xticklabels(CONFIDENCE_LABELS)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("1-α", fontsize=14)
    ax.set_ylabel(r"${\mathrm{WSCG}}$", fontsize=14)

    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for _, label, color in METHODS
    ]
    fig.legend(
        handles=legend_patches,
        loc="right",
        ncol=1,
        prop={"size": 13},
    )
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_wscg()
    pooled = pool_by_alpha(data)
    fig = create_figure(pooled)

    try:
        fig.savefig(
            OUTPUT_DIR / "alpha_ablation.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
