"""Save the original sample-ablation figure, pooling all alpha columns."""

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")  # Save figures without displaying them.

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# Dataset folder and displayed sample count, in the original plotting order.
# The final point uses the standard bio dataset, displayed as 7500 samples.
DATASETS = [
    ("bio_ablation_sample_100", 100),
    ("bio_ablation_sample_200", 200),
    ("bio_ablation_sample_300", 300),
    ("bio_ablation_sample_500", 500),
    ("bio_ablation_sample_750", 750),
    ("bio_ablation_sample_1000", 1000),
    ("bio_ablation_sample_1500", 1500),
    ("bio_ablation_sample_3000", 3000),
    ("bio_ablation_sample_5000", 5000),
    ("bio", 7500),
]
VERSIONS = range(1, 11)
RESULT_DIR = Path("result")
OUTPUT_DIR = Path("figure/msdg")


def load_wscg():
    """Pool test WSCG over versions, environments, and every alpha column."""
    pooled = []

    for dataset, _ in DATASETS:
        values = []

        for version in VERSIONS:
            path = (
                RESULT_DIR / dataset / "msdg"
                / f"bnfcqr_aug_v{version}_msdg"
            )
            with path.open("rb") as stream:
                results = pickle.load(stream)

            for result in results:
                if result.get("data") == "test":
                    values.append(result["wscg"])

        # Preserve the original stacking order, dtype, and all-alpha pooling.
        pooled.append(np.stack(values).flatten())

    return pooled


def create_figure(pooled):
    """Apply the original dimensions, boxplot settings, labels, and legend."""
    fig, ax = plt.subplots(1, 1, figsize=(4.73, 3.68))
    fig.subplots_adjust(
        left=0.18,
        bottom=0.18,
        right=0.93,
        top=0.93,
        hspace=0.083,
        wspace=0.31,
    )

    ax.boxplot(
        pooled,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "lightblue", "color": "black"},
    )
    ax.set_xticklabels([sample_count for _, sample_count in DATASETS], rotation=30)
    ax.set_ylabel("WSCG over confidences", fontsize=13)
    ax.set_xlabel(r"Set size of $\mathcal{S}_{D^k}$", fontsize=13)

    fig.legend(
        handles=[mpatches.Patch(color="lightblue", label="Augmented BNF+CQR")],
        loc="upper right",
        ncol=1,
        prop={"size": 13},
        bbox_to_anchor=(0.93, 0.93),
    )
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled = load_wscg()
    fig = create_figure(pooled)

    try:
        fig.savefig(
            OUTPUT_DIR / "sample_ablation.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
