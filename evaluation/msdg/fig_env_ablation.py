"""Save the original environment-ablation figure, pooling all alpha columns."""

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")  # Save figures without displaying them.

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


ENV_COUNTS = [2, 3, 4, 6, 8, 12]
VERSIONS = range(1, 11)
RESULT_DIR = Path("result")
OUTPUT_DIR = Path("figure/msdg")


def load_wscg():
    """Pool test WSCG over versions, environments, and every alpha column."""
    pooled = []

    for env_count in ENV_COUNTS:
        dataset = f"bio_ablation_env_{env_count}"
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
    """Apply the original dimensions, boxplot settings, axes, and legend."""
    fig, ax = plt.subplots(1, 1, figsize=(4.73, 3.68))
    fig.subplots_adjust(
        left=0.18,
        bottom=0.18,
        right=0.93,
        top=0.96,
        hspace=0.083,
        wspace=0.31,
    )

    ax.boxplot(
        pooled,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "lightblue", "color": "black"},
    )
    ax.set_xticklabels(ENV_COUNTS)
    ax.set_ylabel("WSCG over confidences", fontsize=13)
    ax.set_xlabel(r"$K$", fontsize=13)
    ax.set_ylim([0.0, 0.7])

    fig.legend(
        handles=[mpatches.Patch(color="lightblue", label="Augmented BNF+CQR")],
        loc="upper right",
        ncol=1,
        prop={"size": 13},
        bbox_to_anchor=(0.77, 0.97),
    )
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled = load_wscg()
    fig = create_figure(pooled)

    try:
        fig.savefig(
            OUTPUT_DIR / "env_ablation.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
