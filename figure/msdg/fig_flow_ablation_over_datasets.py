"""Plot flow ablations with test-set results pooled across datasets and trials.

Run from the repository root. Edit DATASETS, VERSIONS, and ALPHAS below
to select the results to plot.
"""

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


DATASETS = ["bike", "bio", "traffic", "states", "hospital"]
VERSIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ALPHAS = [0.1]

RESULT_DIR = Path("result")
OUTPUT_DIR = Path("figure/msdg")

# Result filename prefix, legend label, and color.
METHODS = [
    ("bnfcqr_planar", "BNF by planar flow + CQR", "peachpuff"),
    ("bnfcqr_residual", "BNF by residual flow + CQR", "lightgreen"),
    ("bnfcqr_aug", "Augmented BNF + CQR", "lightblue"),
]

METRICS = [
    ("marginal coverage", "marginal"),
    ("wsc", "WSC"),
    ("wscg", "WSCG"),
]


def load_results():
    """Pool test records without averaging over datasets or trials."""
    pooled = {
        method: {metric: [] for metric, _ in METRICS}
        for method, _, _ in METHODS
    }

    for dataset in DATASETS:
        for version in VERSIONS:
            for method, _, _ in METHODS:
                path = (
                    RESULT_DIR / dataset / "msdg"
                    / f"{method}_v{version}_msdg"
                )
                with path.open("rb") as stream:
                    results = pickle.load(stream)

                for row in results:
                    if row.get("data") != "test":
                        continue

                    # Match the saved alpha values, rather than assuming column 0.
                    saved_alphas = np.asarray(row["alpha"], dtype=float)
                    indices = []
                    for alpha in ALPHAS:
                        matches = np.flatnonzero(np.isclose(saved_alphas, alpha))
                        if matches.size != 1:
                            raise ValueError(
                                f"Expected one alpha={alpha:g} entry in {path}"
                            )
                        indices.append(int(matches[0]))

                    for metric, _ in METRICS:
                        values = np.asarray(row[metric], dtype=float)
                        pooled[method][metric].append(values[indices])

    return {
        method: {
            metric: np.stack(values)
            for metric, values in metrics.items()
        }
        for method, metrics in pooled.items()
    }


def create_figure(data, alpha_index, alpha):
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.0))
    fig.subplots_adjust(
        left=0.085, right=0.99, bottom=0.32, top=0.97, wspace=0.48
    )

    for ax, (metric, ylabel) in zip(axes, METRICS):
        values = [
            data[method][metric][:, alpha_index]
            for method, _, _ in METHODS
        ]
        boxes = ax.boxplot(
            values, widths=0.5, showfliers=False, patch_artist=True
        )
        for patch, (_, _, color) in zip(boxes["boxes"], METHODS):
            patch.set_facecolor(color)

        ax.set_ylabel(ylabel, fontsize=14, labelpad=2)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=11)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        if metric in ("marginal coverage", "wsc"):
            ax.axhline(1 - alpha, color="red", linestyle="--", linewidth=1.5)
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(bottom=0)

    handles = [
        mpatches.Patch(facecolor=color, label=label)
        for _, label, color in METHODS
    ]
    handles.append(
        mlines.Line2D(
            [], [], color="red", linestyle="--", label=r"$1-\alpha$"
        )
    )
    fig.legend(
        handles=handles, loc="lower center", ncol=2, fontsize=11,
        bbox_to_anchor=(0.5, 0.015)
    )
    return fig


def main():
    data = load_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for alpha_index, alpha in enumerate(ALPHAS):
        fig = create_figure(data, alpha_index, alpha)
        stem = f"flow_ablation_alpha_{alpha:g}_over_datasets"
        output_path = OUTPUT_DIR / f"{stem}.pdf"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)



if __name__ == "__main__":
    main()
