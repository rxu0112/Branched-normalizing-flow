from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Save figures without displaying them

output_dir = Path("figure/msdg")
output_dir.mkdir(parents=True, exist_ok=True)

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import pearsonr


dataset_list = ['bike','bio','traffic','states','hospital']
dataset_titles = ['Bike Rental', 'PTS', 'Traffic', 'U.S. ILI', 'Fair Med']
versions = [1,2,3,4,5,6,7,8,9,10]
alpha_list = [0.1]

COVERAGE_TEST = []
WSCG_TEST = []

for idx, dataset in enumerate(dataset_list):
    COVERAGE_TEST.append({})
    COVERAGE_TEST[idx]['dataset'] = dataset
    WSCG_TEST.append({})
    WSCG_TEST[idx]['dataset'] = dataset

    nfcqr_aug_wscg = []
    nfcqr_aug_size = []
    nfcqr_aug_marginal_coverage = []

    nfcqr_planar_wscg = []
    nfcqr_planar_size = []
    nfcqr_planar_marginal_coverage = []

    nfcqr_residual_wscg = []
    nfcqr_residual_size = []
    nfcqr_residual_marginal_coverage = []


    for v in versions:
        NFCQR_AUG_RESULT = pickle.load(open("result/" + dataset + "/msdg/bnfcqr_aug_v" + str(v) + "_msdg", "rb"))
        NFCQR_PLANAR_RESULT = pickle.load(open("result/" + dataset + "/msdg/bnfcqr_planar_v" + str(v) + "_msdg", "rb"))
        NFCQR_RES_RESULT = pickle.load(open("result/" + dataset + "/msdg/bnfcqr_residual_v" + str(v) + "_msdg", "rb"))

        for i in NFCQR_AUG_RESULT:
            if 'data' in i and i['data'] == 'test':
                nfcqr_aug_wscg.append(i['wscg'])
                nfcqr_aug_marginal_coverage.append(i['marginal coverage'])

        for i in NFCQR_PLANAR_RESULT:
            if 'data' in i and i['data'] == 'test':
                nfcqr_planar_wscg.append(i['wscg'])
                nfcqr_planar_marginal_coverage.append(i['marginal coverage'])

        for i in NFCQR_RES_RESULT:
            if 'data' in i and i['data'] == 'test':
                nfcqr_residual_wscg.append(i['wscg'])
                nfcqr_residual_marginal_coverage.append(i['marginal coverage'])

    nfcqr_aug_wscg = np.stack(nfcqr_aug_wscg)
    nfcqr_planar_wscg = np.stack(nfcqr_planar_wscg)
    nfcqr_residual_wscg = np.stack(nfcqr_residual_wscg)

    nfcqr_aug_marginal_coverage = np.stack(nfcqr_aug_marginal_coverage)
    nfcqr_planar_marginal_coverage = np.stack(nfcqr_planar_marginal_coverage)
    nfcqr_residual_marginal_coverage = np.stack(nfcqr_residual_marginal_coverage)

    COVERAGE_TEST[idx]['nfcqr_aug'] = nfcqr_aug_marginal_coverage
    COVERAGE_TEST[idx]['nfcqr_planar'] = nfcqr_planar_marginal_coverage
    COVERAGE_TEST[idx]['nfcqr_residual'] = nfcqr_residual_marginal_coverage

    WSCG_TEST[idx]['nfcqr_aug'] = nfcqr_aug_wscg
    WSCG_TEST[idx]['nfcqr_planar'] = nfcqr_planar_wscg
    WSCG_TEST[idx]['nfcqr_residual'] = nfcqr_residual_wscg


for a, alpha in enumerate(alpha_list):
    box_coverage = []
    box_gap = []

    for idx, dataset in enumerate(dataset_list):
        box_coverage.append([COVERAGE_TEST[idx]['nfcqr_planar'][:, a], COVERAGE_TEST[idx]['nfcqr_residual'][:, a], COVERAGE_TEST[idx]['nfcqr_aug'][:, a]])
        box_gap.append([WSCG_TEST[idx]['nfcqr_planar'][:, a], WSCG_TEST[idx]['nfcqr_residual'][:, a], WSCG_TEST[idx]['nfcqr_aug'][:, a]])

    colors = ['peachpuff', 'lightgreen', 'lightblue']
    labels = [r'BNF by planar flow+ CQR', r'BNF by residual flow + CQR', r'Augmented BNF + CQR']
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    titles = dataset_titles

    fig, axs = plt.subplots(1, 5, figsize=(13.8, 3.65))
    fig.subplots_adjust(left=0.064, bottom=0.164, right=0.995, top=0.936, hspace=0.083, wspace=0.31)

    # for i in range(len(dataset_list)):
    #     box = axs[0, i].boxplot(box_coverage[i], showfliers=False, patch_artist=True)
    #     for patch, color in zip(box['boxes'], colors):
    #         patch.set_facecolor(color)
    #     axs[0, i].axhline(y=1 - alpha, color='r', linestyle='--')
    #     axs[0, i].title.set_text(titles[i])
    #     if i == 0:
    #         axs[0, i].set_ylabel('marginal\ncoverage', fontsize=11, labelpad=10)
    #     axs[0, i].set_xticks([])
    #     axs[0, i].set_xticklabels([])
    #     axs[0, i].set_ylim([0.2, 1.01])
    for i in range(len(dataset_list)):
        box = axs[i].boxplot(box_gap[i], showfliers=False, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        if i == 0:
            axs[i].set_ylabel('WSCG', fontsize=11, labelpad=10)
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i].set_xticks([])
        axs[i].set_xticklabels([])
        axs[i].title.set_text(titles[i])


    line_legend = mlines.Line2D([], [], color='red', linestyle='--', label=r'$1-\alpha$')
    fig.legend(handles=legend_patches + [line_legend], loc='lower center', ncol=7, prop={'size': 12})

    fig.savefig(
                output_dir / f"flow_ablation_alpha_{alpha:g}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        
    plt.close(fig)
