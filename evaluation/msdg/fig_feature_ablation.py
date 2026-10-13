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

dataset_list = ['bike','bio','traffic','states','hospital'] #['bike','bio','traffic','states','hospital']
dataset_titles = ['Bike Rental', 'PTS', 'Traffic', 'U.S. ILI', 'Fair Med']
versions = [1,2,3,4,5,6,7,8,9,10]
alpha_list = [0.1]

COVERAGE_TEST = []
WSCG_TEST = []
WSC_TEST = []

for idx, dataset in enumerate(dataset_list):
    COVERAGE_TEST.append({})
    COVERAGE_TEST[idx]['dataset'] = dataset
    WSCG_TEST.append({})
    WSCG_TEST[idx]['dataset'] = dataset
    WSC_TEST.append({})
    WSC_TEST[idx]['dataset'] = dataset

    nfcqr_aug_wscg = []
    nfcqr_aug_wsc = []
    nfcqr_aug_marginal_coverage = []
    nfcqr_fea_cond_wscg = []
    nfcqr_fea_cond_wsc = []
    nfcqr_fea_cond_marginal_coverage = []
    nfcqr_fea_pca_wscg = []
    nfcqr_fea_pca_wsc = []
    nfcqr_fea_pca_marginal_coverage = []
    nfcqr_fea_tsne_wscg = []
    nfcqr_fea_tsne_wsc = []
    nfcqr_fea_tsne_marginal_coverage = []
    nfcqr_fea_umap_wscg = []
    nfcqr_fea_umap_wsc = []
    nfcqr_fea_umap_marginal_coverage = []


    for v in versions:
        print("dataset: ", dataset, "version: ", v)
        NFCQR_AUG_RESULT = pickle.load(open("result/" + dataset + "/msdg/bnfcqr_aug_v" + str(v) + "_msdg", "rb"))
        NFCQR_FEA_COND_RESULT = pickle.load(open("result/" + dataset + "/msdg/bnfcqr_fea_v" + str(v) + "_msdg", "rb"))
        NFCQR_FEA_PCA_RESULT = pickle.load(open("result/" + dataset + "/msdg/bnfcqr_pca_v" + str(v) + "_msdg", "rb"))
        NFCQR_FEA_TSNE_RESULT = pickle.load(open("result/" + dataset + "/msdg/bnfcqr_tsne_v" + str(v) + "_msdg", "rb"))
        NFCQR_FEA_UMAP_RESULT = pickle.load(open("result/" + dataset + "/msdg/bnfcqr_umap_v" + str(v) + "_msdg", "rb"))


        for i in NFCQR_AUG_RESULT:
            if 'data' in i and i['data'] == 'test':
                nfcqr_aug_wscg.append(i['wscg'])
                nfcqr_aug_wsc.append(i['wsc'])
                nfcqr_aug_marginal_coverage.append(i['marginal coverage'])

        for i in NFCQR_FEA_COND_RESULT:
            if 'data' in i and i['data'] == 'test':
                nfcqr_fea_cond_wscg.append(i['wscg'])
                nfcqr_fea_cond_wsc.append(i['wsc'])
                nfcqr_fea_cond_marginal_coverage.append(i['marginal coverage'])

        for i in NFCQR_FEA_PCA_RESULT:
            if 'data' in i and i['data'] == 'test':
                nfcqr_fea_pca_wscg.append(i['wscg'])
                nfcqr_fea_pca_wsc.append(i['wsc'])
                nfcqr_fea_pca_marginal_coverage.append(i['marginal coverage'])

        for i in NFCQR_FEA_TSNE_RESULT:
            if 'data' in i and i['data'] == 'test':
                nfcqr_fea_tsne_wscg.append(i['wscg'])
                nfcqr_fea_tsne_wsc.append(i['wsc'])
                nfcqr_fea_tsne_marginal_coverage.append(i['marginal coverage'])

        for i in NFCQR_FEA_UMAP_RESULT:
            if 'data' in i and i['data'] == 'test':
                nfcqr_fea_umap_wscg.append(i['wscg'])
                nfcqr_fea_umap_wsc.append(i['wsc'])
                nfcqr_fea_umap_marginal_coverage.append(i['marginal coverage'])


    nfcqr_aug_wscg = np.stack(nfcqr_aug_wscg)
    nfcqr_fea_cond_wscg = np.stack(nfcqr_fea_cond_wscg)
    nfcqr_fea_pca_wscg = np.stack(nfcqr_fea_pca_wscg)
    nfcqr_fea_tsne_wscg = np.stack(nfcqr_fea_tsne_wscg)
    nfcqr_fea_umap_wscg = np.stack(nfcqr_fea_umap_wscg)

    nfcqr_aug_wsc = np.stack(nfcqr_aug_wsc)
    nfcqr_fea_cond_wsc = np.stack(nfcqr_fea_cond_wsc)
    nfcqr_fea_pca_wsc = np.stack(nfcqr_fea_pca_wsc)
    nfcqr_fea_tsne_wsc = np.stack(nfcqr_fea_tsne_wsc)
    nfcqr_fea_umap_wsc = np.stack(nfcqr_fea_umap_wsc)

    nfcqr_aug_marginal_coverage = np.stack(nfcqr_aug_marginal_coverage)
    nfcqr_fea_cond_marginal_coverage = np.stack(nfcqr_fea_cond_marginal_coverage)
    nfcqr_fea_pca_marginal_coverage = np.stack(nfcqr_fea_pca_marginal_coverage)
    nfcqr_fea_tsne_marginal_coverage = np.stack(nfcqr_fea_tsne_marginal_coverage)
    nfcqr_fea_umap_marginal_coverage = np.stack(nfcqr_fea_umap_marginal_coverage)

    COVERAGE_TEST[idx]['nfcqr_aug'] = nfcqr_aug_marginal_coverage
    COVERAGE_TEST[idx]['nfcqr_fea_cond'] = nfcqr_fea_cond_marginal_coverage
    COVERAGE_TEST[idx]['nfcqr_fea_pca'] = nfcqr_fea_pca_marginal_coverage
    COVERAGE_TEST[idx]['nfcqr_fea_tsne'] = nfcqr_fea_tsne_marginal_coverage
    COVERAGE_TEST[idx]['nfcqr_fea_umap'] = nfcqr_fea_umap_marginal_coverage

    WSCG_TEST[idx]['nfcqr_aug'] = nfcqr_aug_wscg
    WSCG_TEST[idx]['nfcqr_fea_cond'] = nfcqr_fea_cond_wscg
    WSCG_TEST[idx]['nfcqr_fea_pca'] = nfcqr_fea_pca_wscg
    WSCG_TEST[idx]['nfcqr_fea_tsne'] = nfcqr_fea_tsne_wscg
    WSCG_TEST[idx]['nfcqr_fea_umap'] = nfcqr_fea_umap_wscg

    WSC_TEST[idx]['nfcqr_aug'] = nfcqr_aug_wsc
    WSC_TEST[idx]['nfcqr_fea_cond'] = nfcqr_fea_cond_wsc
    WSC_TEST[idx]['nfcqr_fea_pca'] = nfcqr_fea_pca_wsc
    WSC_TEST[idx]['nfcqr_fea_tsne'] = nfcqr_fea_tsne_wsc
    WSC_TEST[idx]['nfcqr_fea_umap'] = nfcqr_fea_umap_wsc

for a, alpha in enumerate(alpha_list):
    box_coverage = []
    box_wscg = []
    box_wsc = []

    for idx, dataset in enumerate(dataset_list):
        box_coverage.append([COVERAGE_TEST[idx]['nfcqr_aug'][:, a], COVERAGE_TEST[idx]['nfcqr_fea_cond'][:, a],COVERAGE_TEST[idx]['nfcqr_fea_pca'][:, a],COVERAGE_TEST[idx]['nfcqr_fea_tsne'][:, a],COVERAGE_TEST[idx]['nfcqr_fea_umap'][:, a]])
        box_wscg.append([WSCG_TEST[idx]['nfcqr_aug'][:, a], WSCG_TEST[idx]['nfcqr_fea_cond'][:, a], WSCG_TEST[idx]['nfcqr_fea_pca'][:, a], WSCG_TEST[idx]['nfcqr_fea_tsne'][:, a],WSCG_TEST[idx]['nfcqr_fea_umap'][:, a]])
        box_wsc.append([WSC_TEST[idx]['nfcqr_aug'][:, a], WSC_TEST[idx]['nfcqr_fea_cond'][:, a], WSC_TEST[idx]['nfcqr_fea_pca'][:, a], WSC_TEST[idx]['nfcqr_fea_tsne'][:, a],WSCG_TEST[idx]['nfcqr_fea_umap'][:, a]])

    colors = ['lightblue', 'crimson','bisque','mediumturquoise','pink']
    labels = ['Augmented BNF+CQR', 'Feature-conditioned BNF+CQR','PCA-conditioned BNF+CQR','tSNE-conditioned BNF+CQR','UMAP-conditioned BNF+CQR']
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    titles = dataset_titles

    fig, axs = plt.subplots(3, 5, figsize=(13.8, 4.65))
    fig.subplots_adjust(left=0.062, bottom=0.157, right=0.995, top=0.936, hspace=0.11, wspace=0.31)

    for i in range(len(dataset_list)):
        box = axs[0, i].boxplot(box_coverage[i], showfliers=False, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        axs[0, i].axhline(y=1 - alpha, color='r', linestyle='--')
        axs[0, i].title.set_text(titles[i])
        if i == 0:
            axs[0, i].set_ylabel('marginal', fontsize=14, labelpad=10)
        axs[0, i].set_xticks([])
        axs[0, i].set_xticklabels([])
        axs[0, i].set_ylim([0.8, 1.01])
        # axs[0, i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    for i in range(len(dataset_list)):
        box = axs[1, i].boxplot(box_wsc[i], showfliers=False, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        axs[1, i].axhline(y=1 - alpha, color='r', linestyle='--')
        if i == 0:
            axs[1, i].set_ylabel('WSC', fontsize=14, labelpad=10)
        axs[1, i].set_xticks([])
        axs[1, i].set_xticklabels([])
        # axs[1, i].set_ylim([0.0, 1.01])
        # axs[1, i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    for i in range(len(dataset_list)):
        box = axs[2, i].boxplot(box_wscg[i], showfliers=False, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        if i == 0:
            axs[2, i].set_ylabel('WSCG', fontsize=14, labelpad=10)
        # axs[2, i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs[2, i].set_xticks([])
        axs[2, i].set_xticklabels([])
    line_legend = mlines.Line2D([], [], color='red', linestyle='--', label=r'$1-\alpha$')
    fig.legend(handles=legend_patches + [line_legend], loc='lower center', ncol=3, prop={'size': 12})
    plt.show()

    fig.savefig(
            output_dir / f"feature_ablation_alpha_{alpha:g}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    
    plt.close(fig)