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

nfcqr_aug_coverage_alpha = []
nfcqr_fea_cond_coverage_alpha = []
nfcqr_fea_pca_coverage_alpha = []
nfcqr_fea_tsne_coverage_alpha = []
nfcqr_fea_umap_coverage_alpha = []

nfcqr_aug_wsc_alpha = []
nfcqr_fea_cond_wsc_alpha = []
nfcqr_fea_pca_wsc_alpha = []
nfcqr_fea_tsne_wsc_alpha = []
nfcqr_fea_umap_wsc_alpha = []

nfcqr_aug_wscg_alpha = []
nfcqr_fea_cond_wscg_alpha = []
nfcqr_fea_pca_wscg_alpha = []
nfcqr_fea_tsne_wscg_alpha = []
nfcqr_fea_umap_wscg_alpha = []

for a, alpha in enumerate(alpha_list):
    box_coverage = []
    box_wscg = []
    box_wsc = []

    for idx, dataset in enumerate(dataset_list):
        box_coverage.append([COVERAGE_TEST[idx]['nfcqr_aug'][:, a], COVERAGE_TEST[idx]['nfcqr_fea_cond'][:, a],COVERAGE_TEST[idx]['nfcqr_fea_pca'][:, a],COVERAGE_TEST[idx]['nfcqr_fea_tsne'][:, a],COVERAGE_TEST[idx]['nfcqr_fea_umap'][:, a]])
        box_wscg.append([WSCG_TEST[idx]['nfcqr_aug'][:, a], WSCG_TEST[idx]['nfcqr_fea_cond'][:, a], WSCG_TEST[idx]['nfcqr_fea_pca'][:, a], WSCG_TEST[idx]['nfcqr_fea_tsne'][:, a],WSCG_TEST[idx]['nfcqr_fea_umap'][:, a]])
        box_wsc.append([WSC_TEST[idx]['nfcqr_aug'][:, a], WSC_TEST[idx]['nfcqr_fea_cond'][:, a], WSC_TEST[idx]['nfcqr_fea_pca'][:, a], WSC_TEST[idx]['nfcqr_fea_tsne'][:, a],WSCG_TEST[idx]['nfcqr_fea_umap'][:, a]])

    nfcqr_aug_coverage_alpha.append(
        np.fromiter((box_coverage[b][0] for b in range(len(box_coverage))),
                    dtype=np.dtype((float, len(box_coverage[0][0])))).flatten())
    nfcqr_fea_cond_coverage_alpha.append(
        np.fromiter((box_coverage[b][1] for b in range(len(box_coverage))),
                    dtype=np.dtype((float, len(box_coverage[0][1])))).flatten())
    nfcqr_fea_pca_coverage_alpha.append(
        np.fromiter((box_coverage[b][2] for b in range(len(box_coverage))),
                    dtype=np.dtype((float, len(box_coverage[0][2])))).flatten())
    nfcqr_fea_tsne_coverage_alpha.append(
        np.fromiter((box_coverage[b][3] for b in range(len(box_coverage))),
                    dtype=np.dtype((float, len(box_coverage[0][3])))).flatten())
    nfcqr_fea_umap_coverage_alpha.append(
        np.fromiter((box_coverage[b][4] for b in range(len(box_coverage))),
                    dtype=np.dtype((float, len(box_coverage[0][4])))).flatten())

    nfcqr_aug_wsc_alpha.append(
        np.fromiter((box_wsc[b][0] for b in range(len(box_wsc))),
                    dtype=np.dtype((float, len(box_coverage[0][0])))).flatten())
    nfcqr_fea_cond_wsc_alpha.append(
        np.fromiter((box_wsc[b][1] for b in range(len(box_wsc))),
                    dtype=np.dtype((float, len(box_coverage[0][1])))).flatten())
    nfcqr_fea_pca_wsc_alpha.append(
        np.fromiter((box_wsc[b][2] for b in range(len(box_wsc))),
                    dtype=np.dtype((float, len(box_coverage[0][2])))).flatten())
    nfcqr_fea_tsne_wsc_alpha.append(
        np.fromiter((box_wsc[b][3] for b in range(len(box_wsc))),
                    dtype=np.dtype((float, len(box_coverage[0][3])))).flatten())
    nfcqr_fea_umap_wsc_alpha.append(
        np.fromiter((box_wsc[b][4] for b in range(len(box_wsc))),
                    dtype=np.dtype((float, len(box_coverage[0][4])))).flatten())

    nfcqr_aug_wscg_alpha.append(
        np.fromiter((box_wscg[b][0] for b in range(len(box_wscg))),
                    dtype=np.dtype((float, len(box_coverage[0][0])))).flatten())
    nfcqr_fea_cond_wscg_alpha.append(
        np.fromiter((box_wscg[b][1] for b in range(len(box_wscg))),
                    dtype=np.dtype((float, len(box_coverage[0][1])))).flatten())
    nfcqr_fea_pca_wscg_alpha.append(
        np.fromiter((box_wscg[b][2] for b in range(len(box_wscg))),
                    dtype=np.dtype((float, len(box_coverage[0][2])))).flatten())
    nfcqr_fea_tsne_wscg_alpha.append(
        np.fromiter((box_wscg[b][3] for b in range(len(box_wscg))),
                    dtype=np.dtype((float, len(box_coverage[0][3])))).flatten())
    nfcqr_fea_umap_wscg_alpha.append(
        np.fromiter((box_wscg[b][4] for b in range(len(box_wscg))),
                    dtype=np.dtype((float, len(box_coverage[0][4])))).flatten())

colors = ['lightblue', 'crimson','bisque','mediumturquoise','pink']
labels = ['Augmented BNF+CQR', 'Feature-conditioned BNF+CQR','PCA-conditioned BNF+CQR','tSNE-conditioned BNF+CQR','UMAP-conditioned BNF+CQR']

coverage_method = [nfcqr_aug_coverage_alpha[0], nfcqr_fea_cond_coverage_alpha[0], nfcqr_fea_pca_coverage_alpha[0],nfcqr_fea_tsne_coverage_alpha[0],nfcqr_fea_umap_coverage_alpha[0]]
wsc_method = [nfcqr_aug_wsc_alpha[0], nfcqr_fea_cond_wsc_alpha[0], nfcqr_fea_pca_wsc_alpha[0],nfcqr_fea_tsne_wsc_alpha[0],nfcqr_fea_umap_wsc_alpha[0]]
wscg_method = [nfcqr_aug_wscg_alpha[0], nfcqr_fea_cond_wscg_alpha[0], nfcqr_fea_pca_wscg_alpha[0],nfcqr_fea_tsne_wscg_alpha[0],nfcqr_fea_umap_wscg_alpha[0]]

boxes = [coverage_method, wsc_method, wscg_method]
fig, axs = plt.subplots(1, 3, figsize=(6.89, 2.38))
fig.subplots_adjust(left=0.074, bottom=0.436, right=0.98, top=0.98, hspace=0.3, wspace=0.37)

for i in range(3):
    box = axs[i].boxplot(boxes[i], showfliers=False, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    axs[i].title.set_fontsize(13)
    if i == 0:
        axs[i].set_ylabel('marginal', fontsize=14, labelpad=0)
        axs[i].axhline(y=0.9, color='r', linestyle='--')
        axs[i].set_ylim(0.7, 1.01)
    if i == 1:
        axs[i].set_ylabel('WSC', fontsize=14, labelpad=0)
        axs[i].axhline(y=0.9, color='r', linestyle='--')
    if i == 2:
        axs[i].set_ylabel('WSCG', fontsize=14, labelpad=0)
    axs[i].set_xticks([])
    axs[i].set_xticklabels([])
    axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

line_legend = mlines.Line2D([], [], color='red', linestyle='--', label=r'$1-\alpha$')
legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
fig.legend(handles=legend_patches + [line_legend], fontsize=14, loc='lower center', ncol=2, prop={'size': 12})

fig.savefig(
            output_dir / f"feature_ablation_alpha_{alpha:g}_over_datasets.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    
plt.close(fig)
