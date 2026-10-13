from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Render to files without opening a window.

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset_list = ['bike','bio','traffic','states','hospital']
versions = [1,2,3,4,5,6,7,8,9,10]
alpha_list = [0.1]

COVERAGE_TEST = []
WSC_TEST = []
WSCG_TEST = []

for idx, dataset in enumerate(dataset_list):
    COVERAGE_TEST.append({})
    COVERAGE_TEST[idx]['dataset'] = dataset
    WSC_TEST.append({})
    WSC_TEST[idx]['dataset'] = dataset
    WSCG_TEST.append({})
    WSCG_TEST[idx]['dataset'] = dataset

    nfcqr_wsc = []
    nfcqr_wscg = []
    nfcqr_marginal_coverage = []
    cqr_wsc = []
    cqr_wscg = []
    cqr_marginal_coverage = []
    cqr_iid_wsc = []
    cqr_iid_wscg = []
    cqr_iid_marginal_coverage = []


    for v_idx, v in enumerate(versions):
        NFCQR_RESULT = pickle.load(open("result/" + dataset + "/msdg/bnfcqr_aug_v" + str(v) + "_msdg", "rb"))
        CQR_RESULT = pickle.load(open("result/" + dataset + "/msdg/cqr_v" + str(v) + "_msdg", "rb"))
        CQR_IID_RESULT = pickle.load(open("result/" + dataset + "/iid/cqr_v" + str(v) + "_iid", "rb"))

        for i in NFCQR_RESULT:
            if 'data' in i and i['data'] == 'test':
                nfcqr_wsc.append(i['wsc'])
                nfcqr_wscg.append(i['wscg'])
                nfcqr_marginal_coverage.append(i['marginal coverage'])

        for i in CQR_RESULT:
            if 'data' in i and i['data'] == 'test':
                cqr_wsc.append(i['wsc'])
                cqr_wscg.append(i['wscg'])
                cqr_marginal_coverage.append(i['marginal coverage'])

        for i in CQR_IID_RESULT:
            if 'data' in i and i['data'] == 'test':
                cqr_iid_wsc.append(i['wsc'])
                cqr_iid_wscg.append(i['wscg'])
                cqr_iid_marginal_coverage.append(i['marginal coverage'])

    nfcqr_wsc = np.stack(nfcqr_wsc)
    cqr_wsc = np.stack(cqr_wsc)
    cqr_iid_wsc = np.stack(cqr_iid_wsc)

    nfcqr_wscg = np.stack(nfcqr_wscg)
    cqr_wscg = np.stack(cqr_wscg)
    cqr_iid_wscg = np.stack(cqr_iid_wscg)

    nfcqr_marginal_coverage = np.stack(nfcqr_marginal_coverage)
    cqr_marginal_coverage = np.stack(cqr_marginal_coverage)
    cqr_iid_marginal_coverage = np.stack(cqr_iid_marginal_coverage)

    COVERAGE_TEST[idx]['nfcqr'] = nfcqr_marginal_coverage
    COVERAGE_TEST[idx]['cqr'] = cqr_marginal_coverage
    COVERAGE_TEST[idx]['cqr_iid'] = cqr_iid_marginal_coverage

    WSC_TEST[idx]['nfcqr'] = nfcqr_wsc
    WSC_TEST[idx]['cqr'] = cqr_wsc
    WSC_TEST[idx]['cqr_iid'] = cqr_iid_wsc

    WSCG_TEST[idx]['nfcqr'] = nfcqr_wscg
    WSCG_TEST[idx]['cqr'] = cqr_wscg
    WSCG_TEST[idx]['cqr_iid'] = cqr_iid_wscg

# COVERAGE_TEST, GAP_TEST, SIZE_TEST = COVERAGE_CAL, GAP_CAL, SIZE_CAL
cqr_coverage_alpha = []
cqr_iid_coverage_alpha = []
nfcqr_coverage_alpha = []

cqr_wsc_alpha = []
cqr_iid_wsc_alpha = []
nfcqr_wsc_alpha = []

cqr_wscg_alpha = []
cqr_iid_wscg_alpha = []
nfcqr_wscg_alpha = []

for a, alpha in enumerate(alpha_list):
    box_coverage = []
    box_wsc = []
    box_wscg = []

    for idx, dataset in enumerate(dataset_list):
        box_coverage.append(
            [COVERAGE_TEST[idx]['cqr'][:, a], COVERAGE_TEST[idx]['cqr_iid'][:, a], COVERAGE_TEST[idx]['nfcqr'][:, a]])
        box_wsc.append([WSC_TEST[idx]['cqr'][:, a], WSC_TEST[idx]['cqr_iid'][:, a], WSC_TEST[idx]['nfcqr'][:, a]])
        box_wscg.append([WSCG_TEST[idx]['cqr'][:, a], WSCG_TEST[idx]['cqr_iid'][:, a], WSCG_TEST[idx]['nfcqr'][:, a]])

    cqr_coverage_alpha.append(
        np.fromiter((box_coverage[b][0] for b in range(len(box_coverage))),
                    dtype=np.dtype((float, len(box_coverage[0][0])))).flatten())
    cqr_iid_coverage_alpha.append(
        np.fromiter((box_coverage[b][1] for b in range(len(box_coverage))),
                    dtype=np.dtype((float, len(box_coverage[0][1])))).flatten())
    nfcqr_coverage_alpha.append(
        np.fromiter((box_coverage[b][2] for b in range(len(box_coverage))),
                    dtype=np.dtype((float, len(box_coverage[0][2])))).flatten())

    cqr_wsc_alpha.append(
        np.fromiter((box_wsc[b][0] for b in range(len(box_wsc))),
                    dtype=np.dtype((float, len(box_coverage[0][0])))).flatten())
    cqr_iid_wsc_alpha.append(
        np.fromiter((box_wsc[b][1] for b in range(len(box_wsc))),
                    dtype=np.dtype((float, len(box_coverage[0][1])))).flatten())
    nfcqr_wsc_alpha.append(
        np.fromiter((box_wsc[b][2] for b in range(len(box_wsc))),
                    dtype=np.dtype((float, len(box_coverage[0][2])))).flatten())

    cqr_wscg_alpha.append(
        np.fromiter((box_wscg[b][0] for b in range(len(box_wscg))),
                    dtype=np.dtype((float, len(box_coverage[0][0])))).flatten())
    cqr_iid_wscg_alpha.append(
        np.fromiter((box_wscg[b][1] for b in range(len(box_wscg))),
                    dtype=np.dtype((float, len(box_coverage[0][1])))).flatten())
    nfcqr_wscg_alpha.append(
        np.fromiter((box_wscg[b][2] for b in range(len(box_wscg))),
                    dtype=np.dtype((float, len(box_coverage[0][2])))).flatten())

colors = ['lightpink', 'deeppink', 'lightblue']
labels = ['CQR under non-i.i.d.', 'CQR under i.i.d.','Augmented BNF+CQR under non-i.i.d.']

coverage_method = [cqr_coverage_alpha[0], cqr_iid_coverage_alpha[0], nfcqr_coverage_alpha[0]]
wsc_method = [cqr_wsc_alpha[0], cqr_iid_wsc_alpha[0], nfcqr_wsc_alpha[0]]
wscg_method = [cqr_wscg_alpha[0], cqr_iid_wscg_alpha[0], nfcqr_wscg_alpha[0]]

boxes = [coverage_method, wsc_method, wscg_method]
# Match the reference PDF: 474.48 x 138.24 points = 6.59 x 1.92 inches.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "ytick.labelsize": 10,
    "figure.autolayout": False,
    "figure.constrained_layout.use": False,
    "savefig.bbox": None,
})
fig, axs = plt.subplots(1, 3, figsize=(6.59, 1.92))
# Reserve the lower part of the canvas for the two-row legend.
fig.subplots_adjust(
    left=0.0809524, right=0.98,
    bottom=0.3857143, top=0.9714286,
    wspace=0.43095,
)

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
fig.legend(
    handles=legend_patches + [line_legend],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=2,
    fontsize=12,
    frameon=True,
)

output_dir = Path("figure/msdg")
output_dir.mkdir(parents=True, exist_ok=True)
# Keep the fixed canvas dimensions; tight cropping would change the PDF size.
fig.savefig(output_dir / "cond_lower_bound.pdf", bbox_inches=None)
plt.close(fig)
