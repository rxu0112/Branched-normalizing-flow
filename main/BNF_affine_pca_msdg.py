import argparse
import torch
import torch.optim as optim
import normflows as nf
import numpy as np
from tqdm import tqdm
import math
import torch.nn as nn
from geomloss import SamplesLoss
from sklearn.decomposition import PCA
import pickle
import pykeops
from main.module import *

parser = argparse.ArgumentParser(description='PCA-conditioned BNF + CQR under multi-source domain generalization')
parser.add_argument('--mlp_hidden_dim', type=int, default=64)
parser.add_argument('--nf_num_layer', type=int, default=48)
parser.add_argument('--bins', type=int, default=10)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--mlp_lr', type=float, default=1e-3)  # 0.001
parser.add_argument('--nf_lr', type=float, default=5e-4)
parser.add_argument('--mlp_steps', type=int, default=3000)
parser.add_argument('--nf_steps', type=int, default=2000)
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--num_size_eval_env', type=int, default=10)
parser.add_argument('--blur', type=float, default=1e-3) 
parser.add_argument('--dataset', type=list, default=['bike','bio','traffic','states'])
parser.add_argument('--version', type=list, default=['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10'])
parser.add_argument('--device', type=str, default='cuda:0')
flags = parser.parse_args()

def adjust_learning_rate(optimizer, step):
    """ Adjust the learning rate based on the iteration step.
    same setup for all datasets and ablation studies: 5e-4 [0, 500], 3e-4 [500,1000], 1e-4 [1000,2000]
    """
    if step == 500:
        lr = 3e-4
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif step == 1000:
        lr = 1e-4
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def find_spectrum(f, scaled_image, pca_image, num_samples):
    with torch.no_grad():
        mid_samples = torch.linspace(0, 1, steps=num_samples).unsqueeze(1).to(flags.device)
        x_vals = torch.column_stack([scaled_image.repeat(num_samples, 1), mid_samples, pca_image.repeat(num_samples, 1)])
        y_vals = f.forward(x_vals)[:, dim-1]
    return y_vals

def find_valid_range(y_vals, c, d, num_samples):

    mid_samples = torch.linspace(0, 1, steps=num_samples).unsqueeze(1).to(flags.device)

    valid_intervals = []
    in_range = (y_vals >= c) & (y_vals <= d)
    i = 0
    while i < len(mid_samples):
        if in_range[i]:
            start = mid_samples[i]
            while i < len(mid_samples) - 1 and in_range[i + 1]:
                i += 1
            end = mid_samples[i]
            valid_intervals.append((start, end))
        i += 1

    return valid_intervals


class ConditionalAffineFlow(nn.Module):  # Inherit from nn.Module
    def __init__(self, dim, cut, act):
        super(ConditionalAffineFlow, self).__init__()  # Call super().__init__() first
        self.dim = dim
        self.cut = cut

        # Affine coupling layer for the first n-2 dimensions
        param_map_first = nf.nets.MLP(
            [math.ceil((dim - cut) / 2), 64, 128, 256, 128, 64, ((dim - cut) - math.ceil((dim - cut) / 2)) * 2],
            init_zeros=True, output_fn=act)
        self.affine_coupling_block_first = nf.flows.AffineCouplingBlock(param_map_first)

        # Affine coupling layer for the last two dimensions
        param_map_last = nf.nets.MLP([math.ceil(cut / 2), 64, 128, 256, 128, 64, (cut - math.ceil(cut / 2)) * 2],
                                     init_zeros=True, output_fn=act)
        self.affine_coupling_block_last = nf.flows.AffineCouplingBlock(param_map_last)

        self.permute_first = nf.flows.Permute(dim - cut, mode='swap')
        self.permute_last = nf.flows.Permute(cut, mode='swap')

    def forward(self, z):
        z_first, log_det_first1 = self.affine_coupling_block_first(z[:, :(self.dim - self.cut)])
        z_temp, log_det_last1 = self.affine_coupling_block_last(z[:, (self.dim - self.cut):])

        z_first, log_det_first2 = self.permute_first(z_first)
        z_temp, log_det_last2 = self.permute_last(z_temp)

        log_det_first = log_det_first1 + log_det_first2
        log_det_last = log_det_last1 + log_det_last2

        # Concatenate the transformed parts back together
        z_out = torch.cat([z_first, z_temp], dim=1)

        # Sum the log determinants from both transformations
        log_det = log_det_first + log_det_last

        return z_out, log_det


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        lin1 = nn.Linear(input_size, flags.mlp_hidden_dim)
        lin2 = nn.Linear(flags.mlp_hidden_dim, flags.mlp_hidden_dim)
        lin3 = nn.Linear(flags.mlp_hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(
            lin1, nn.Tanh(),  # nn.ReLU(True),
            nn.Dropout(),
            lin2, nn.Tanh(),  # nn.ReLU(True),
            nn.Dropout(),
            lin3)

    def forward(self, x):
        x = x.view(x.shape[0], self.input_size)
        out = self._main(x)
        return out


if __name__ == "__main__":
    # torch.manual_seed(0)
    print('Flags:')
    for k, v in sorted(vars(flags).items()):
        print("    {}: {}".format(k, v))
    
    
    pykeops.test_numpy_bindings()
    pykeops.test_torch_bindings()
    sinkhorn_divergence = SamplesLoss("sinkhorn", blur=flags.blur, p=2)

    for dataset in flags.dataset:
        for ver in flags.version:
            print(dataset + '_' + ver)

            # drop data
            train_envs = pickle.load(open("data/" + dataset + "/msdg/train_" + ver, "rb"))
            test_envs = pickle.load(open("data/" + dataset + "/msdg/test_" + ver, "rb"))
            cal_envs_uni = np.load("data/" + dataset + "/msdg/cal_uni_" + ver + '.npy',
                                   allow_pickle='TRUE').item()

            universal_image_train = []
            universal_label_train = []
            for e in train_envs:
                universal_image_train.append(e['images'])
                universal_label_train.append(e['labels'])
            universal_image_train = torch.concatenate(universal_image_train)
            universal_label_train = torch.concatenate(universal_label_train)
            train_envs_uni = ({'images': universal_image_train, 'labels': universal_label_train, 'info': 'universal'})
            
            for env in train_envs+test_envs+[cal_envs_uni]+[train_envs_uni]:
                env['images'] = env['images'].to(flags.device)
                env['labels'] = env['labels'].to(flags.device)
                if 'tags' in env:
                    env['tags'] = env['tags'].to(flags.device)

            RESULT = []
            alpha_array = np.around(np.linspace(0.1, 0.9, 9), 2)
            # alpha_array = [0.1]
            # train empirical risk minimization model
            print('train empirical risk minimization model')
            MLP_LO = []
            MLP_HI = []
            for alpha in alpha_array:
                mlp_lo = MLP(train_envs[0]['images'].shape[1]).to(flags.device)
                optimizer_mlp_lo = optim.Adam(mlp_lo.parameters(), lr=flags.mlp_lr)
                mlp_hi = MLP(train_envs[0]['images'].shape[1]).to(flags.device)
                optimizer_mlp_hi = optim.Adam(mlp_hi.parameters(), lr=flags.mlp_lr)

                for step in tqdm(range(flags.mlp_steps), desc=f"Training for alpha={alpha}", ncols=100,
                                 mininterval=1.0):
                    for env in train_envs:
                        env['pred_lo'] = mlp_lo(env['images'])
                        env['pinball_lo'] = pinball_loss(env['pred_lo'], env['labels'], alpha=alpha / 2)
                        env['pred_hi'] = mlp_hi(env['images'])
                        env['pinball_hi'] = pinball_loss(env['pred_hi'], env['labels'], alpha=1 - alpha / 2)
                    loss_lo = torch.stack([e['pinball_lo'] for e in train_envs]).mean()
                    loss_hi = torch.stack([e['pinball_hi'] for e in train_envs]).mean()

                    weight_norm_lo = torch.tensor(0.).to(flags.device)
                    for w in mlp_lo.parameters():
                        weight_norm_lo += w.norm().pow(2)
                    loss_lo += flags.l2_regularizer_weight * weight_norm_lo

                    weight_norm_hi = torch.tensor(0.).to(flags.device)
                    for w in mlp_hi.parameters():
                        weight_norm_hi += w.norm().pow(2)
                    loss_hi += flags.l2_regularizer_weight * weight_norm_hi

                    optimizer_mlp_lo.zero_grad()
                    loss_lo.backward()
                    optimizer_mlp_lo.step()
                    optimizer_mlp_hi.zero_grad()
                    loss_hi.backward()
                    optimizer_mlp_hi.step()

                MLP_LO.append(mlp_lo)
                MLP_HI.append(mlp_hi)

            # transfer data
            trans_image_universal = []
            trans_label_universal = []
            for env in train_envs + test_envs + [train_envs_uni]:
                env['trans_labels'] = torch.sign(env['labels']) * torch.log1p(torch.abs(env['labels']))
                env['trans_images'] = torch.sign(env['images']) * torch.log1p(torch.abs(env['images']))
                trans_image_universal.append(env['trans_images'])
                trans_label_universal.append(env['trans_labels'])
            trans_image_universal = torch.concatenate(trans_image_universal)
            trans_label_universal = torch.concatenate(trans_label_universal)

            # whiten data
            whiten_image_universal = []
            whiten_label_universal = []
            for env in train_envs + test_envs + [train_envs_uni]:
                env['whiten_images'] = whiten(env['trans_images'], trans_image_universal)
                env['whiten_labels'] = whiten(env['trans_labels'], trans_label_universal)
                whiten_image_universal.append(env['whiten_images'])
                whiten_label_universal.append(env['whiten_labels'])
            whiten_image_universal = torch.concatenate(whiten_image_universal)
            whiten_label_universal = torch.concatenate(whiten_label_universal)

            # scale data
            for env in train_envs + test_envs + [train_envs_uni]:
                env['scaled_images'] = scale(env['whiten_images'], whiten_image_universal)
                env['scaled_labels'] = scale(env['whiten_labels'], whiten_label_universal)

            # base
            train_envs_uni['scaled_base'] = torch.column_stack(
                (train_envs_uni['scaled_images'], train_envs_uni['scaled_labels']))

            # pca conditional target
            train_scaled_images = torch.cat([env['scaled_images'] for env in train_envs], dim=0)
            pca = PCA(n_components=1)
            pca.fit(train_scaled_images.cpu().numpy()) 
            for env in train_envs + test_envs:
                images_np = env['scaled_images'].cpu().numpy()
                env['pca_images'] = torch.from_numpy(pca.transform(images_np)).float().to(flags.device)
                env['pca_conditional_scaled_target'] = torch.column_stack((
                    env['scaled_images'],
                    env['scaled_labels'],
                    env['pca_images']  # <-- use PCA features here
                ))
    

            dim = train_envs[0]['images'].shape[1] + train_envs[0]['labels'].shape[1] + 1
            cut = train_envs[0]['labels'].shape[1] + 1

            # train a normalizing flow
            base = nf.distributions.base.DiagGaussian(dim)
            flows_branched = [ConditionalAffineFlow(dim, cut, act='tanh') for _ in range(flags.nf_num_layer - 1)]
            flows_branched.append(ConditionalAffineFlow(dim, cut, act=None))
            model_branched = nf.NormalizingFlow(base, flows_branched).to(flags.device)

            loss_hist = np.array([])
            optimizer_nf = torch.optim.Adam(model_branched.parameters(), lr=flags.nf_lr)

            print('train a normalizing flow')
            for step in tqdm(range(flags.nf_steps)):
                optimizer_nf.zero_grad()

                adjust_learning_rate(optimizer_nf, step)

                for env in train_envs:
                    env['scaled_normalized'] = model_branched.forward(env['pca_conditional_scaled_target'].contiguous())[:, :-1]
                    env['W-distance'] = sinkhorn_divergence(env['scaled_normalized'].contiguous(),
                                                            train_envs_uni['scaled_base'].contiguous())

                loss = torch.stack([e['W-distance'] for e in train_envs]).mean()
                if not (torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer_nf.step()

                loss_hist = np.append(loss_hist, loss.cpu().data.numpy())

            RESULT.append({'loss_hist': loss_hist})
            model_branched.eval()

            # marginal coverage, conditional gap, and set size on calibration data
            tau_list = []
            marginal_coverage_cal_list = []
            conditional_gap_cal_list = []
            size_cal_list = []
            for a_idx, alpha in enumerate(alpha_array):
                cal_envs_uni['pred_lo'] = MLP_LO[a_idx](cal_envs_uni['images'])
                cal_envs_uni['pred_hi'] = MLP_HI[a_idx](cal_envs_uni['images'])
                R_cal = []
                for i in range(len(cal_envs_uni['labels'])):
                    R_cal.append(max(cal_envs_uni['pred_lo'][i] - cal_envs_uni['labels'][i], cal_envs_uni['labels'][i] -
                                     cal_envs_uni['pred_hi'][i]))
                R_cal = torch.tensor(R_cal)
                tau = np.quantile(R_cal, 1 - alpha)
                covered_index = (cal_envs_uni['labels'] >= cal_envs_uni['pred_lo'] - tau) & (
                        cal_envs_uni['labels'] <= cal_envs_uni['pred_hi'] + tau)
                marginal_coverage = (torch.sum(covered_index) / len(cal_envs_uni['labels'])).cpu().detach().numpy()

                conditional_gap = calculate_adaptivity_bins(cal_envs_uni['images'], covered_index, num_bins=flags.bins,
                                                            alpha=alpha)
                size = torch.mean(
                    (cal_envs_uni['pred_hi'] + tau) - (cal_envs_uni['pred_lo'] - tau)).cpu().detach().numpy()

                tau_list.append(tau)
                marginal_coverage_cal_list.append(marginal_coverage)
                conditional_gap_cal_list.append(conditional_gap)
                size_cal_list.append(size)

            RESULT.append(
                {'data': 'cal', 'alpha': alpha_array, 'tau': tau_list, 'marginal coverage': marginal_coverage_cal_list,
                 'conditional gap': conditional_gap_cal_list, 'size': size_cal_list})

            # marginal coverage, conditional gap, and set size on test data
            print('prediction on test envs')
            for e_idx, env in enumerate(tqdm(test_envs)):
                with torch.no_grad():
                    env['scaled_normalized'] = model_branched.forward(env['pca_conditional_scaled_target'])[:, :-1]  # feature part independent to label
                    env['scaled_normalized_images'] = env['scaled_normalized'][:, :-1]
                    env['whiten_normalized_images'] = scale_inverse(env['scaled_normalized_images'],
                                                                    whiten_image_universal)
                    env['trans_normalized_images'] = whiten_inverse(env['whiten_normalized_images'],
                                                                    trans_image_universal)
                    env['normalized_images'] = torch.sign(env['trans_normalized_images']) * torch.expm1(
                        torch.abs(env['trans_normalized_images']))
                    
                    env['scaled_normalized_labels'] = env['scaled_normalized'][:, -1]
                    env['whiten_normalized_labels'] = scale_inverse(env['scaled_normalized_labels'],
                                                                    whiten_label_universal)
                    env['trans_normalized_labels'] = whiten_inverse(env['whiten_normalized_labels'],
                                                                    trans_label_universal)
                    env['normalized_labels'] = torch.sign(env['trans_normalized_labels']) * torch.expm1(
                        torch.abs(env['trans_normalized_labels']))
                    
                if e_idx < flags.num_size_eval_env:
                    y_spectrum_list = []
                    for s_idx, _ in enumerate(env['scaled_images']):
                        scaled_image = env['scaled_images'][s_idx]
                        pca_image = env['pca_images'][s_idx]
                        y_spectrum = find_spectrum(model_branched, scaled_image, pca_image, num_samples=flags.num_samples)
                        y_spectrum_list.append(y_spectrum)

                marginal_coverage_test_list = []
                conditional_gap_test_list = []
                conditional_gap_test_normalized_list = []
                size_test_list = []
                wsc_test_list = []
                wscg_test_list = []
                ascg_multi_test_list = []
                for a_idx, alpha in enumerate(alpha_array):
                    tau = tau_list[a_idx]
                    normalized_pred_lo = MLP_LO[a_idx](env['normalized_images'])
                    normalized_bound_lo = normalized_pred_lo - tau

                    normalized_pred_hi = MLP_HI[a_idx](env['normalized_images'])
                    normalized_bound_hi = normalized_pred_hi + tau

                    if e_idx < flags.num_size_eval_env:
                        size_env = []
                        trans_normalized_bound_lo = torch.sign(normalized_bound_lo) * torch.log1p(torch.abs(normalized_bound_lo))
                        whiten_normalized_bound_lo = whiten(trans_normalized_bound_lo, trans_label_universal)
                        scaled_normalized_bound_lo = scale(whiten_normalized_bound_lo, whiten_label_universal)
                        trans_normalized_bound_hi = torch.sign(normalized_bound_hi) * torch.log1p(torch.abs(normalized_bound_hi))
                        whiten_normalized_bound_hi = whiten(trans_normalized_bound_hi, trans_label_universal)
                        scaled_normalized_bound_hi = scale(whiten_normalized_bound_hi, whiten_label_universal)

                        for s_idx, _ in enumerate(env['scaled_images']):
                            scaled_valid_intervals = find_valid_range(y_spectrum_list[s_idx], scaled_normalized_bound_lo[s_idx],
                                                            scaled_normalized_bound_hi[s_idx], num_samples=flags.num_samples)
                            if scaled_valid_intervals:
                                scaled_valid_tensor = torch.cat([torch.cat(tup) for tup in scaled_valid_intervals]).reshape(-1,1)
                                whiten_valid_tensor = scale_inverse(scaled_valid_tensor, whiten_label_universal)
                                trans_valid_tensor = whiten_inverse(whiten_valid_tensor, trans_label_universal)
                                valid_tensor = torch.sign(trans_valid_tensor) * torch.expm1(torch.abs(trans_valid_tensor))
                                valid_intervals = [tuple(row) for row in valid_tensor.reshape(-1, 2)]
                                size = sum(end - start for start, end in valid_intervals)
                                size_env.append(size)
                            else:
                                size_env.append(torch.tensor([0.0]).to(flags.device))
                        size_env_mean = torch.mean(torch.tensor(size_env)).cpu().detach().numpy()
                    else:
                        size_env_mean = np.mean([
                            env['size'][a_idx]
                            for env in RESULT[2:2 + flags.num_size_eval_env]
                            if env['data'] == 'test'
                        ])
                    size_test_list.append(size_env_mean)

                    covered_env = (env['normalized_labels'].reshape(-1, 1) >= normalized_bound_lo) & (env['normalized_labels'].reshape(-1, 1) <= normalized_bound_hi)
                    ascg_multi = compute_ascg_multi_gpu(env['images'], covered_env, alpha)
                    wsc = compute_wsc_gpu(env['images'], covered_env)
                    wscg = compute_wscg_gpu(env['images'], covered_env, alpha)
                    marginal_coverage = (torch.sum(torch.tensor(covered_env)) / len(env['labels'])).cpu().detach().numpy()
                    conditional_gap = calculate_adaptivity_bins(env['images'], torch.tensor(covered_env), num_bins=flags.bins, alpha=alpha)
                    conditional_gap_normalized = calculate_adaptivity_bins(env['normalized_images'], torch.tensor(covered_env), num_bins=flags.bins, alpha=alpha)
                    marginal_coverage_test_list.append(marginal_coverage)
                    conditional_gap_test_list.append(conditional_gap)
                    conditional_gap_test_normalized_list.append(conditional_gap_normalized)
                    ascg_multi_test_list.append(ascg_multi)
                    wsc_test_list.append(wsc)
                    wscg_test_list.append(wscg)

                RESULT.append(
                    {'data': 'test', 'alpha': alpha_array, 'marginal coverage': marginal_coverage_test_list,
                    'conditional gap': conditional_gap_test_list,
                    'conditional gap normalized': conditional_gap_test_normalized_list,
                    'size': size_test_list, 'wsc': wsc_test_list, 'wscg':wscg_test_list, 'ascg multi': ascg_multi_test_list})

            pickle.dump(RESULT,
                        open("result/" + dataset + "/msdg/bnfcqr_pca_" + ver + "_msdg", "wb"))
