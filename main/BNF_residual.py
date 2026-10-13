import argparse
import torch
import torch.optim as optim
import normflows as nf
import numpy as np
from tqdm import tqdm
import math
import torch.nn as nn
from geomloss import SamplesLoss
import pickle
import pykeops
from main.module import *

parser = argparse.ArgumentParser(description='Residual BNF + CQR under multi-source domain generalization')
parser.add_argument('--mlp_hidden_dim', type=int, default=64)
parser.add_argument('--nf_num_layer', type=int, default=16) # 16
parser.add_argument('--bins', type=int, default=10)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--mlp_lr', type=float, default=1e-3)  # 0.001
parser.add_argument('--nf_lr', type=float, default=3e-4) # 3e-4
parser.add_argument('--mlp_steps', type=int, default=3000) 
parser.add_argument('--nf_steps', type=int, default=10000)
parser.add_argument('--blur', type=float, default=1e-3) 
parser.add_argument('--dataset', type=list, default=['bike','bio','traffic','states'])
parser.add_argument('--version', type=list, default=['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10'])
parser.add_argument('--device', type=str, default='cuda:0')
flags = parser.parse_args()

def bound_generative(bound, env_local):
    bound_scaled = scale(whiten(torch.sign(bound) * torch.log1p(torch.abs(bound)), trans_universal[:, -1]),
                      whiten_universal[:, -1])
    bound_scaled_gen = model_branched.forward(
        torch.concatenate((env_local['scaled_target'][:, 0:-1], bound_scaled), axis=1))[:,
                    -1]  # f_X branch with input only for functionality
    bound_gen = torch.sign(whiten_inverse(scale_inverse(bound_scaled_gen, whiten_universal[:, -1]),
                                       trans_universal[:, -1])) * torch.expm1(
        torch.abs(whiten_inverse(scale_inverse(bound_scaled_gen, whiten_universal[:, -1]),
                                 trans_universal[:, -1])))
    return bound_gen



class IndependentResidualFlow(nn.Module):  # Inherit from nn.Module
    def __init__(self, dim, cut):
        super(IndependentResidualFlow, self).__init__()  # Call super().__init__() first
        self.dim = dim
        self.cut = cut

        # Residual layer for the first n-1 dimensions
        net_first = nf.nets.LipschitzMLP([dim - cut, 128, 128, dim - cut],
                                         init_zeros=True, lipschitz_const=0.90)
        self.residual_block_first = nf.flows.Residual(net_first, reduce_memory=True)

        # Residual layer for the last dimension
        net_last = nf.nets.LipschitzMLP([cut, 128, 128, cut],
                                        init_zeros=True, lipschitz_const=0.90)
        self.residual_block_last = nf.flows.Residual(net_last, reduce_memory=True)

        self.actnorm_first = nf.flows.ActNorm(dim - cut)
        self.actnorm_last = nf.flows.ActNorm(cut)

    def forward(self, z):
        # Split the input into first n-1 dimensions and the last dimension
        z_first, z_last = z[:, :-self.cut], z[:, -self.cut:]

        # Apply residual layer to the first n-1 dimensions
        z_first, log_det_first2 = self.actnorm_first(z_first)
        z_first, log_det_first1 = self.residual_block_first(z_first)

        log_det_first = log_det_first1 + log_det_first2

        # Apply residual layer to the last dimension
        z_last, log_det_last2 = self.actnorm_last(z_last)
        z_last, log_det_last1 = self.residual_block_last(z_last)

        log_det_last = log_det_last1 + log_det_last2

        # Concatenate the transformed parts back together
        z_out = torch.cat([z_first, z_last], dim=1)

        # Sum the log determinants from both transformations
        log_det = log_det_first + log_det_last

        return z_out, log_det

    def inverse(self, z):
        # Split the input into first n-1 dimensions and the last dimension
        z_first, z_last = z[:, :-self.cut], z[:, -self.cut:]

        # Inverse residual layer for the first n-1 dimensions
        z_first, log_det_first1 = self.residual_block_first.inverse(z_first)
        z_first, log_det_first2 = self.actnorm_first.inverse(z_first)

        log_det_first = log_det_first1 + log_det_first2

        # Inverse residual layer for the last dimension
        z_last, log_det_last1 = self.residual_block_last.inverse(z_last)
        z_last, log_det_last2 = self.actnorm_last.inverse(z_last)

        log_det_last = log_det_last1 + log_det_last2

        # Concatenate the transformed parts back together
        z_out = torch.cat([z_first, z_last], dim=1)

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
    print('Flags:')
    for k, v in sorted(vars(flags).items()):
        print("    {}: {}".format(k, v))
    pykeops.test_numpy_bindings()
    pykeops.test_torch_bindings()
    sinkhorn_divergence = SamplesLoss("sinkhorn", blur=flags.blur, p=2)
    
    for dataset in flags.dataset:
        for ver in flags.version:
            print(dataset+'_'+ver)
            # torch.manual_seed(0)

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

                for step in tqdm(range(flags.mlp_steps), desc=f"Training for alpha={alpha}", ncols=100, mininterval=1.0):
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
            for env in train_envs + test_envs + [train_envs_uni]:
                env['trans_labels'] = torch.sign(env['labels']) * torch.log1p(torch.abs(env['labels']))
                env['trans_images'] = torch.sign(env['images']) * torch.log1p(torch.abs(env['images']))

            # whiten data
            trans_universal = []
            for env in train_envs + test_envs:
                env['trans_target'] = torch.column_stack((env['trans_images'], env['trans_labels']))
                trans_universal.append(env['trans_target'])
            train_envs_uni['trans_base'] = torch.column_stack((train_envs_uni['trans_images'], train_envs_uni['trans_labels']))
            trans_universal.append(train_envs_uni['trans_base'])
            trans_universal = torch.concatenate(trans_universal)

            for env in train_envs + test_envs:
                env['whiten_target'] = whiten(env['trans_target'], trans_universal)
            train_envs_uni['whiten_base'] = whiten(train_envs_uni['trans_base'], trans_universal)

            # scale data
            whiten_universal = []
            for env in train_envs + test_envs:
                whiten_universal.append(env['whiten_target'])
            whiten_universal.append(train_envs_uni['whiten_base'])
            whiten_universal = torch.concatenate(whiten_universal)

            for env in train_envs + test_envs:
                env['scaled_target']=scale(env['whiten_target'], whiten_universal)
            train_envs_uni['scaled_base']=scale(train_envs_uni['whiten_base'], whiten_universal)

            input_dim = train_envs[0]['images'].shape[1]
            output_dim = train_envs[0]['labels'].shape[1]

            dim = input_dim + output_dim
            cut = output_dim

            # train a normalizing flow
            base = nf.distributions.base.DiagGaussian(dim)
            flows_branched = [IndependentResidualFlow(dim, cut) for _ in range(flags.nf_num_layer)]
            model_branched = nf.NormalizingFlow(base, flows_branched).to(flags.device)
            _ = model_branched.log_prob(train_envs_uni['scaled_base'].contiguous())

            loss_hist = np.array([])
            optimizer_nf = torch.optim.Adam(model_branched.parameters(), lr=flags.nf_lr)

            print('train a normalizing flow')
            for step in tqdm(range(flags.nf_steps)):
                optimizer_nf.zero_grad()

                for env in train_envs:
                    env['scaled_normalized'] = model_branched.inverse(env['scaled_target'].contiguous())
                    env['W-distance'] = sinkhorn_divergence(env['scaled_normalized'].contiguous(), train_envs_uni['scaled_base'].contiguous())

                loss = torch.stack([e['W-distance'] for e in train_envs]).mean()
                if not (torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer_nf.step()

                nf.utils.update_lipschitz(model_branched, 50)
                loss_hist = np.append(loss_hist, loss.cpu().data.numpy())

            RESULT.append({'loss_hist': loss_hist})
            model_branched.eval()

            # marginal coverage, conditional gap, and set size on calibration data
            tau_list = []
            marginal_coverage_cal_list = []
            conditional_gap_cal_list = []
            size_cal_list = []
            for idx, alpha in enumerate(alpha_array):
                cal_envs_uni['pred_lo'] = MLP_LO[idx](cal_envs_uni['images'])
                cal_envs_uni['pred_hi'] = MLP_HI[idx](cal_envs_uni['images'])
                R_cal = []
                for i in range(len(cal_envs_uni['labels'])):
                    R_cal.append(max(cal_envs_uni['pred_lo'][i] - cal_envs_uni['labels'][i], cal_envs_uni['labels'][i] -
                                    cal_envs_uni['pred_hi'][i]))
                R_cal = torch.tensor(R_cal)
                tau = np.quantile(R_cal, 1 - alpha)
                covered_index = (cal_envs_uni['labels'] >= cal_envs_uni['pred_lo'] - tau) & (
                        cal_envs_uni['labels'] <= cal_envs_uni['pred_hi'] + tau)
                marginal_coverage = (torch.sum(covered_index) / len(cal_envs_uni['labels'])).cpu().detach().numpy()

                conditional_gap = calculate_adaptivity_bins(cal_envs_uni['images'], covered_index, num_bins=flags.bins, alpha=alpha)
                size = torch.mean((cal_envs_uni['pred_hi']+tau) - (cal_envs_uni['pred_lo']-tau)).cpu().detach().numpy()

                tau_list.append(tau)
                marginal_coverage_cal_list.append(marginal_coverage)
                conditional_gap_cal_list.append(conditional_gap)
                size_cal_list.append(size)

            RESULT.append(
                {'data': 'cal', 'alpha': alpha_array, 'tau': tau_list, 'marginal coverage': marginal_coverage_cal_list,
                'conditional gap': conditional_gap_cal_list, 'size': size_cal_list})

            # marginal coverage, conditional gap, and set size on test data
            print('prediction on test envs')
            for env in tqdm(test_envs):
                with torch.no_grad():
                    env['scaled_normalized'] = model_branched.inverse(env['scaled_target']) # feature part independent to label
                    env['whiten_normalized'] = scale_inverse(env['scaled_normalized'], whiten_universal)
                    env['trans_normalized'] = whiten_inverse(env['whiten_normalized'], trans_universal)
                    env['normalized'] = torch.sign(env['trans_normalized']) * torch.expm1(torch.abs(env['trans_normalized']))
                marginal_coverage_test_list = []
                conditional_gap_test_list = []
                conditional_gap_test_normalized_list = []
                size_test_list = []
                wsc_test_list = []
                wscg_test_list = []
                ascg_multi_test_list = []
                for idx, alpha in enumerate(alpha_array):
                    env['pred_lo_normalized'] = MLP_LO[idx](env['normalized'][:, :-1]) # env['normalized'][:, :-1] is normalized feature
                    env['pred_hi_normalized'] = MLP_HI[idx](env['normalized'][:, :-1])
                    tau = tau_list[idx]

                    hi = env['pred_hi_normalized'] + tau
                    lo = env['pred_lo_normalized'] - tau
                    hi_gen = bound_generative(hi, env)
                    lo_gen = bound_generative(lo, env)

                    covered_index = (env['labels'].reshape(-1, 1) >= lo_gen.reshape(-1, 1)) & (
                            env['labels'].reshape(-1, 1) <= hi_gen.reshape(-1, 1))
                    ascg_multi = compute_ascg_multi_gpu(env['images'], covered_index, alpha)
                    wsc = compute_wsc_gpu(env['images'], covered_index)
                    wscg = compute_wscg_gpu(env['images'], covered_index, alpha)
                    marginal_coverage = (torch.sum(covered_index) / len(env['labels'])).cpu().detach().numpy()
                    conditional_gap = calculate_adaptivity_bins(env['images'], covered_index, num_bins=flags.bins, alpha=alpha)
                    conditional_gap_normalized = calculate_adaptivity_bins(env['normalized'][:, :-1], covered_index, num_bins=flags.bins, alpha=alpha)

                    size = torch.mean(hi_gen - lo_gen).cpu().detach().numpy()
                    marginal_coverage_test_list.append(marginal_coverage)
                    conditional_gap_test_list.append(conditional_gap)
                    conditional_gap_test_normalized_list.append(conditional_gap_normalized)

                    size_test_list.append(size)
                    ascg_multi_test_list.append(ascg_multi)
                    wsc_test_list.append(wsc)
                    wscg_test_list.append(wscg)

                RESULT.append(
                    {'data': 'test', 'alpha': alpha_array, 'marginal coverage': marginal_coverage_test_list,
                    'conditional gap': conditional_gap_test_list,
                    'conditional gap normalized': conditional_gap_test_normalized_list,
                    'size': size_test_list, 'wsc': wsc_test_list, 'wscg':wscg_test_list, 'ascg multi': ascg_multi_test_list})

            pickle.dump(RESULT,
                        open("result/" + dataset + "/msdg/bnfcqr_residual_" + ver + "_msdg", "wb"))
