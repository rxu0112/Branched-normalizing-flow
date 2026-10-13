import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from main.module import (
    calculate_adaptivity_bins,
    compute_ascg_multi_gpu,
    compute_wsc_gpu,
    compute_wsc_plus_gpu,
    compute_wscg_gpu,
    pinball_loss,
)


parser = argparse.ArgumentParser(description='CQR with signed-log transformation and scaling')
parser.add_argument('--mlp_hidden_dim', type=int, default=64)
parser.add_argument('--bins', type=int, default=10)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--mlp_lr', type=float, default=1e-3)
parser.add_argument('--mlp_steps', type=int, default=3000)
parser.add_argument('--dataset', type=str, default=['bike','bio','traffic','states'])
parser.add_argument('--version', type=str, default=['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10'])
parser.add_argument('--device', type=str, default='cuda:0')
flags = parser.parse_args()


class SignedLogScaler:
    """Training-fitted signed-log, standardization, and min-max transform."""

    def __init__(self, training_values):
        transformed = self.signed_log(training_values)
        self.mean = transformed.mean(dim=0)
        std = transformed.std(dim=0)
        self.std = torch.where(std > 0, std, torch.ones_like(std))
        standardized = (transformed - self.mean) / self.std
        self.minimum = standardized.min(dim=0).values
        span = standardized.max(dim=0).values - self.minimum
        self.span = torch.where(span > 0, span, torch.ones_like(span))

    @staticmethod
    def signed_log(values):
        return torch.sign(values) * torch.log1p(torch.abs(values))

    def transform(self, values):
        standardized = (self.signed_log(values) - self.mean) / self.std
        return (standardized - self.minimum) / self.span

    def inverse(self, values):
        # Use double precision when converting bounds back to label units.
        transformed = ((values.double() * self.span.double() + self.minimum.double())
                       * self.std.double() + self.mean.double())
        return torch.sign(transformed) * torch.expm1(torch.abs(transformed))


def mean_interval_width(pred_lo, pred_hi, tau, label_scaler):
    """Average interval width after mapping bounds to original label units."""
    lower = label_scaler.inverse(pred_lo - tau)
    upper = label_scaler.inverse(pred_hi + tau)
    return (upper - lower).mean().to(pred_lo.dtype).cpu().numpy()


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
            lin1, nn.Tanh(),
            nn.Dropout(),
            lin2, nn.Tanh(),
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

    for dataset in flags.dataset:
        for ver in flags.version:
            print(f'{dataset}_{ver}')
            data_dir = Path('data') / dataset / 'iid'
            with (data_dir / f'train_{ver}').open('rb') as stream:
                train_envs = pickle.load(stream)
            with (data_dir / f'test_{ver}').open('rb') as stream:
                test_envs = pickle.load(stream)
            cal_envs_uni = np.load(
                data_dir / f'cal_uni_{ver}.npy', allow_pickle=True).item()

            all_envs = train_envs + [cal_envs_uni] + test_envs
            for env in all_envs:
                env['raw_images'] = torch.as_tensor(
                    env['images'], dtype=torch.float32, device=flags.device).detach()
                env['labels'] = torch.as_tensor(
                    env['labels'], dtype=torch.float32, device=flags.device).detach()

            image_scaler = SignedLogScaler(torch.cat([env['raw_images'] for env in train_envs]))
            label_scaler = SignedLogScaler(torch.cat([env['labels'] for env in train_envs]))
            for env in all_envs:
                env['images'] = image_scaler.transform(env['raw_images'])
                env['labels'] = label_scaler.transform(env['labels'])

            RESULT = []
            alpha_array = np.array([0.1], dtype=float)
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

            with torch.no_grad():
                # marginal coverage, conditional gap, and set size on calibration data
                tau_list = []
                marginal_coverage_cal_list = []
                conditional_gap_cal_list = []
                size_cal_list = []
                for idx, alpha in enumerate(alpha_array):
                    cal_envs_uni['pred_lo'] = MLP_LO[idx](cal_envs_uni['images'])
                    cal_envs_uni['pred_hi'] = MLP_HI[idx](cal_envs_uni['images'])

                    # Vectorized calibration scores; retain the original quantile rule.
                    R_cal = torch.maximum(
                        cal_envs_uni['pred_lo'] - cal_envs_uni['labels'],
                        cal_envs_uni['labels'] - cal_envs_uni['pred_hi'],
                    ).reshape(-1).cpu().numpy()
                    tau = np.quantile(R_cal, 1 - alpha)
                    covered_index = (cal_envs_uni['labels'] >= cal_envs_uni['pred_lo'] - tau) & (
                            cal_envs_uni['labels'] <= cal_envs_uni['pred_hi'] + tau)
                    marginal_coverage = (torch.sum(covered_index) / len(cal_envs_uni['labels'])).cpu().detach().numpy()
                    conditional_gap = calculate_adaptivity_bins(cal_envs_uni['raw_images'], covered_index, num_bins=flags.bins, alpha=alpha)
                    size = mean_interval_width(cal_envs_uni['pred_lo'], cal_envs_uni['pred_hi'], tau, label_scaler)

                    tau_list.append(tau)
                    marginal_coverage_cal_list.append(marginal_coverage)
                    conditional_gap_cal_list.append(conditional_gap)
                    size_cal_list.append(size)

                RESULT.append(
                    {'data': 'cal', 'alpha': alpha_array, 'tau': tau_list, 'marginal coverage': marginal_coverage_cal_list,
                    'conditional gap': conditional_gap_cal_list, 'size': size_cal_list})

                # marginal coverage, conditional gap, and set size on test data

                for env in tqdm(test_envs):
                    marginal_coverage_test_list = []
                    conditional_gap_test_list = []
                    size_test_list = []
                    wsc_test_list = []
                    wsc_plus_test_list = []
                    wscg_test_list = []
                    ascg_multi_test_list = []
                    for idx, alpha in enumerate(alpha_array):
                        env['pred_lo'] = MLP_LO[idx](env['images'])
                        env['pred_hi'] = MLP_HI[idx](env['images'])
                        tau = tau_list[idx]
                        covered_index = (env['labels'] >= env['pred_lo'] - tau) & (
                                env['labels'] <= env['pred_hi'] + tau)
                        marginal_coverage = (torch.sum(covered_index) / len(env['labels'])).cpu().detach().numpy()
                        conditional_gap = calculate_adaptivity_bins(env['raw_images'], covered_index, num_bins=flags.bins, alpha=alpha)
                        wsc = compute_wsc_gpu(env['raw_images'], covered_index)
                        wsc_plus = compute_wsc_plus_gpu(env['raw_images'], covered_index)
                        wscg = compute_wscg_gpu(env['raw_images'], covered_index, alpha)
                        size = mean_interval_width(env['pred_lo'], env['pred_hi'], tau, label_scaler)
                        ascg_multi = compute_ascg_multi_gpu(env['raw_images'], covered_index, alpha)
                        marginal_coverage_test_list.append(marginal_coverage)
                        conditional_gap_test_list.append(conditional_gap)
                        wscg_test_list.append(wscg)
                        wsc_test_list.append(wsc)
                        wsc_plus_test_list.append(wsc_plus)
                        ascg_multi_test_list.append(ascg_multi)
                        size_test_list.append(size)
                    RESULT.append(
                        {'data': 'test', 'alpha': alpha_array, 'marginal coverage': marginal_coverage_test_list,
                        'conditional gap': conditional_gap_test_list, 'size': size_test_list,
                        'wscg': wscg_test_list, 'wsc': wsc_test_list, 'ascg multi': ascg_multi_test_list, 'wsc plus': wsc_plus_test_list})

            result_dir = Path('result') / dataset / 'iid'
            result_dir.mkdir(parents=True, exist_ok=True)
            with (result_dir / f'cqr_{ver}_iid').open('wb') as stream:
                pickle.dump(RESULT, stream)
