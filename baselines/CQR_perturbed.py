import argparse
import torch.optim as optim
from tqdm import tqdm
import pickle
from main.module import *


parser = argparse.ArgumentParser(description='CQR')
parser.add_argument('--mlp_hidden_dim', type=int, default=64)
parser.add_argument('--bins', type=int, default=10)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--mlp_lr', type=float, default=1e-3)  # 0.001
parser.add_argument('--mlp_steps', type=int, default=3000)
parser.add_argument('--dataset', type=list, default=['bike','bio','traffic','states'])
parser.add_argument('--version', type=list, default=['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10'])
flags = parser.parse_args()


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

    for dataset in flags.dataset:
        for ver in flags.version:
            print(dataset+'_'+ver)
            # drop data
            train_env = np.load("data/" + dataset + "/perturbed/train_" + ver + '.npy', allow_pickle='TRUE').item()
            cal_env = np.load("data/" + dataset + "/perturbed/cal_" + ver + '.npy', allow_pickle='TRUE').item()
            test_env = np.load("data/" + dataset + "/perturbed/test_" + ver + '.npy', allow_pickle='TRUE').item()

            test_envs_perturbed = []
            for _ in range(100):
                scale_x = 1.0
                scale_y = np.random.uniform(1.0, 1.5)
                env = perturb_scale(test_env, scale_x, scale_y)
                test_envs_perturbed.append(env)

            for env in [train_env, test_env, cal_env]+test_envs_perturbed:
                env['images'] = env['images'].cuda()
                env['labels'] = env['labels'].cuda()

            RESULT = []
            alpha_array = np.around(np.linspace(0.1, 0.9, 9), 2)
            # alpha_array = [0.1]
            # train empirical risk minimization model
            print('train empirical risk minimization model')
            MLP_LO = []
            MLP_HI = []
            for alpha in alpha_array:
                mlp_lo = MLP(train_env['images'].shape[1]).cuda()
                optimizer_mlp_lo = optim.Adam(mlp_lo.parameters(), lr=flags.mlp_lr)
                mlp_hi = MLP(train_env['images'].shape[1]).cuda()
                optimizer_mlp_hi = optim.Adam(mlp_hi.parameters(), lr=flags.mlp_lr)

                for step in tqdm(range(flags.mlp_steps), desc=f"Training for alpha={alpha}", ncols=100, mininterval=1.0):
                    train_env['pred_lo'] = mlp_lo(train_env['images'])
                    train_env['pinball_lo'] = pinball_loss(train_env['pred_lo'], train_env['labels'], alpha=alpha / 2)
                    train_env['pred_hi'] = mlp_hi(train_env['images'])
                    train_env['pinball_hi'] = pinball_loss(train_env['pred_hi'], train_env['labels'], alpha=1 - alpha / 2)
                    loss_lo = train_env['pinball_lo']
                    loss_hi = train_env['pinball_hi']

                    weight_norm_lo = torch.tensor(0.).cuda()
                    for w in mlp_lo.parameters():
                        weight_norm_lo += w.norm().pow(2)
                    loss_lo += flags.l2_regularizer_weight * weight_norm_lo

                    weight_norm_hi = torch.tensor(0.).cuda()
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

            # marginal coverage, conditional gap, and set size on calibration data
            tau_list = []
            marginal_coverage_cal_list = []
            conditional_gap_cal_list = []
            size_cal_list = []
            for idx, alpha in enumerate(alpha_array):
                cal_env['pred_lo'] = MLP_LO[idx](cal_env['images'])
                cal_env['pred_hi'] = MLP_HI[idx](cal_env['images'])
                R_cal = []
                for i in range(len(cal_env['labels'])):
                    R_cal.append(max(cal_env['pred_lo'][i] - cal_env['labels'][i], cal_env['labels'][i] -
                                    cal_env['pred_hi'][i]))
                R_cal = torch.tensor(R_cal)
                tau = np.quantile(R_cal, 1 - alpha)
                covered_index = (cal_env['labels'] >= cal_env['pred_lo'] - tau) & (
                        cal_env['labels'] <= cal_env['pred_hi'] + tau)
                marginal_coverage = (torch.sum(covered_index) / len(cal_env['labels'])).cpu().detach().numpy()

                conditional_gap = calculate_adaptivity_bins(cal_env['images'], covered_index, num_bins=flags.bins, alpha=alpha)
                size = torch.mean((cal_env['pred_hi'] + tau) - (cal_env['pred_lo']-tau)).cpu().detach().numpy()

                tau_list.append(tau)
                marginal_coverage_cal_list.append(marginal_coverage)
                conditional_gap_cal_list.append(conditional_gap)
                size_cal_list.append(size)

            RESULT.append(
                {'data': 'cal', 'alpha': alpha_array, 'tau': tau_list, 'marginal coverage': marginal_coverage_cal_list,
                'conditional gap': conditional_gap_cal_list, 'size': size_cal_list})

            # marginal coverage, conditional gap, and set size on test data

            for env in tqdm(test_envs_perturbed):
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
                    conditional_gap = calculate_adaptivity_bins(env['images'], covered_index, num_bins=flags.bins, alpha=alpha)
                    wsc = compute_wsc_gpu(env['images'], covered_index)
                    wsc_plus = compute_wsc_plus_gpu(env['images'], covered_index)
                    wscg = compute_wscg_gpu(env['images'], covered_index, alpha)
                    size = torch.mean((env['pred_hi']+tau) - (env['pred_lo']-tau)).cpu().detach().numpy()
                    ascg_multi = compute_ascg_multi_gpu(env['images'], covered_index, alpha)
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

            pickle.dump(RESULT,
                        open("result/" + dataset + "/perturbed/cqr_" + ver + "_perturbed", "wb"))
