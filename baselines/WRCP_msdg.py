import argparse
from torch import optim
import pickle
from main.module import *
from tqdm import tqdm


parser = argparse.ArgumentParser(description='WR-CP')
parser.add_argument('--mlp_hidden_dim', type=int, default=64)
parser.add_argument('--bins', type=int, default=10)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=1e-3)  # 0.001
parser.add_argument('--penalty_anneal_iters', type=int, default=1000)
parser.add_argument('--mlp_steps', type=int, default=3000)
parser.add_argument('--dataset', type=list, default=['bike','bio','traffic','states'])
parser.add_argument('--version', type=list, default=['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10'])
parser.add_argument('--w_regularization', type=list, default=[1,5,10,20])

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

def mean_l1(pred, y):
    return nn.functional.l1_loss(pred, y)

if __name__ == "__main__":

    print('Flags:')
    for k, v in sorted(vars(flags).items()):
        print("    {}: {}".format(k, v))
    
    for dataset in flags.dataset:
        for ver in flags.version:
            print(dataset+'_'+ver)
            RESULT = []
            for beta in flags.w_regularization:
                # drop data
                train_envs = pickle.load(open("data/" + dataset + "/msdg/train_" + ver, "rb"))
                test_envs = pickle.load(open("data/" + dataset + "/msdg/test_" + ver, "rb"))
                cal_envs_uni = np.load("data/" + dataset + "/msdg/cal_uni_" + ver + '.npy',
                                    allow_pickle='TRUE').item()

                # init
                alpha_array = np.linspace(0.1, 0.9, 9)

                mlp = MLP(train_envs[0]['images'].shape[1]).cuda()
                optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

                for step in tqdm(range(flags.mlp_steps), ncols=100, mininterval=1.0):
                    cal_envs_uni['pred'] = mlp(cal_envs_uni['images'])
                    cal_envs_uni['res_set'] = torch.abs(torch.subtract(cal_envs_uni['pred'], cal_envs_uni['labels']))

                    for env in train_envs:
                        env['pred'] = mlp(env['images'])
                        env['l1'] = mean_l1(env['pred'], env['labels'])
                        env['res_set'] = torch.abs(torch.subtract(env['pred'], env['labels']))

                    train_l1 = torch.stack([e['l1'] for e in train_envs]).mean()
                    cpdi_penalty = get_cpdi_penalty(train_envs, cal_envs_uni)

                    weight_norm = torch.tensor(0.).cuda()
                    for w in mlp.parameters():
                        weight_norm += w.norm().pow(2)

                    loss = train_l1.clone()

                    loss += flags.l2_regularizer_weight * weight_norm

                    if step >= flags.penalty_anneal_iters:
                        loss /= beta
                    loss += cpdi_penalty

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                

                # marginal coverage, conditional gap, and set size on calibration data
                tau_list = []
                marginal_coverage_cal_list = []
                conditional_gap_cal_list = []
                size_cal_list = []
                for idx, alpha in enumerate(alpha_array):
                    cal_envs_uni['pred'] = mlp(cal_envs_uni['images'])
                    R_cal = torch.abs(cal_envs_uni['pred'] - cal_envs_uni['labels']).cpu().detach().numpy().flatten()
                    tau = np.quantile(R_cal, 1 - alpha)
                    covered_index = (cal_envs_uni['labels'] >= cal_envs_uni['pred'] - tau) & (
                            cal_envs_uni['labels'] <= cal_envs_uni['pred'] + tau)
                    marginal_coverage = (torch.sum(covered_index) / len(cal_envs_uni['labels'])).cpu().detach().numpy()

                    conditional_gap = calculate_adaptivity_bins(cal_envs_uni['images'], covered_index, num_bins=flags.bins, alpha=alpha)
                    size = 2*tau

                    tau_list.append(tau)
                    marginal_coverage_cal_list.append(marginal_coverage)
                    conditional_gap_cal_list.append(conditional_gap)
                    size_cal_list.append(size)

                RESULT.append(
                    {'data': 'cal', 'alpha': alpha_array, 'tau': tau_list, 'marginal coverage': marginal_coverage_cal_list,
                    'conditional gap': conditional_gap_cal_list, 'size': size_cal_list, 'beta': beta})


                # marginal coverage, conditional gap, and set size on test data
                for env in test_envs:
                    env['pred'] = mlp(env['images'])

                for env in test_envs:
                    marginal_coverage_test_list = []
                    conditional_gap_test_list = []
                    size_test_list = []
                    wsc_test_list = []
                    wsc_plus_test_list = []
                    wscg_test_list = []
                    ascg_multi_test_list = []
                    weights = env['weights'].cpu().detach().numpy()
                    for idx, alpha in enumerate(alpha_array):
                        tau = weighted_quantile(R_cal, 1 - alpha, weights)
                        covered_index = (env['labels'] >= env['pred'] - tau) & (env['labels'] <= env['pred'] + tau)
                        wsc = compute_wsc_gpu(env['images'], covered_index)
                        wsc_plus = compute_wsc_plus_gpu(env['images'], covered_index)
                        wscg = compute_wscg_gpu(env['images'], covered_index, alpha)
                        ascg_multi = compute_ascg_multi_gpu(env['images'], covered_index, alpha)
                        marginal_coverage = (torch.sum(covered_index) / len(env['labels'])).cpu().detach().numpy()
                        conditional_gap = calculate_adaptivity_bins(env['images'], covered_index, num_bins=flags.bins, alpha=alpha)
                        size = 2 * tau
                        marginal_coverage_test_list.append(marginal_coverage)
                        conditional_gap_test_list.append(conditional_gap)
                        size_test_list.append(size)
                        wsc_test_list.append(wsc)
                        wsc_plus_test_list.append(wsc_plus)
                        wscg_test_list.append(wscg)
                        ascg_multi_test_list.append(ascg_multi)

                    RESULT.append(
                        {'data': 'test', 'alpha': alpha_array, 'marginal coverage': marginal_coverage_test_list,
                        'conditional gap': conditional_gap_test_list, 'size': size_test_list, 'beta': beta, 
                        'wsc':wsc_test_list, 'wscg': wscg_test_list, 'ascg multi': ascg_multi_test_list, 'wsc plus': wsc_plus_test_list})


                pickle.dump(RESULT,
                            open("result/" + dataset + "/msdg/wrcp_"+ ver + "_msdg", "wb"))
