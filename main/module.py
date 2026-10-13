import numpy as np
import torch
from torch import nn

def whiten(x, uni):
    with torch.no_grad():
        x_copy = x.clone()
        x_copy -= uni.mean(dim=0)
        x_copy /= uni.std(dim=0)
    return torch.nan_to_num(x_copy)

def whiten_inverse(x, uni):
    with torch.no_grad():
        x_copy = x.clone()
        x_copy *= uni.std(dim=0)
        x_copy += uni.mean(dim=0)
    return torch.nan_to_num(x_copy)

def scale(x,uni):
    with torch.no_grad():
        x_copy = x.clone()
        min_val = uni.min(dim=0).values
        max_val = uni.max(dim=0).values
        x_copy = (x_copy - min_val) / (max_val - min_val)
    return torch.nan_to_num(x_copy)

def scale_inverse(x,uni):
    with torch.no_grad():
        x_copy = x.clone()
        min_val = uni.min(dim=0).values
        max_val = uni.max(dim=0).values
        x_copy = x_copy*(max_val - min_val) + min_val
    return torch.nan_to_num(x_copy) 

def pinball_loss(pred, y, alpha=0.5):
    # Compute the difference between prediction and true value
    diff = y - pred

    # Implement the pinball loss formula
    loss = torch.maximum(alpha * diff, (alpha - 1) * diff)

    # Return the mean loss
    return torch.mean(loss)




def calculate_adaptivity_bins(X, covered_index, num_bins, alpha):
    # when num_bins is 0, the function returns marginal coverage gap.
    X, covered_index = X.cpu().detach().numpy(), covered_index.cpu().detach().numpy()

    # Split data into covered and uncovered based on the indices
    X_covered = X[covered_index.flatten(), :]

    # Generate bin edges for each feature dimension
    bin_edges = [np.percentile(X[:, i], np.linspace(0, 100, num_bins + 1)) for i in range(X.shape[1])]

    counts_between_bins = {}
    for i in range(X.shape[1]):
        if not (np.all(np.diff(bin_edges[i]) >= 0) or np.all(np.diff(bin_edges[i]) <= 0)):
            print(f"Bin edges at index {i} are not monotonically increasing or decreasing. Skipping.")
            break
        
        bin_indices = np.digitize(X[:, i], bin_edges[i]) - 1
        bin_counts = np.zeros(num_bins)
        for idx in bin_indices:
            if idx < num_bins:
                bin_counts[idx] += 1
        counts_between_bins[i] = bin_counts

    covered_between_bins = {}
    for i in range(X.shape[1]):
        if not (np.all(np.diff(bin_edges[i]) >= 0) or np.all(np.diff(bin_edges[i]) <= 0)):
            print(f"Bin edges at index {i} are not monotonically increasing or decreasing. Skipping.")
            break

        bin_indices = np.digitize(X_covered[:, i], bin_edges[i]) - 1
        bin_counts = np.zeros(num_bins)
        for idx in bin_indices:
            if idx < num_bins:
                bin_counts[idx] += 1
        covered_between_bins[i] = bin_counts

    counts_array = np.array([counts_between_bins[i] for i in range(len(counts_between_bins))])
    counts_array = np.where(counts_array == 0, np.nan, counts_array)
    covered_array = np.array([covered_between_bins[i] for i in range(len(covered_between_bins))])

    conditional_gap = np.nanmean(abs(covered_array/counts_array - (1-alpha)))

    return conditional_gap

def mean_l1(pred, y):
    return nn.functional.l1_loss(pred, y)

def Wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
    """
    A modified version of scipy.stats.wasserstein_distance

    """
    u_sorter = torch.argsort(u_values)
    v_sorter = torch.argsort(v_values)
    all_values = torch.concatenate((u_values, v_values))
    all_values, _ = torch.sort(all_values)

    deltas = torch.diff(all_values)
    u_cdf_indices = torch.searchsorted(u_values[u_sorter], all_values[:-1], side='right')
    v_cdf_indices = torch.searchsorted(v_values[v_sorter], all_values[:-1], side='right')

    zero = torch.tensor([0]).to('cuda:0')
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.shape[0]
    else:
        u_sorted_cumweights = torch.concatenate((zero, torch.cumsum(u_weights[u_sorter], dim=0)))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.shape[0]
    else:
        v_sorted_cumweights = torch.concatenate((zero, torch.cumsum(v_weights[v_sorter], dim=0)))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    pos = torch.searchsorted(u_cdf, torch.tensor([1.0]).to('cuda:0')).item()

    u_cdf_scaled = u_cdf[0:pos]
    v_cdf_scaled = v_cdf[0:pos]
    deltas_scaled = deltas[0:pos]
    W = torch.sum(torch.multiply(torch.abs(u_cdf_scaled - v_cdf_scaled), deltas_scaled))
    return W


def get_cpdi_penalty(train_envs, cal_envs_uni):
    # require (n,) format, so flatten()
    cwn_losses = torch.stack(
        [Wasserstein_distance(cal_envs_uni['res_set'].flatten(), e['res_set'].flatten(),
                                         u_weights=e['weights'], v_weights=None) for e
         in train_envs])
    penalty = torch.mean(cwn_losses)  # must include mean as a loss, only mean is better
    return penalty


def weighted_quantile(values, quantiles, weight, values_sorted=False):
    """ Very close to np.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]! """
    values = np.array(values)
    quantiles = np.array(quantiles)
    weight = np.array(weight)
    sorter = np.argsort(values)
    if not values_sorted:
        values = values[sorter]
        weight = weight[sorter]
    weighted_quantiles = np.cumsum(weight) - 0.5 * weight
    weighted_quantiles /= np.sum(weight)
    return np.interp(quantiles, weighted_quantiles, values)


def compute_wsc_gpu(
    X,                     # (n, p) torch.float32 CUDA tensor
    covered_index,         # (n,) torch.bool or float CUDA tensor
    delta=0.1,
    num_v=1000,
    num_grid=50,
    seed=0
):
    """
    Fully vectorized, robust GPU version of Worst-Slice Coverage (WSC).
    """
    device = X.device
    torch.manual_seed(seed)

    n, p = X.shape
    covered = covered_index.float()   # convert to 0/1 float

    # ---- 1. Sample random directions on GPU ----
    V = torch.randn(num_v, p, device=device)
    V = V / (V.norm(dim=1, keepdim=True) + 1e-12)

    # ---- 2. Compute projections ----
    proj = V @ X.t()  # (num_v, n)

    # ---- 3. Sort projections & reorder coverage ----
    proj_sorted, idx = torch.sort(proj, dim=1)
    covered_sorted = covered[idx]      # (num_v, n)

    # ---- 4. Cumulative sums for coverage ----
    c_cov = torch.cumsum(covered_sorted, dim=1)  # (num_v, n)

    # ---- 5. Compute quantile boundaries ----
    q = torch.linspace(0, 1, num_grid, device=device)
    grid_vals = torch.quantile(proj_sorted, q, dim=1).transpose(0, 1).contiguous()  # (num_v, num_grid)

    # ---- 6. Map grid_vals → indices using searchsorted ----
    left_idx  = torch.searchsorted(proj_sorted, grid_vals, right=False)  # (num_v, num_grid)
    right_idx = torch.searchsorted(proj_sorted, grid_vals, right=True)   # (num_v, num_grid)

    # ---- 7. Expand indices for all slab pairs ----
    g = num_grid
    ai_idx = left_idx.unsqueeze(2).expand(-1, g, g)   # (num_v, g, g)
    bi_idx = right_idx.unsqueeze(1).expand(-1, g, g)  # (num_v, g, g)

    # Masks and slab sizes
    valid_pairs = bi_idx > ai_idx
    slab_sizes = bi_idx - ai_idx
    valid = valid_pairs & (slab_sizes.float() / n >= delta)

    if not valid.any():
        return 1.0

    # ---- 8. Compute coverage using flattened advanced indexing ----
    num_v = X.shape[0] if len(X.shape) == 1 else ai_idx.shape[0]

    # Flatten everything
    ai_flat = ai_idx.reshape(-1)       # (num_v * g * g,)
    bi_flat = bi_idx.reshape(-1)
    v_flat  = torch.arange(num_v, device=device).unsqueeze(1).expand(-1, g*g).reshape(-1)

    # Safe indexing
    cov_r_flat = c_cov[v_flat, bi_flat.clamp(max=n-1)]
    cov_l_flat = c_cov[v_flat, (ai_flat-1).clamp(min=0)]
    cov_l_flat[ai_flat == 0] = 0

    # Reshape back
    cov_r = cov_r_flat.view(num_v, g, g)
    cov_l = cov_l_flat.view(num_v, g, g)

    # ---- 9. Compute coverage and WSC ----
    coverage = (cov_r - cov_l).float() / slab_sizes.float().clamp(min=1)
    coverage = coverage.masked_fill(~valid, float("inf"))

    wsc = coverage.min().item()
    return wsc

def compute_wscg_gpu(
    X,                     # (n, p) torch.float32 CUDA tensor
    covered_index,         # (n,) torch.bool or float CUDA tensor
    alpha,
    delta=0.1,
    num_v=1000,
    num_grid=50,
    seed=0
):
    """
    Fully vectorized, robust GPU version of Worst-Slice Coverage (WSC).
    """
    device = X.device
    torch.manual_seed(seed)

    n, p = X.shape
    covered = covered_index.float()   # convert to 0/1 float

    # ---- 1. Sample random directions on GPU ----
    V = torch.randn(num_v, p, device=device)
    V = V / (V.norm(dim=1, keepdim=True) + 1e-12)

    # ---- 2. Compute projections ----
    proj = V @ X.t()  # (num_v, n)

    # ---- 3. Sort projections & reorder coverage ----
    proj_sorted, idx = torch.sort(proj, dim=1)
    covered_sorted = covered[idx]      # (num_v, n)

    # ---- 4. Cumulative sums for coverage ----
    c_cov = torch.cumsum(covered_sorted, dim=1)  # (num_v, n)

    # ---- 5. Compute quantile boundaries ----
    q = torch.linspace(0, 1, num_grid, device=device)
    grid_vals = torch.quantile(proj_sorted, q, dim=1).transpose(0, 1).contiguous()  # (num_v, num_grid)

    # ---- 6. Map grid_vals → indices using searchsorted ----
    left_idx  = torch.searchsorted(proj_sorted, grid_vals, right=False)  # (num_v, num_grid)
    right_idx = torch.searchsorted(proj_sorted, grid_vals, right=True)   # (num_v, num_grid)

    # ---- 7. Expand indices for all slab pairs ----
    g = num_grid
    ai_idx = left_idx.unsqueeze(2).expand(-1, g, g)   # (num_v, g, g)
    bi_idx = right_idx.unsqueeze(1).expand(-1, g, g)  # (num_v, g, g)

    # Masks and slab sizes
    valid_pairs = bi_idx > ai_idx
    slab_sizes = bi_idx - ai_idx
    valid = valid_pairs & (slab_sizes.float() / n >= delta)

    if not valid.any():
        return 1.0

    # ---- 8. Compute coverage using flattened advanced indexing ----
    num_v = X.shape[0] if len(X.shape) == 1 else ai_idx.shape[0]

    # Flatten everything
    ai_flat = ai_idx.reshape(-1)       # (num_v * g * g,)
    bi_flat = bi_idx.reshape(-1)
    v_flat  = torch.arange(num_v, device=device).unsqueeze(1).expand(-1, g*g).reshape(-1)

    # Safe indexing
    cov_r_flat = c_cov[v_flat, bi_flat.clamp(max=n-1)]
    cov_l_flat = c_cov[v_flat, (ai_flat-1).clamp(min=0)]
    cov_l_flat[ai_flat == 0] = 0

    # Reshape back
    cov_r = cov_r_flat.view(num_v, g, g)
    cov_l = cov_l_flat.view(num_v, g, g)

    # ---- 9. Compute coverage and Worst case coverage gap ----
    coverage = (cov_r - cov_l).float() / slab_sizes.float().clamp(min=1)

    # Mask invalid slabs *before* computing gap
    valid_cov = coverage.masked_fill(~valid, float("nan"))

    coverage_gap = (valid_cov - (1-alpha)).abs()

    wscg = coverage_gap[~torch.isnan(coverage_gap)].max().item()

    return wscg


def compute_ascg_multi_gpu(
    X,                     # (n, p) torch.float32 CUDA tensor
    covered_index,         # (n,) torch.bool or float CUDA tensor
    alpha,
    delta=0.1,
    num_v=1000,
    num_grid=50,
    seed=0
):
    """
    Fully vectorized, robust GPU version of ASCG.
    """
    device = X.device
    torch.manual_seed(seed)

    n, p = X.shape
    covered = covered_index.float()   # convert to 0/1 float

    # ---- 1. Sample random directions on GPU ----
    V = torch.randn(num_v, p, device=device)
    V = V / (V.norm(dim=1, keepdim=True) + 1e-12)

    # ---- 2. Compute projections ----
    proj = V @ X.t()  # (num_v, n)

    # ---- 3. Sort projections & reorder coverage ----
    proj_sorted, idx = torch.sort(proj, dim=1)
    covered_sorted = covered[idx]      # (num_v, n)

    # ---- 4. Cumulative sums for coverage ----
    c_cov = torch.cumsum(covered_sorted, dim=1)  # (num_v, n)

    # ---- 5. Compute quantile boundaries ----
    q = torch.linspace(0, 1, num_grid, device=device)
    grid_vals = torch.quantile(proj_sorted, q, dim=1).transpose(0, 1).contiguous()  # (num_v, num_grid)

    # ---- 6. Map grid_vals → indices using searchsorted ----
    left_idx  = torch.searchsorted(proj_sorted, grid_vals, right=False)  # (num_v, num_grid)
    right_idx = torch.searchsorted(proj_sorted, grid_vals, right=True)   # (num_v, num_grid)

    # ---- 7. Expand indices for all slab pairs ----
    g = num_grid
    ai_idx = left_idx.unsqueeze(2).expand(-1, g, g)   # (num_v, g, g)
    bi_idx = right_idx.unsqueeze(1).expand(-1, g, g)  # (num_v, g, g)

    # Masks and slab sizes
    valid_pairs = bi_idx > ai_idx
    slab_sizes = bi_idx - ai_idx
    valid = valid_pairs & (slab_sizes.float() / n >= delta)

    if not valid.any():
        return 1.0

    # ---- 8. Compute coverage using flattened advanced indexing ----
    num_v = X.shape[0] if len(X.shape) == 1 else ai_idx.shape[0]

    # Flatten everything
    ai_flat = ai_idx.reshape(-1)       # (num_v * g * g,)
    bi_flat = bi_idx.reshape(-1)
    v_flat  = torch.arange(num_v, device=device).unsqueeze(1).expand(-1, g*g).reshape(-1)

    # Safe indexing
    cov_r_flat = c_cov[v_flat, bi_flat.clamp(max=n-1)]
    cov_l_flat = c_cov[v_flat, (ai_flat-1).clamp(min=0)]
    cov_l_flat[ai_flat == 0] = 0

    # Reshape back
    cov_r = cov_r_flat.view(num_v, g, g)
    cov_l = cov_l_flat.view(num_v, g, g)


     # ---- 9. Compute ASCG (sample-weighted) ----
    coverage = (cov_r - cov_l).float() / slab_sizes.float().clamp(min=1)

    # Mask invalid slabs before computing gaps
    valid_cov = coverage.masked_fill(~valid, float("nan"))

    # Compute absolute gaps
    coverage_gap = (valid_cov - (1-alpha)).abs()  # (num_v, g, g)

    # ---- Sample weights: number of samples in each slab ----
    slab_weights = slab_sizes.float().masked_fill(~valid, 0.0)  # (num_v, g, g)

    total_weight = slab_weights.sum()
    if total_weight == 0:
        return 1.0

    # ---- Weighted average ----
    ascg = (coverage_gap * slab_weights).nansum().item() / total_weight.item()

    # # ---- 9. Compute coverage and ASCG ----
    # coverage = (cov_r - cov_l).float() / slab_sizes.float().clamp(min=1)

    # # Mask invalid slabs *before* computing gap
    # valid_cov = coverage.masked_fill(~valid, float("nan"))

    # coverage_gap = (valid_cov - (1-alpha)).abs()

    # ascg = coverage_gap[~torch.isnan(coverage_gap)].mean().item()

    return ascg

def compute_wsc_plus_gpu(
    X,                     # (n, p)
    covered_index,         # (n,)
    delta=0.1,
    num_v=1000,
    num_grid=50,
    seed=0
):
    """
    Worst-Slice Coverage Plus (quadratic slices):
    Pr(a <= x^T W x + v^T x <= b) >= delta
    """

    device = X.device
    torch.manual_seed(seed)

    n, p = X.shape
    covered = covered_index.float()

    # ---- 1. Sample random linear + quadratic directions ----
    V = torch.randn(num_v, p, device=device)
    V = V / (V.norm(dim=1, keepdim=True) + 1e-12)

    # Quadratic matrices
    W = torch.randn(num_v, p, p, device=device)

    # Make symmetric
    W = 0.5 * (W + W.transpose(-1, -2))

    # Optional: normalize to control magnitude
    W = W / (W.norm(dim=(1, 2), keepdim=True) + 1e-12)

    # ---- 2. Compute scores s(x) = x^T W x + v^T x ----

    # Linear term
    lin = V @ X.t()   # (num_v, n)

    # Quadratic term: (num_v, n)
    # x^T W x = sum_i sum_j x_i W_ij x_j
    quad = torch.einsum('bi,vij,bj->vb', X, W, X)

    scores = lin + quad   # (num_v, n)

    # ---- 3. Sort scores & reorder coverage ----
    scores_sorted, idx = torch.sort(scores, dim=1)
    covered_sorted = covered[idx]

    # ---- 4. Cumulative sums ----
    c_cov = torch.cumsum(covered_sorted, dim=1)

    # ---- 5. Quantile grid ----
    q = torch.linspace(0, 1, num_grid, device=device)
    grid_vals = torch.quantile(scores_sorted, q, dim=1).transpose(0, 1).contiguous()

    # ---- 6. Map to indices ----
    left_idx  = torch.searchsorted(scores_sorted, grid_vals, right=False)
    right_idx = torch.searchsorted(scores_sorted, grid_vals, right=True)

    # ---- 7. Build slabs ----
    g = num_grid
    ai_idx = left_idx.unsqueeze(2).expand(-1, g, g)
    bi_idx = right_idx.unsqueeze(1).expand(-1, g, g)

    valid_pairs = bi_idx > ai_idx
    slab_sizes = bi_idx - ai_idx
    valid = valid_pairs & (slab_sizes.float() / n >= delta)

    if not valid.any():
        return 1.0

    # ---- 8. Flatten indexing ----
    ai_flat = ai_idx.reshape(-1)
    bi_flat = bi_idx.reshape(-1)
    v_flat  = torch.arange(num_v, device=device).unsqueeze(1).expand(-1, g*g).reshape(-1)

    cov_r_flat = c_cov[v_flat, bi_flat.clamp(max=n-1)]
    cov_l_flat = c_cov[v_flat, (ai_flat-1).clamp(min=0)]
    cov_l_flat[ai_flat == 0] = 0

    cov_r = cov_r_flat.view(num_v, g, g)
    cov_l = cov_l_flat.view(num_v, g, g)

    # ---- 9. Coverage ----
    coverage = (cov_r - cov_l).float() / slab_sizes.float().clamp(min=1)
    coverage = coverage.masked_fill(~valid, float("inf"))

    return coverage.min().item()


def perturb_scale(env, scale_x=1.0, scale_y=1.0):
    new_env = {}

    # IMPORTANT: clone to avoid modifying original data
    new_env['images'] = (env['images'] * scale_x).clone()
    new_env['labels'] = (env['labels'] * scale_y).clone()

    # keep tags if exist
    if 'tags' in env:
        new_env['tags'] = env['tags'].clone()

    return new_env