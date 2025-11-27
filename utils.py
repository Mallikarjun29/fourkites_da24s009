import torch
import copy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten_params(params):
    return torch.cat([p.contiguous().view(-1) for p in params])

def flatten_like(params, vec):
    out = []
    offset = 0
    for p in params:
        numel = p.numel()
        out.append(vec[offset:offset + numel].view_as(p))
        offset += numel
    return out

def hessian_vector_product(model, loss, params, vec):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g_flat = flatten_params(grads)
    v_flat = vec
    hvp = torch.autograd.grad(g_flat @ v_flat, params, retain_graph=True)
    hvp_flat = flatten_params(hvp)
    return hvp_flat

def estimate_top_hessian_eig(model, loader, loss_fn,
                             num_iters=20, tol=1e-3):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = loss_fn(logits, y)

    v = torch.randn_like(flatten_params(params))
    v = v / (v.norm() + 1e-12)

    last_lambda = None

    for i in range(num_iters):
        hv = hessian_vector_product(model, loss, params, v)
        hv_norm = hv.norm()
        if hv_norm == 0:
            break
        v = hv / hv_norm
        lam = torch.dot(v, hv).item()
        if last_lambda is not None and abs(lam - last_lambda) < tol:
            break
        last_lambda = lam

    return last_lambda

def compute_flatness(model, loader, loss_fn,
                     epsilon=1e-3, num_samples=10):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    base_state = copy.deepcopy(model.state_dict())

    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        logits = model(x)
        base_loss = loss_fn(logits, y).item()

    deltas = []
    losses = []

    for k in range(num_samples):
        with torch.no_grad():
            current_flat = flatten_params(params)
            noise = torch.randn_like(current_flat)
            noise = epsilon * noise / (noise.norm() + 1e-8)

            perturbed_flat = current_flat + noise
            perturbed_params = flatten_like(params, perturbed_flat)
            for p, p_new in zip(params, perturbed_params):
                p.copy_(p_new)

        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits, y).item()

        deltas.append(k)
        losses.append(loss - base_loss)

        model.load_state_dict(base_state)

    avg_increase = float(np.mean(losses))
    return avg_increase, deltas, losses

def interpolate_models(model_a, model_b, alpha):
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    new_state = {}
    for k in state_a.keys():
        new_state[k] = (1 - alpha) * state_a[k] + alpha * state_b[k]
    model_interp = copy.deepcopy(model_a)
    model_interp.load_state_dict(new_state)
    return model_interp

def mode_connectivity_curve(model_a, model_b, loader, loss_fn,
                            num_points=21):
    alphas = np.linspace(0.0, 1.0, num_points)
    losses = []
    for alpha in alphas:
        model_interp = interpolate_models(model_a, model_b, alpha).to(device)
        loss, _ = evaluate_quick(model_interp, loader, loss_fn)
        losses.append(loss)
    return alphas, losses

# small fast evaluate used by mode connectivity (to avoid circular imports)
def evaluate_quick(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            break  # use single batch for speed
    return total_loss / total_samples, None

def compute_gradient_noise_scale(model, loader, loss_fn,
                                 num_batches=20):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    grads_list = []
    data_iter = iter(loader)

    for i in range(num_batches):
        try:
            x, y = next(data_iter)
        except StopIteration:
            break

        x, y = x.to(device), y.to(device)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        g_flat = flatten_params([p.grad for p in params])
        grads_list.append(g_flat.detach().cpu())

    if len(grads_list) == 0:
        return None

    G = torch.stack(grads_list, dim=0)
    mean_g = G.mean(dim=0)
    sq_norm_mean_g = mean_g.norm().item() ** 2

    norms = G.norm(dim=1)
    mean_sq_norm = (norms ** 2).mean().item()

    denom = mean_sq_norm - sq_norm_mean_g
    if denom <= 0:
        S = float("inf")
    else:
        S = sq_norm_mean_g / (denom + 1e-12)

    return S, sq_norm_mean_g, mean_sq_norm

def compute_2d_loss_surface(model, loader, loss_fn,
                            center_flat, d1, d2,
                            alpha_range, beta_range):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    base_state = copy.deepcopy(model.state_dict())

    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    A = len(alpha_range)
    B = len(beta_range)
    Z = np.zeros((A, B), dtype=np.float32)

    for i, a in enumerate(alpha_range):
        for j, b in enumerate(beta_range):
            with torch.no_grad():
                theta_flat = center_flat + a * d1 + b * d2
                theta_params = flatten_like(params, theta_flat)
                for p, p_new in zip(params, theta_params):
                    p.copy_(p_new)

                logits = model(x)
                loss = loss_fn(logits, y).item()
                Z[i, j] = loss

    model.load_state_dict(base_state)
    return Z
