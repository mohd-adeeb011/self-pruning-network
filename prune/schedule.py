import numpy as np
from engine import cross_entropy
from .hessian import model_scores


def cubic_sparsity(epoch, start, end, target):
    if end <= start or not 0 <= target < 1:
        raise ValueError("Invalid pruning schedule.")
    progress = np.clip((epoch-start)/(end-start), 0.0, 1.0)
    return float(target * (1 - (1-progress)**3))


def prune_to(model, optimizer, sparsity, method, x=None, y=None,
             step=1e-4, damping=1e-3):
    if not 0 <= sparsity < 1 or method not in ('magnitude', 'hessian'):
        raise ValueError("Invalid sparsity or pruning method.")
    weights = model.weights()
    masks = [p.mask for p in weights]
    flat_mask = np.concatenate([mask.ravel() for mask in masks])
    # At most floor((1-sparsity)*total) connections may survive.
    desired_removed = int(np.ceil(sparsity * len(flat_mask) - 1e-12))
    remove_count = max(0, desired_removed - int((~flat_mask).sum()))
    if remove_count == 0:
        return {'removed_now': 0}
    diagnostics = {}
    if method == 'magnitude':
        arrays = [np.abs(p.data) for p in weights]
    else:
        arrays, diagnostics = model_scores(model, x, y, step, damping)
    scores = np.concatenate([array.ravel() for array in arrays])
    candidates = np.flatnonzero(flat_mask)
    if not np.isfinite(scores[candidates]).all():
        raise FloatingPointError("Non-finite active score.")
    # Stable sorting gives deterministic tie-breaking by global index.
    chosen = candidates[np.argsort(scores[candidates], kind='stable')[:remove_count]]
    diagnostics['sum_individual_scores'] = float(scores[chosen].sum())
    if x is not None:
        diagnostics['calibration_loss_before'] = float(cross_entropy(model(x), y).data)
    flat_mask[chosen] = False
    offset = 0
    for p in weights:
        size = p.data.size
        optimizer.set_mask(p, flat_mask[offset:offset+size].reshape(p.data.shape))
        offset += size
    if x is not None:
        diagnostics['calibration_loss_after'] = float(cross_entropy(model(x), y).data)
        diagnostics['actual_calibration_loss_change'] = (
            diagnostics['calibration_loss_after']-diagnostics['calibration_loss_before'])
    return dict(diagnostics, removed_now=len(chosen), total_removed=int((~flat_mask).sum()))
