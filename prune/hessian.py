"""A numerical Hessian of our own autodiff gradients and a damped Taylor score."""
import numpy as np
from engine import cross_entropy


def central_hessian(gradient_function, point, step=1e-4):
    point = np.array(point, dtype=np.float64, copy=True)
    if point.ndim != 1 or not len(point) or step <= 0:
        raise ValueError("Expected a nonempty vector and positive step.")
    hessian = np.empty((len(point), len(point)))
    for j in range(len(point)):
        h = step * max(1.0, abs(point[j]))
        plus, minus = point.copy(), point.copy()
        plus[j] += h
        minus[j] -= h
        hessian[:, j] = (gradient_function(plus) - gradient_function(minus)) / (2*h)
    if not np.isfinite(hessian).all():
        raise FloatingPointError("Non-finite numerical Hessian.")
    asymmetry = float(np.linalg.norm(hessian-hessian.T) /
                      max(np.linalg.norm(hessian), 1e-12))
    return (hessian + hessian.T) / 2, asymmetry


def stabilize_hessian(hessian, damping=1e-3):
    """Shift the symmetric Hessian to have a scale-aware positive eigenvalue floor.

    This changes the quadratic model and is reported explicitly. No inverse is
    needed for a one-coordinate deletion score. It is not Optimal Brain Surgeon.
    """
    if damping <= 0 or not np.isfinite(hessian).all():
        raise ValueError("Damping must be positive and Hessian finite.")
    symmetric = (hessian + hessian.T) / 2
    eigenvalues = np.linalg.eigvalsh(symmetric)
    floor = damping * max(1.0, float(np.max(np.abs(eigenvalues))))
    shift = max(0.0, floor - float(eigenvalues[0]))
    stable = symmetric + shift * np.eye(len(symmetric))
    diagnostics = {
        'min_eigenvalue': float(eigenvalues[0]),
        'max_eigenvalue': float(eigenvalues[-1]),
        'diagonal_shift': shift,
        'condition_after_shift': float((eigenvalues[-1]+shift)/(eigenvalues[0]+shift)),
    }
    return stable, diagnostics


def removal_scores(weights, gradient, hessian):
    """Signed second-order estimate for independently zeroing each coordinate."""
    return -weights * gradient + 0.5 * weights**2 * np.diag(hessian)


def model_scores(model, x, y, step=1e-4, damping=1e-3):
    parameters = model.parameters()
    snapshots = [(p.data.copy(), p.grad.copy()) for p in parameters]
    weights = model.weights()
    active = [np.flatnonzero(p.mask.ravel()) for p in weights]
    sizes = [len(indices) for indices in active]
    point = np.concatenate([p.data.ravel()[indices] for p, indices in zip(weights, active)])
    if len(point) == 0:
        raise ValueError("No active weights remain.")

    def gradient_at(vector):
        offset = 0
        for p, indices, size in zip(weights, active, sizes):
            p.data.ravel()[indices] = vector[offset:offset+size]
            offset += size
        objective = cross_entropy(model(x), y)
        objective.backward()
        return np.concatenate([p.grad.ravel()[indices] for p, indices in zip(weights, active)])

    try:
        gradient = gradient_at(point)
        hessian, asymmetry = central_hessian(gradient_at, point, step)
        stable, diagnostics = stabilize_hessian(hessian, damping)
        scores = removal_scores(point, gradient, stable)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Non-finite pruning score.")
        diagnostics.update(active_dimension=len(point), asymmetry=asymmetry,
                           gradient_norm=float(np.linalg.norm(gradient)),
                           finite_difference_step=step, damping=damping,
                           gradient_evaluations=2*len(point)+1)
        result, offset = [], 0
        for p, indices, size in zip(weights, active, sizes):
            array = np.full(p.data.size, np.inf)
            array[indices] = scores[offset:offset+size]
            result.append(array.reshape(p.data.shape))
            offset += size
        return result, diagnostics
    finally:
        # Curvature probes must not become optimizer updates or change its inputs.
        for p, (values, gradient) in zip(parameters, snapshots):
            p.data[...] = values
            p.grad[...] = gradient
