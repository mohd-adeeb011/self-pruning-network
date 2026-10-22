"""Adam with per-entry ages, so revived connections get fresh bias correction."""
import numpy as np


class Adam:
    def __init__(self, parameters, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        if lr <= 0 or eps <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Invalid Adam hyperparameters.")
        self.parameters = list(parameters)
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.state = {
            p: dict(m=np.zeros_like(p.data), v=np.zeros_like(p.data),
                    age=np.zeros_like(p.data, dtype=np.int64))
            for p in self.parameters
        }

    def set_mask(self, parameter, mask):
        if parameter not in self.state:
            raise ValueError("Parameter is not owned by this optimizer.")
        mask = np.asarray(mask)
        if mask.shape != parameter.data.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError("Mask must be binary and match parameter shape.")
        mask = mask.astype(bool)
        if not parameter.prunable and not mask.all():
            raise ValueError("Bias parameters are not prunable in this model.")
        changed = mask != parameter._mask
        clear = changed | ~mask
        # Both removed and revived entries start at zero with fresh state.
        parameter.data[clear] = 0.0
        parameter.grad[clear] = 0.0
        for state in self.state[parameter].values():
            state[clear] = 0
        parameter._mask = mask.copy()

    def step(self):
        # Check all gradients before changing any parameter.
        if any(not np.isfinite(p.grad).all() for p in self.parameters):
            raise FloatingPointError("Non-finite optimizer gradient.")
        for p in self.parameters:
            active = p._mask
            state = self.state[p]
            m, v, age = state['m'], state['v'], state['age']
            g = np.where(active, p.grad, 0.0)
            m[...] = self.beta1 * m + (1 - self.beta1) * g
            v[...] = self.beta2 * v + (1 - self.beta2) * g * g
            age[active] += 1
            m[~active] = 0
            v[~active] = 0
            age[~active] = 0
            # Inactive entries have age zero; never divide by their zero correction.
            first = m[active] / (1 - self.beta1 ** age[active])
            second = v[active] / (1 - self.beta2 ** age[active])
            p.data[active] -= self.lr * first / (np.sqrt(second) + self.eps)
            p.data[~active] = 0.0
            if not np.isfinite(p.data).all():
                raise FloatingPointError("Non-finite parameter after Adam step.")
