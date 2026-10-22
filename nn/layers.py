"""Small composable layers. Masks are explicit nodes in the forward graph."""
import numpy as np
from engine import Tensor


class Parameter(Tensor):
    def __init__(self, data, prunable=False):
        super().__init__(data)
        self.prunable = prunable
        self._mask = np.ones_like(self.data, dtype=bool)

    @property
    def mask(self):
        # Prevent accidental in-place mask changes that bypass optimizer resets.
        return self._mask.copy()

    def effective(self):
        return self * Tensor(self._mask, requires_grad=False)


class Linear:
    def __init__(self, inputs, outputs, rng):
        # Xavier uniform suits the tanh layers used by this project.
        bound = np.sqrt(6.0 / (inputs + outputs))
        self.weight = Parameter(rng.uniform(-bound, bound, (inputs, outputs)), True)
        self.bias = Parameter(np.zeros(outputs))

    def __call__(self, x):
        return Tensor.wrap(x) @ self.weight.effective() + self.bias

    def parameters(self):
        return [self.weight, self.bias]


class Tanh:
    def __call__(self, x):
        return x.tanh()


class MLP:
    def __init__(self, inputs=2, hidden=24, outputs=2, seed=0):
        rng = np.random.default_rng(seed)
        self.linears = [Linear(inputs, hidden, rng), Linear(hidden, outputs, rng)]
        self.layers = [self.linears[0], Tanh(), self.linears[1]]

    def __call__(self, x):
        x = Tensor.wrap(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.linears for p in layer.parameters()]

    def weights(self):
        return [layer.weight for layer in self.linears]

    def dense_logits(self, x):
        """Inference without autograd overhead; still performs dense matmuls."""
        for i, layer in enumerate(self.linears):
            x = x @ (layer.weight.data * layer.weight.mask) + layer.bias.data
            if i < len(self.linears) - 1:
                x = np.tanh(x)
        return x
