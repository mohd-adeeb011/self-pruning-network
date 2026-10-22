"""Float64 reverse-mode autodiff. Backward resets reachable gradients each call.

Backward rules use saved forward values. Gradients are NumPy arrays, so this
engine computes first derivatives; Hessians are obtained by finite differences
of those derivatives, not by differentiating the backward graph.
"""
import numpy as np


def unbroadcast(gradient, shape):
    """Undo NumPy broadcasting by adding contributions to repeated inputs."""
    gradient = np.asarray(gradient, dtype=np.float64)
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    def __init__(self, data, requires_grad=True, parents=()):
        self.data = np.array(data, dtype=np.float64, copy=True)
        self.requires_grad = bool(requires_grad)
        self.grad = np.zeros_like(self.data)
        self.parents = tuple(parents)
        self._backward = lambda: None

    @staticmethod
    def wrap(value):
        return value if isinstance(value, Tensor) else Tensor(value, False)

    def _add_grad(self, gradient):
        if self.requires_grad:
            self.grad += unbroadcast(gradient, self.data.shape)

    def backward(self, gradient=None):
        if gradient is None:
            if self.data.size != 1:
                raise ValueError("Non-scalar output requires an explicit gradient.")
            gradient = np.ones_like(self.data)
        gradient = np.asarray(gradient, dtype=np.float64)
        if gradient.shape != self.data.shape:
            raise ValueError("Seed gradient must match the output shape.")
        # Iterative postorder avoids recursion depth limits on long chains.
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in seen:
                seen.add(node)
                stack.append((node, True))
                stack.extend((parent, False) for parent in reversed(node.parents))
        for node in order:
            node.grad.fill(0.0)
        self.grad[...] = gradient
        for node in reversed(order):
            node._backward()

    def __add__(self, other):
        other = Tensor.wrap(other)
        out = Tensor(self.data + other.data,
                     self.requires_grad or other.requires_grad, (self, other))
        def backward():
            self._add_grad(out.grad)
            other._add_grad(out.grad)
        out._backward = backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Tensor.wrap(other))

    def __rsub__(self, other):
        return Tensor.wrap(other) - self

    def __mul__(self, other):
        other = Tensor.wrap(other)
        left, right = self.data.copy(), other.data.copy()
        out = Tensor(left * right, self.requires_grad or other.requires_grad,
                     (self, other))
        def backward():
            self._add_grad(out.grad * right)
            other._add_grad(out.grad * left)
        out._backward = backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Tensor.wrap(other)
        left, right = self.data.copy(), other.data.copy()
        if np.any(right == 0):
            raise ZeroDivisionError("Tensor division by zero.")
        out = Tensor(left / right, self.requires_grad or other.requires_grad,
                     (self, other))
        def backward():
            self._add_grad(out.grad / right)
            other._add_grad(-out.grad * (left / right) / right)
        out._backward = backward
        return out

    def __rtruediv__(self, other):
        return Tensor.wrap(other) / self

    def __matmul__(self, other):
        other = Tensor.wrap(other)
        left, right = self.data.copy(), other.data.copy()
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("This minimal engine supports 2D matrix multiplication.")
        out = Tensor(left @ right, self.requires_grad or other.requires_grad,
                     (self, other))
        def backward():
            self._add_grad(out.grad @ right.T)
            other._add_grad(left.T @ out.grad)
        out._backward = backward
        return out

    def sum(self, axis=None, keepdims=False):
        # NumPy validates axis values, including duplicates, on the forward pass.
        value = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(value, self.requires_grad, (self,))
        axes = None if axis is None else tuple(
            a % self.data.ndim for a in ((axis,) if isinstance(axis, int) else axis)
        )
        def backward():
            grad = out.grad
            if axes is not None and not keepdims:
                for a in sorted(axes):
                    grad = np.expand_dims(grad, a)
            self._add_grad(np.broadcast_to(grad, self.data.shape))
        out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = int(np.prod([self.data.shape[a] for a in axes]))
        if count == 0:
            raise ValueError("Mean of an empty reduction.")
        return self.sum(axis=axis, keepdims=keepdims) / count

    def relu(self):
        positive = self.data > 0
        out = Tensor(np.maximum(self.data, 0), self.requires_grad, (self,))
        out._backward = lambda: self._add_grad(out.grad * positive)
        return out

    def tanh(self):
        value = np.tanh(self.data)
        out = Tensor(value, self.requires_grad, (self,))
        out._backward = lambda: self._add_grad(out.grad * (1 - value * value))
        return out


def cross_entropy(logits, labels):
    """Mean softmax cross-entropy; avoids log(softmax) underflow."""
    logits = Tensor.wrap(logits)
    labels = np.asarray(labels)
    values = logits.data
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("Expected nonempty logits with shape (batch, classes).")
    if labels.shape != (len(values),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be a one-dimensional integer array.")
    if np.any(labels < 0) or np.any(labels >= values.shape[1]):
        raise ValueError("Label is outside the class range.")
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite logits.")
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    totals = exp_values.sum(axis=1, keepdims=True)
    loss = (np.log(totals[:, 0]) - shifted[np.arange(len(labels)), labels]).mean()
    derivative = exp_values / totals
    derivative[np.arange(len(labels)), labels] -= 1
    derivative /= len(labels)
    out = Tensor(loss, logits.requires_grad, (logits,))
    out._backward = lambda: logits._add_grad(out.grad * derivative)
    return out
