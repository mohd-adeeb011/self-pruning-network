"""Frozen dense and sparse exports. Sparse inference skips removed connections."""
import time
import numpy as np


class DenseExport:
    def __init__(self, model):
        self.layers = [(layer.weight.data * layer.weight.mask, layer.bias.data.copy())
                       for layer in model.linears]

    def __call__(self, x):
        for i, (weight, bias) in enumerate(self.layers):
            x = x @ weight + bias
            if i < len(self.layers)-1:
                x = np.tanh(x)
        return x

    @property
    def nbytes(self):
        return sum(w.nbytes+b.nbytes for w,b in self.layers)


class SparseExport:
    def __init__(self, model):
        self.layers = []
        for layer in model.linears:
            weight = layer.weight.data * layer.weight.mask
            columns = []
            for j in range(weight.shape[1]):
                indices = np.flatnonzero(weight[:, j])
                columns.append((indices, weight[indices, j].copy()))
            self.layers.append((columns, layer.bias.data.copy()))

    def __call__(self, x):
        for i, (columns, bias) in enumerate(self.layers):
            out = np.broadcast_to(bias, (len(x), len(bias))).copy()
            for j, (indices, values) in enumerate(columns):
                if len(indices):
                    # Only retained, nonzero connections enter this dot product.
                    out[:, j] += x[:, indices] @ values
            x = np.tanh(out) if i < len(self.layers)-1 else out
        return x

    @property
    def active(self):
        return sum(len(values) for cols, _ in self.layers for _, values in cols)

    @property
    def nbytes(self):
        # Array payload only. Python object overhead is intentionally not hidden.
        return sum(b.nbytes + sum(ix.nbytes+v.nbytes for ix,v in cols)
                   for cols,b in self.layers)


def measure_cost(model, x, repeats=100, rounds=5):
    dense, sparse = DenseExport(model), SparseExport(model)
    np.testing.assert_allclose(dense(x), sparse(x), rtol=1e-11, atol=1e-11)
    for _ in range(10):
        dense(x)
        sparse(x)
    timings = {'dense': [], 'sparse': []}
    # Alternate order across rounds to reduce timing-order bias.
    for r in range(rounds):
        items = [('dense',dense),('sparse',sparse)]
        for name, forward in items[::1 if r % 2 == 0 else -1]:
            start = time.perf_counter()
            for _ in range(repeats):
                forward(x)
            timings[name].append((time.perf_counter()-start)/repeats)
    total = sum(p.data.size for p in model.weights())
    bias_count = sum(layer.bias.data.size for layer in model.linears)
    return {
        'batch_size': len(x), 'total_weights': total,
        'active_mask_weights': sum(int(p.mask.sum()) for p in model.weights()),
        'nonzero_weights': sparse.active, 'bias_parameters': bias_count,
        'dense_linear_flops_per_example': 2*total+bias_count,
        'sparse_linear_flops_per_example': 2*sparse.active+bias_count,
        'dense_seconds': float(np.median(timings['dense'])),
        'sparse_seconds': float(np.median(timings['sparse'])),
        'timing_rounds_seconds': timings,
        'dense_array_bytes': dense.nbytes, 'sparse_array_bytes': sparse.nbytes,
        'flop_convention': '2 per weight MAC plus 1 per bias; excludes tanh, indexing, allocation',
        'byte_convention': 'NumPy payload; excludes Python object/container overhead',
    }
