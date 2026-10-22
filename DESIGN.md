# Design and limits

## Why a weight is removed

Let `L` be cross-entropy on fixed training rows. Let `g` be its gradient and
`H` its Hessian. For a small weight change `delta`, the Taylor estimate is:

```text
change in L ≈ g.T @ delta + 0.5 * delta.T @ H @ delta
```

To delete weight `i`, set `delta[i] = -w[i]` and all other entries to zero. Its
estimated cost is `-g[i]*w[i] + 0.5*H[i,i]*w[i]**2`. The first term matters
because we prune during training, when the gradient may be nonzero. We remove
the active weights with the lowest **signed** cost. This ranks individual
deletions; deleting many together adds interactions that this score omits.

The [score implementation](prune/hessian.py) computes a numerical Hessian by
perturbing each active weight and differencing gradients from our autodiff
engine. It holds biases and already removed weights fixed. We use float64,
symmetrize the estimate, and add a diagonal shift if needed to give the matrix
a positive eigenvalue floor. The score uses the shifted diagonal. We record
asymmetry, the shift, condition number, and actual calibration-loss change.
The shift makes the score more stable but changes the approximation.

We compute the **full** Hessian, but its off-diagonal values affect ranking only
through the global shift. This is a cost and a limit of this small-model design.
Our method does not use a full compensation update such as [Optimal Brain
Surgeon](https://papers.nips.cc/paper_files/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf).
That method changes surviving weights as it removes one weight near a trained
minimum. We prune many weights during training and let later Adam steps adjust
the survivors. A full update would need a stable solve, a policy for many
deletions, and a policy for Adam state after abrupt weight changes. We have not
tested whether it would improve these results.

## Schedule, masks, and gradients

After a warm-up, a cubic ramp increases the target sparsity. A global sort
selects weights across both layers. We remove at least `ceil(target * P)` of
the `P` connection weights. Biases are not in this budget. Each selected weight
becomes exactly zero and stays behind a binary mask.

The forward graph uses `effective_weight = mask * raw_weight`. The chain rule
gives `raw_gradient = mask * effective_gradient`. The raw gradient of a removed
weight is therefore zero. We do not pass a surrogate gradient through the
mask. The [Adam implementation](nn/optim.py) clears a removed or restored
entry's value, gradient, first moment, second moment, and update age. An
unchanged active entry keeps its state. A test compares a restored entry with
a fresh Adam entry. The current schedule only removes weights; it has no
automatic regrowth policy.

The autodiff engine visits graph nodes in reverse topological order and sums
gradients from shared paths. It reduces broadcast gradients back to input
shapes. The softmax cross-entropy uses shifted logits and log-sum-exp for
numeric stability. Finite-difference checks cover each implemented operation.

## Cost and serving

With `A` active weights, one Hessian estimate needs `2*A+1` gradient
evaluations, `O(A^2)` storage, and a dense eigenvalue calculation. Python graph
nodes, intermediate arrays, and float64 values add time and memory. At larger
model sizes we would use a curvature approximation or Hessian-vector products;
those methods are not implemented here.

The training path still runs dense matrix operations with masks. For inference,
[SparseExport](prune/inference.py) gathers only retained input positions and
does dot products with nonzero weights. It matches the dense export's logits.
Our linear FLOP count includes two operations per active weight and one per
bias. It excludes `tanh`, indexing, allocation, and memory traffic. The sparse
Python path reduced counted work but was slower in measured wall-clock time.
There is no measured serving speedup.

For a multi-tenant service, we would prune offline and export immutable model
and preprocessing versions. We would check outputs before rollout, keep a
rollback version, set memory and request limits per tenant, and measure tail
latency under real batch sizes. We would choose a sparse kernel or a smaller
dense model only after those measurements. The current Python sparse path is
an experiment, not a production inference service.

## What the evidence supports

The Covertype results compare the methods at the same weights and three paired
training seeds. They support the fixed-split claim in the README. They do not
prove that Hessian pruning wins on other data. The random row split may place
nearby forest cells in both training and test sets. The 90% result was inspected
before the rest of the grid was run. At high sparsity, several small classes
lose recall even when ordinary accuracy remains well above the largest-class
baseline. We report balanced accuracy and macro F1 so this failure is visible.
