"""Generate plots and a literal, data-backed report from completed runs."""
import csv
import json
import os
from pathlib import Path
import numpy as np


def plotting():
    os.environ.setdefault('MPLCONFIGDIR', str(Path('.mplconfig').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_learning(result, path):
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    history = result['history']
    epochs = [row['epoch'] for row in history]
    for split in ('train','val'):
        axes[0].plot(epochs, [r[f'{split}_loss'] for r in history], label=split)
        axes[1].plot(epochs, [r[f'{split}_accuracy'] for r in history], label=split)
    for ax, label in zip(axes, ('Cross-entropy loss','Accuracy')):
        ax.set(xlabel='Epoch', ylabel=label)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def report(results, output):
    output = Path(output)
    groups = {}
    for r in results:
        key = (r['config']['method'], r['config']['sparsity'])
        groups.setdefault(key, []).append(r)
    rows = []
    for (method, target), runs in sorted(groups.items()):
        accuracy = np.array([r['test']['accuracy'] for r in runs])
        rows.append(dict(method=method, target_sparsity=target,
                         achieved_sparsity=float(np.mean([r['achieved_sparsity'] for r in runs])),
                         seeds=len(runs), mean_test_accuracy=float(accuracy.mean()),
                         sd_test_accuracy=float(accuracy.std(ddof=1)) if len(runs)>1 else 0.0,
                         mean_training_seconds=float(np.mean([r['training_seconds'] for r in runs]))))
    with (output/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt = plotting()
    fig, ax = plt.subplots(figsize=(7, 4))
    dense = [r for r in rows if r['method']=='dense']
    for method in ('magnitude','hessian'):
        selected = dense + [r for r in rows if r['method']==method]
        selected.sort(key=lambda r:r['achieved_sparsity'])
        ax.errorbar([100*r['achieved_sparsity'] for r in selected],
                    [100*r['mean_test_accuracy'] for r in selected],
                    yerr=[100*r['sd_test_accuracy'] for r in selected],
                    marker='o', capsize=3, label=method)
    ax.set(xlabel='Achieved weight sparsity (%)', ylabel='Test accuracy (%)',
           title='Accuracy–sparsity trade-off (error bars: sample SD across seeds)')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output/'pareto.png', dpi=160)
    plt.close(fig)
    pairs = []
    for target in sorted({key[1] for key in groups if key[0]=='hessian'}):
        first = {r['config']['seed']:r for r in groups['hessian',target]}
        second = {r['config']['seed']:r for r in groups['magnitude',target]}
        delta = [first[s]['test']['accuracy']-second[s]['test']['accuracy'] for s in sorted(first)]
        pairs.append(dict(target=target, seeds=sorted(first), differences=delta,
                          mean_difference=float(np.mean(delta)),
                          sd_difference=float(np.std(delta, ddof=1)) if len(delta)>1 else 0.0))
    (output/'paired_comparisons.json').write_text(json.dumps(pairs, indent=2)+'\n')
    # Predeclared claim: target 90%, irrespective of whether Hessian wins.
    claim = next((r for r in pairs if np.isclose(r['target'],0.9)), None)
    lines = ['# Measured results', '',
             'All numbers below come from the metrics.json files in this directory.', '',
             '| Method | Target | Achieved | Test accuracy mean ± SD | Seeds |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['method']} | {row['target_sparsity']:.0%} | {row['achieved_sparsity']:.2%} | "
                     f"{100*row['mean_test_accuracy']:.2f} ± {100*row['sd_test_accuracy']:.2f}% | {row['seeds']} |")
    if claim:
        h = next(r for r in rows if r['method']=='hessian' and np.isclose(r['target_sparsity'],0.9))
        m = next(r for r in rows if r['method']=='magnitude' and np.isclose(r['target_sparsity'],0.9))
        lines += ['', '**Falsifiable claim:** At a 90% target, the Hessian method achieved '
                  f"{100*h['mean_test_accuracy']:.2f}% mean test accuracy versus "
                  f"{100*m['mean_test_accuracy']:.2f}% for magnitude pruning across {h['seeds']} seeds. "
                  f"The paired difference was {100*claim['mean_difference']:.2f} percentage points "
                  f"with sample SD {100*claim['sd_difference']:.2f} points."]
    lines += ['', 'These seeds vary initialization and batch order on one fixed data split. '
              'They do not estimate variation across datasets. Small seed counts do not establish broad superiority.',
              '', '| Method / target | Active / total weights | Sparse / dense linear FLOPs | Sparse / dense latency (µs/batch) |',
              '|---|---:|---:|---:|']
    for key, runs in sorted(groups.items()):
        costs = [r['cost'] for r in runs]
        lines.append(f"| {key[0]} / {key[1]:.0%} | {costs[0]['active_mask_weights']} / {costs[0]['total_weights']} | "
                     f"{np.mean([c['sparse_linear_flops_per_example'] for c in costs]):.0f} / {costs[0]['dense_linear_flops_per_example']} | "
                     f"{1e6*np.mean([c['sparse_seconds'] for c in costs]):.1f} / {1e6*np.mean([c['dense_seconds'] for c in costs]):.1f} |")
    lines += ['', 'Latency uses a batch of 128, excludes export time, and compares frozen exports. '
              'Python sparse loops can be slower than dense BLAS despite doing fewer weight operations. '
              'FLOPs count two operations per retained weight plus bias addition; tanh and data movement are excluded. '
              'Training uses dense masked matrices and incurs additional Hessian work. No training speedup is claimed.',
              '', 'Sparse array payload includes indices; it can exceed dense storage for this small model. '
              'Raw metrics contain byte counts, timing rounds, environment details, and curvature diagnostics.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    for method in ('dense','hessian','magnitude'):
        candidates = [r for r in results if r['config']['method']==method]
        if candidates:
            chosen = next((r for r in candidates if np.isclose(r['config']['sparsity'],0.9)), candidates[0])
            plot_learning(chosen, output/f'learning_{method}.png')

