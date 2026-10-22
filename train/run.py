"""Run one reproducible training experiment: python -m train.run --help."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time
import numpy as np
from engine import Tensor, cross_entropy
from nn import Adam, MLP
from prune.inference import measure_cost
from prune.schedule import cubic_sparsity, prune_to
from .data import load_data


@dataclass
class Config:
    method: str = 'hessian'
    sparsity: float = 0.9
    seed: int = 0
    data_seed: int = 2026
    examples: int = 1200
    hidden: int = 24
    epochs: int = 120
    batch_size: int = 64
    lr: float = 0.01
    prune_start: int = 30
    prune_end: int = 90
    prune_every: int = 10
    calibration_size: int = 128
    hessian_step: float = 1e-4
    damping: float = 1e-3
    dataset: str = 'moons'

    def validate(self):
        if self.dataset not in ('moons', 'covertype'):
            raise ValueError('Unknown dataset.')
        if self.method not in ('dense','magnitude','hessian'):
            raise ValueError('Unknown method.')
        if not 0 <= self.sparsity < 1:
            raise ValueError('Sparsity must be in [0, 1).')
        if min(self.hidden, self.epochs, self.batch_size, self.calibration_size, self.prune_every) <= 0:
            raise ValueError('Counts must be positive.')
        if self.method != 'dense' and not 0 <= self.prune_start < self.prune_end < self.epochs:
            raise ValueError('Need 0 <= prune_start < prune_end < epochs for recovery training.')


def evaluate(model, split):
    x, y = split
    logits = model.dense_logits(x)
    return {
        'loss': float(cross_entropy(Tensor(logits, False), y).data),
        'accuracy': float((logits.argmax(axis=1) == y).mean()),
    }


def save_model(model, config, normalization, path, preprocessing=None):
    arrays = {'config': np.array(json.dumps(asdict(config))),
              'mean': normalization[0], 'scale': normalization[1]}
    if preprocessing is not None:
        arrays['preprocessing'] = np.array(json.dumps(preprocessing))
    for i, layer in enumerate(model.linears):
        arrays[f'w{i}'] = layer.weight.data
        arrays[f'mask{i}'] = layer.weight.mask
        arrays[f'b{i}'] = layer.bias.data
    np.savez_compressed(path, **arrays)


def load_model(path):
    with np.load(path, allow_pickle=False) as saved:
        config = Config(**json.loads(str(saved['config'])))
        model = MLP(inputs=saved['w0'].shape[0], hidden=config.hidden,
                    outputs=saved['w1'].shape[1], seed=config.seed)
        optimizer = Adam(model.parameters())
        for i, layer in enumerate(model.linears):
            optimizer.set_mask(layer.weight, saved[f'mask{i}'])
            layer.weight.data[...] = saved[f'w{i}']
            layer.bias.data[...] = saved[f'b{i}']
        normalization = (saved['mean'].copy(), saved['scale'].copy())
    return model, normalization


def run(config, directory, benchmark=True, evaluate_test=True):
    config.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = load_data(config.dataset, config.data_seed, config.examples)
    x, y = data['train']
    model = MLP(inputs=x.shape[1], hidden=config.hidden,
                outputs=len(np.unique(y)), seed=config.seed)
    optimizer = Adam(model.parameters(), lr=config.lr)
    batches = np.random.default_rng(config.seed + 5000)
    calibration_indices = np.random.default_rng(777).choice(
        len(x), min(config.calibration_size, len(x)), replace=False)
    calibration = (x[calibration_indices], y[calibration_indices])
    total = sum(p.data.size for p in model.weights())
    history, events = [], []
    start = time.perf_counter()
    for epoch in range(config.epochs+1):
        if epoch:
            for indices in np.array_split(batches.permutation(len(x)),
                                          int(np.ceil(len(x)/config.batch_size))):
                objective = cross_entropy(model(Tensor(x[indices], False)), y[indices])
                if not np.isfinite(objective.data):
                    raise FloatingPointError('Non-finite training loss.')
                objective.backward()
                optimizer.step()
            due = (epoch == config.prune_end or
                   config.prune_start < epoch < config.prune_end and
                   (epoch-config.prune_start) % config.prune_every == 0)
            if config.method != 'dense' and due:
                target = cubic_sparsity(epoch, config.prune_start, config.prune_end, config.sparsity)
                event_start = time.perf_counter()
                event = prune_to(model, optimizer, target, config.method, *calibration,
                                 step=config.hessian_step, damping=config.damping)
                events.append(dict(event, epoch=epoch, scheduled_sparsity=target,
                                   seconds=time.perf_counter()-event_start))
        train_metrics, val_metrics = evaluate(model, data['train']), evaluate(model, data['val'])
        active = sum(int(p.mask.sum()) for p in model.weights())
        history.append(dict(epoch=epoch, train_loss=train_metrics['loss'],
                            train_accuracy=train_metrics['accuracy'],
                            val_loss=val_metrics['loss'], val_accuracy=val_metrics['accuracy'],
                            sparsity=1-active/total))
    training_seconds = time.perf_counter()-start
    target = 0.0 if config.method == 'dense' else config.sparsity
    assert active <= int(np.floor((1-target)*total + 1e-10)), 'Hard budget missed.'
    assert all(np.all(p.data[~p.mask] == 0) for p in model.weights())
    # Test data is evaluated after the fixed training schedule, never for selection.
    result = {
        'config': asdict(config), 'train': train_metrics, 'val': val_metrics,
        'test': evaluate(model, data['test']) if evaluate_test else None,
        'achieved_sparsity': 1-active/total,
        'active_weights': active, 'total_weights': total,
        'training_seconds': training_seconds, 'history': history, 'pruning_events': events,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'platform': platform.platform(), 'machine': platform.machine()},
    }
    if 'metadata' in data:
        result['dataset'] = data['metadata']
    if benchmark:
        result['cost'] = measure_cost(model, data['val'][0][:128] if not evaluate_test
                                     else data['test'][0][:128])
    save_model(model, config, data['normalization'], directory/'model.npz',
               preprocessing=data.get('preprocessing'))
    (directory/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['dense','magnitude','hessian'], default='hessian')
    parser.add_argument('--sparsity', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', choices=['moons', 'covertype'], default='moons')
    parser.add_argument('--data-seed', type=int, default=2026)
    parser.add_argument('--hidden', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--prune-every', type=int, default=10)
    parser.add_argument('--out', default='results/single')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--prune-start', type=int, default=30)
    parser.add_argument('--prune-end', type=int, default=90)
    args = parser.parse_args()
    config = Config(method=args.method, sparsity=0 if args.method=='dense' else args.sparsity,
                    seed=args.seed, epochs=args.epochs, dataset=args.dataset,
                    data_seed=args.data_seed, hidden=args.hidden, batch_size=args.batch_size,
                    prune_start=args.prune_start, prune_end=args.prune_end,
                    prune_every=args.prune_every)
    result = run(config, args.out)
    from .report import plot_learning
    plot_learning(result, Path(args.out)/'learning.png')
    print(json.dumps({key: result[key] for key in ('test','achieved_sparsity','training_seconds')}, indent=2))


if __name__ == '__main__':
    main()
