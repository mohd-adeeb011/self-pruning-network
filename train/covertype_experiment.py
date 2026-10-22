"""Compare the existing pruning methods on UCI Covertype.

Run: python -m train.covertype_experiment
This is a focused 90% comparison, not the full submission sparsity sweep.
"""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from prune.inference import DenseExport, SparseExport
from .covertype_data import load_covertype
from .provenance import source_digest, write_json
from .run import Config, evaluate, load_model, run


def multiclass_metrics(logits, labels):
    """Keep all seven classes visible, including classes never predicted."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(logits).argmax(axis=1)
    confusion = np.bincount(7*labels+predictions, minlength=49).reshape(7, 7)
    correct = confusion.diagonal()
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    recall = np.divide(correct, support, out=np.zeros(7), where=support > 0)
    precision = np.divide(correct, predicted, out=np.zeros(7), where=predicted > 0)
    f1 = np.divide(2*precision*recall, precision+recall,
                   out=np.zeros(7), where=precision+recall > 0)
    return dict(confusion_matrix=confusion.tolist(), per_class_recall=recall.tolist(),
                balanced_accuracy=float(recall.mean()), macro_f1=float(f1.mean()),
                predicted_class_counts=predicted.tolist())


def configurations(seeds):
    entries = []
    for seed in seeds:
        base = Config(dataset='covertype', examples=581012, hidden=8, seed=seed,
                      data_seed=2026, epochs=60, batch_size=512, lr=.003,
                      prune_start=15, prune_end=45, prune_every=5,
                      calibration_size=128)
        for method in ('dense', 'magnitude', 'hessian'):
            config = replace(base, method=method,
                             sparsity=0 if method == 'dense' else .9)
            entries.append(dict(method=method, seed=seed,
                                directory=f'{method}_s{config.sparsity:g}_seed{seed}',
                                config=asdict(config)))
    return entries


def finalize(entry, output, data):
    directory = output / entry['directory']
    result = json.loads((directory / 'metrics.json').read_text())
    model, normalization = load_model(directory / 'model.npz')
    for actual, expected in zip(normalization, data['normalization']):
        np.testing.assert_array_equal(actual, expected)
    assert result['dataset']['sha256'] == data['metadata']['sha256']
    assert result['dataset']['split_class_counts'] == data['metadata']['split_class_counts']
    for split in ('train', 'val', 'test'):
        x, y = data[split]
        actual = evaluate(model, (x, y))
        if result[split] is not None:
            for name in ('accuracy', 'loss'):
                np.testing.assert_allclose(actual[name], result[split][name], rtol=0, atol=1e-12)
        result[split] = actual
        if split == 'test':
            result[split].update(multiclass_metrics(DenseExport(model)(x), y))
        # Sparse export must give the same answers without multiplying by zeros.
        sample = x[:min(512, len(x))]
        np.testing.assert_allclose(DenseExport(model)(sample), SparseExport(model)(sample),
                                   rtol=1e-11, atol=1e-11)
    active = sum(int(p.mask.sum()) for p in model.weights())
    assert active == result['active_weights']
    assert all(np.all(p.data[~p.mask] == 0) for p in model.weights())
    if entry['method'] != 'dense':
        target = result['config']['sparsity']
        assert active <= int(np.floor((1-target) * result['total_weights'] + 1e-10))
    assert np.isfinite([[h['train_loss'], h['val_loss']] for h in result['history']]).all()
    write_json(directory / 'metrics.json', result)
    return result


def summarize(entries, results, output, data):
    rows = []
    for method in ('dense', 'magnitude', 'hessian'):
        group = [r for e, r in zip(entries, results) if e['method'] == method]
        row = dict(method=method, seeds=len(group),
                   active_weights=group[0]['active_weights'],
                   total_weights=group[0]['total_weights'],
                   achieved_sparsity=group[0]['achieved_sparsity'])
        for split in ('train', 'val', 'test'):
            values = [r[split]['accuracy'] for r in group]
            row[f'{split}_accuracy_mean'] = float(np.mean(values))
            row[f'{split}_accuracy_sd'] = (float(np.std(values, ddof=1))
                                            if len(values) > 1 else 0.)
        for name in ('balanced_accuracy', 'macro_f1'):
            values = [r['test'][name] for r in group]
            row[f'test_{name}_mean'] = float(np.mean(values))
            row[f'test_{name}_sd'] = (float(np.std(values, ddof=1))
                                      if len(values) > 1 else 0.)
        for name in ('dense_linear_flops_per_example', 'sparse_linear_flops_per_example',
                     'dense_seconds', 'sparse_seconds'):
            row[name] = float(np.mean([r['cost'][name] for r in group]))
        row['mean_training_seconds'] = float(np.mean([r['training_seconds'] for r in group]))
        rows.append(row)
    with (output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_method = {method: {e['seed']: r for e, r in zip(entries, results)
                          if e['method'] == method}
                 for method in ('dense', 'magnitude', 'hessian')}
    differences = [dict(seed=seed, test_accuracy_difference=(
        by_method['hessian'][seed]['test']['accuracy']-
        by_method['magnitude'][seed]['test']['accuracy']))
        for seed in sorted(by_method['dense'])]
    write_json(output / 'paired.json', dict(
        comparison='Hessian minus magnitude, test accuracy, 90% target',
        pairs=differences,
        mean_difference=float(np.mean([v['test_accuracy_difference'] for v in differences])),
        sd_difference=(float(np.std([v['test_accuracy_difference'] for v in differences],
                                    ddof=1)) if len(differences) > 1 else 0.)))
    lines = ['# UCI Covertype: focused larger-data comparison', '',
             f"Original rows: {data['metadata']['rows']:,}; 54 inputs; 7 classes. ",
             f"Split: {data['metadata']['split_sizes']}. Source SHA-256: "
             f"`{data['metadata']['sha256']}`. All three methods use the same "
             'initialization, batches, model width (8), 60 epochs and 90% target. '
             'The existing score, damping and masking code is unchanged. '
             'Test labels were evaluated after all fits finished.', '',
             '| Method | Active weights | Train accuracy | Validation accuracy | '
             'Test accuracy | Test SD | Balanced accuracy | Macro F1 | Linear FLOPs/example |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['method']} | {row['active_weights']}/"
                     f"{row['total_weights']} | {row['train_accuracy_mean']:.2%} | "
                     f"{row['val_accuracy_mean']:.2%} | {row['test_accuracy_mean']:.2%} | "
                     f"{row['test_accuracy_sd']:.2%} | "
                     f"{row['test_balanced_accuracy_mean']:.2%} | "
                     f"{row['test_macro_f1_mean']:.2%} | "
                     f"{row['sparse_linear_flops_per_example']:.0f} |")
    lines += ['', f"Paired Hessian minus magnitude test accuracy: "
              f"{100*np.mean([v['test_accuracy_difference'] for v in differences]):+.2f} "
              'percentage points. Per-seed values are in `paired.json`. '
              'Three seeds on one random split cannot establish general superiority.', '',
              'FLOPs count linear arithmetic only. `summary.csv` includes measured '
              'dense and sparse wall-clock time; sparse Python loops can be slower. '
              'The random row split can mix nearby forest cells across splits, '
              'so it may overstate performance on a new geographical area. '
              'This 90% comparison does not replace the full sparsity sweep '
              'required for submission.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/covertype')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--report-only', action='store_true',
                        help='Regenerate summary from completed, checked model results.')
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('Seeds must be unique.')
    # Data is checked before any output is created.
    data = load_covertype()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    entries = configurations(args.seeds)
    if args.report_only:
        protocol = json.loads((output / 'protocol.json').read_text())
        assert protocol['entries'] == entries
        assert protocol['dataset_sha256'] == data['metadata']['sha256']
        assert all(json.loads((output / entry['directory'] / 'metrics.json').read_text())['test']
                   is not None for entry in entries)
        results = [finalize(entry, output, data) for entry in entries]
        rows = summarize(entries, results, output, data)
        write_json(output / 'audit.json', dict(
            verified_models=len(results), sparse_dense_logits_match=True,
            reload_metrics_match=True, masks_and_budget_checked=True,
            pre_pruning_histories_match=True, test_used_for_selection=False,
            training_source_sha256=protocol['source_sha256'],
            reporting_source_sha256=source_digest(),
            report_recovered_after_all_fits_and_checks=True))
        for row in rows:
            print(f"{row['method']:9} test={row['test_accuracy_mean']:.2%} "
                  f"SD={row['test_accuracy_sd']:.2%}", flush=True)
        print(f"Wrote {output / 'RESULTS.md'} from checked results", flush=True)
        return
    protocol = dict(source_sha256=source_digest(),
                    dataset_sha256=data['metadata']['sha256'],
                    comparison='paired test accuracy at 90% target',
                    test_used_for_selection=False, entries=entries)
    write_json(output / 'protocol.json', protocol)
    started = time.perf_counter()
    for index, entry in enumerate(entries, 1):
        print(f"[{index}/{len(entries)}] {entry['directory']}", flush=True)
        result = run(Config(**entry['config']), output / entry['directory'],
                     evaluate_test=False)
        print(f"  train={result['train']['accuracy']:.3f} "
              f"val={result['val']['accuracy']:.3f} "
              f"weights={result['active_weights']}/{result['total_weights']} "
              f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    assert protocol['source_sha256'] == source_digest(), 'Source changed during fits.'
    dense_history = {e['seed']: json.loads((output / e['directory'] / 'metrics.json').read_text())['history']
                     for e in entries if e['method'] == 'dense'}
    for entry in entries:
        if entry['method'] == 'dense':
            continue
        history = json.loads((output / entry['directory'] / 'metrics.json').read_text())['history']
        first_removal = entry['config']['prune_start']+entry['config']['prune_every']
        assert history[:first_removal] == dense_history[entry['seed']][:first_removal]
    print('All fits complete. Evaluating test labels.', flush=True)
    results = [finalize(entry, output, data) for entry in entries]
    rows = summarize(entries, results, output, data)
    write_json(output / 'audit.json', dict(
        verified_models=len(results), sparse_dense_logits_match=True,
        reload_metrics_match=True, masks_and_budget_checked=True,
        pre_pruning_histories_match=True, test_used_for_selection=False,
        source_sha256=source_digest(), elapsed_seconds=time.perf_counter()-started))
    for row in rows:
        print(f"{row['method']:9} test={row['test_accuracy_mean']:.2%} "
              f"SD={row['test_accuracy_sd']:.2%}", flush=True)
    print(f"Wrote {output / 'RESULTS.md'} in {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == '__main__':
    main()
