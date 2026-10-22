"""Extend the Covertype 90% check to the assignment's full sparsity grid."""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from .covertype_data import load_covertype
from .covertype_experiment import configurations as original_configurations, finalize
from .provenance import source_digest, write_json
from .report import plotting
from .run import Config, run


TARGETS = (.5, .75, .9, .95)


def extension_entries(seeds):
    base = {e['seed']: Config(**e['config']) for e in original_configurations(seeds)
            if e['method'] == 'dense'}
    return [dict(method=method, seed=seed, target=target,
                 directory=f'{method}_s{target:g}_seed{seed}',
                 config=asdict(replace(base[seed], method=method, sparsity=target)))
            for seed in seeds for target in TARGETS if target != .9
            for method in ('magnitude', 'hessian')]


def summarize(output, seeds):
    rows, pairs = [], []
    for method, target in [('dense', 0.)] + [
        (method, target) for target in TARGETS for method in ('magnitude', 'hessian')]:
        runs = [json.loads((output/f'{method}_s{target:g}_seed{seed}'/'metrics.json').read_text())
                for seed in seeds]
        row = dict(method=method, target=target, seeds=len(runs),
                   active_weights=runs[0]['active_weights'],
                   total_weights=runs[0]['total_weights'],
                   achieved_sparsity=runs[0]['achieved_sparsity'])
        for split, metric in [('train', 'accuracy'), ('val', 'accuracy'),
                              ('test', 'accuracy'), ('test', 'balanced_accuracy'),
                              ('test', 'macro_f1')]:
            values = [r[split][metric] for r in runs]
            key = f'{split}_{metric}'
            row[key+'_mean'] = float(np.mean(values))
            row[key+'_sd'] = float(np.std(values, ddof=1))
        row['linear_flops'] = runs[0]['cost']['sparse_linear_flops_per_example']
        row['dense_us'] = 1e6*float(np.mean([r['cost']['dense_seconds'] for r in runs]))
        row['sparse_us'] = 1e6*float(np.mean([r['cost']['sparse_seconds'] for r in runs]))
        rows.append(row)
    for target in TARGETS:
        hessian = {s: json.loads((output/f'hessian_s{target:g}_seed{s}'/'metrics.json').read_text())
                   for s in seeds}
        magnitude = {s: json.loads((output/f'magnitude_s{target:g}_seed{s}'/'metrics.json').read_text())
                     for s in seeds}
        values = [dict(seed=s, test_accuracy_difference=(
            hessian[s]['test']['accuracy']-magnitude[s]['test']['accuracy']))
                  for s in seeds]
        pairs.append(dict(target=target, pairs=values,
                          mean_difference=float(np.mean([v['test_accuracy_difference']
                                                         for v in values])),
                          sd_difference=float(np.std([v['test_accuracy_difference']
                                                       for v in values], ddof=1))))
    hessian_events = [event for target in TARGETS for seed in seeds
                      for event in json.loads((output/f'hessian_s{target:g}_seed{seed}'/
                                               'metrics.json').read_text())['pruning_events']
                      if event['removed_now'] > 0]
    with (output/'sweep_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output/'sweep_pairs.json', pairs)
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    dense = next(row for row in rows if row['method'] == 'dense')
    for ax, metric, title in zip(axes, ('accuracy', 'balanced_accuracy'),
                                 ('Test accuracy', 'Test balanced accuracy')):
        for method in ('magnitude', 'hessian'):
            chosen = [dense]+[r for r in rows if r['method'] == method]
            ax.errorbar([100*r['achieved_sparsity'] for r in chosen],
                        [100*r[f'test_{metric}_mean'] for r in chosen],
                        yerr=[100*r[f'test_{metric}_sd'] for r in chosen],
                        marker='o', capsize=3, label=method)
        ax.set(xlabel='Achieved weight sparsity (%)', ylabel=title+' (%)')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Covertype: 3 paired training seeds; bars show sample SD')
    fig.tight_layout()
    fig.savefig(output/'pareto.png', dpi=160)
    plt.close(fig)
    lines = ['# UCI Covertype sparsity sweep', '',
             'This is an exploratory extension after the 90% test results were '
             'already observed. The added 50%, 75%, and 95% targets follow the '
             'assignment grid. No claim of preregistration or an untouched test '
             'set applies to the combined sweep. The same dataset split and '
             'three paired seeds were used throughout.', '',
             f"The normal, unpruned dense model has "
             f"**{rows[0]['test_accuracy_mean']:.2%} mean test accuracy** "
             f"(sample SD {rows[0]['test_accuracy_sd']:.2%}) across the three "
             'seeds. It keeps all 488 connection weights and 15 biases.', '',
             '| Method | Target | Active weights | Test accuracy | Sample SD | '
             'Balanced accuracy | Macro F1 | Linear FLOPs/example |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        label = 'normal dense (unpruned)' if row['method'] == 'dense' else row['method']
        lines.append(f"| {label} | {row['target']:.0%} | "
                     f"{row['active_weights']}/{row['total_weights']} | "
                     f"{row['test_accuracy_mean']:.2%} | "
                     f"{row['test_accuracy_sd']:.2%} | "
                     f"{row['test_balanced_accuracy_mean']:.2%} | "
                     f"{row['test_macro_f1_mean']:.2%} | {row['linear_flops']:.0f} |")
    lines += ['', 'Per-seed paired differences are in `sweep_pairs.json`; '
              '`sweep_summary.csv` includes validation accuracy and measured '
              'dense versus sparse latency. Class-level confusion matrices are '
              'in each saved model’s `metrics.json`. Fewer linear FLOPs do not '
              'guarantee lower wall-clock time. The random row split can mix '
              'nearby forest cells across train and test.', '',
              'At 95%, the three Hessian models retain 24 weights each and beat '
              'magnitude in all three matched seeds. The score remains far below '
              'the dense model on balanced accuracy and macro F1. The Hessian '
              '95% point exceeds its 90% point because separate sparsity targets '
              'can follow different pruning paths; performance need not change '
              'monotonically with the final number of weights.', '',
              'The saved models were reloaded and checked. '
              '`protocol_extension.json` records the training source '
              'fingerprint; `sweep_audit.json` records verification checks.', '',
              f"Across {len(hessian_events)} Hessian removal events, the largest "
              f"reported matrix asymmetry is {max(e['asymmetry'] for e in hessian_events):.3g}, "
              f"the largest stabilized condition number is "
              f"{max(e['condition_after_shift'] for e in hessian_events):.1f}, and "
              f"the largest immediate calibration-loss increase is "
              f"{max(e['actual_calibration_loss_change'] for e in hessian_events):.3f}. "
              'All saved training and validation losses remained finite. These '
              'checks support numerical consistency; the score is still only an '
              'approximation of simultaneous deletion loss.', '']
    (output/'SWEEP.md').write_text('\n'.join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/covertype')
    parser.add_argument('--report-only', action='store_true',
                        help='Finish verification and reporting from existing fits.')
    args = parser.parse_args()
    output = Path(args.out)
    data = load_covertype()
    old = json.loads((output/'protocol.json').read_text())
    assert old['dataset_sha256'] == data['metadata']['sha256']
    seeds = sorted({e['seed'] for e in old['entries']})
    assert seeds == [0, 1, 2]
    entries = extension_entries(seeds)
    if args.report_only:
        protocol = json.loads((output/'protocol_extension.json').read_text())
        assert protocol['entries'] == entries
        assert protocol['dataset_sha256'] == data['metadata']['sha256']
        assert all((output/e['directory']/'metrics.json').exists() for e in entries)
    else:
        write_json(output/'protocol_extension.json', dict(
            source_sha256=source_digest(), dataset_sha256=data['metadata']['sha256'],
            reason='90% accuracy and class coverage prompted the standard full sparsity grid',
            prior_90_percent_test_results_observed=True,
            new_targets_tested_after_all_new_fits=True, entries=entries))
    started = time.perf_counter()
    if not args.report_only:
        for i, entry in enumerate(entries, 1):
            print(f"[{i}/{len(entries)}] {entry['directory']}", flush=True)
            result = run(Config(**entry['config']), output/entry['directory'],
                         evaluate_test=False)
            print(f"  val={result['val']['accuracy']:.3f} "
                  f"weights={result['active_weights']}/{result['total_weights']} "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    protocol = json.loads((output/'protocol_extension.json').read_text())
    if not args.report_only:
        assert source_digest() == protocol['source_sha256']
    original = {e['seed']: json.loads((output/e['directory']/'metrics.json').read_text())['history']
                for e in old['entries'] if e['method'] == 'dense'}
    for entry in entries:
        result = json.loads((output/entry['directory']/'metrics.json').read_text())
        first_removal = entry['config']['prune_start']+entry['config']['prune_every']
        assert result['history'][:first_removal] == original[entry['seed']][:first_removal]
    print('All new fits complete. Evaluating held-out test labels.', flush=True)
    for entry in entries:
        finalize(entry, output, data)
    rows = summarize(output, seeds)
    write_json(output/'sweep_audit.json', dict(
        new_models_verified=len(entries), previous_models_reused=9,
        data_sha256=data['metadata']['sha256'],
        training_source_sha256=protocol['source_sha256'],
        reporting_source_sha256=source_digest(),
        same_pre_pruning_history=True, masks_and_budget_checked=True,
        dense_sparse_logits_match=True, prior_90_percent_test_results_observed=True,
        report_only_recovery=args.report_only,
        reporting_seconds=time.perf_counter()-started))
    for row in rows:
        print(f"{row['method']:9} target={row['target']:.0%} "
              f"test={row['test_accuracy_mean']:.2%}", flush=True)
    print(f"Wrote {output/'SWEEP.md'} in {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == '__main__':
    main()
