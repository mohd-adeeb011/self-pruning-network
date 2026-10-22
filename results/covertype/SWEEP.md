# UCI Covertype sparsity sweep

This is an exploratory extension after the 90% test results were already observed. The added 50%, 75%, and 95% targets follow the assignment grid. No claim of preregistration or an untouched test set applies to the combined sweep. The same dataset split and three paired seeds were used throughout.

The normal, unpruned dense model has **75.94% mean test accuracy** (sample SD 0.18%) across the three seeds. It keeps all 488 connection weights and 15 biases.

| Method | Target | Active weights | Test accuracy | Sample SD | Balanced accuracy | Macro F1 | Linear FLOPs/example |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal dense (unpruned) | 0% | 488/488 | 75.94% | 0.18% | 57.52% | 60.44% | 991 |
| magnitude | 50% | 244/488 | 73.29% | 0.33% | 53.03% | 54.93% | 503 |
| hessian | 50% | 244/488 | 74.39% | 0.25% | 53.17% | 55.09% | 503 |
| magnitude | 75% | 122/488 | 69.49% | 0.59% | 43.89% | 44.19% | 259 |
| hessian | 75% | 122/488 | 71.44% | 0.59% | 44.45% | 44.59% | 259 |
| magnitude | 90% | 48/488 | 52.64% | 0.96% | 22.13% | 18.97% | 111 |
| hessian | 90% | 48/488 | 63.12% | 7.76% | 34.53% | 31.87% | 111 |
| magnitude | 95% | 24/488 | 49.14% | 0.66% | 15.80% | 11.16% | 63 |
| hessian | 95% | 24/488 | 67.20% | 0.33% | 34.90% | 34.03% | 63 |

Per-seed paired differences are in `sweep_pairs.json`; `sweep_summary.csv` includes validation accuracy and measured dense versus sparse latency. Class-level confusion matrices are in each saved model’s `metrics.json`. Fewer linear FLOPs do not guarantee lower wall-clock time. The random row split can mix nearby forest cells across train and test.

At 95%, the three Hessian models retain 24 weights each and beat magnitude in all three matched seeds. The score remains far below the dense model on balanced accuracy and macro F1. The Hessian 95% point exceeds its 90% point because separate sparsity targets can follow different pruning paths; performance need not change monotonically with the final number of weights.

The saved models were reloaded and checked. `protocol_extension.json` records the training source fingerprint; `sweep_audit.json` records verification checks.

Across 72 Hessian removal events, the largest reported matrix asymmetry is 3.57e-08, the largest stabilized condition number is 1112.3, and the largest immediate calibration-loss increase is 0.726. All saved training and validation losses remained finite. These checks support numerical consistency; the score is still only an approximation of simultaneous deletion loss.
