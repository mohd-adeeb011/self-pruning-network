# UCI Covertype: focused larger-data comparison

Original rows: 581,012; 54 inputs; 7 classes.
Split: {'train': 348605, 'val': 116202, 'test': 116205}. Source SHA-256: `0a9371cef7c964b5475d6053cc3e0894a5aa6f65ad1ed3ecb01c45aa96217945`. All three methods use the same initialization, batches, model width (8), 60 epochs and 90% target. The existing score, damping and masking code is unchanged. Test labels were evaluated after all fits finished.

| Method | Active weights | Train accuracy | Validation accuracy | Test accuracy | Test SD | Balanced accuracy | Macro F1 | Linear FLOPs/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 488/488 | 76.18% | 75.82% | 75.94% | 0.18% | 57.52% | 60.44% | 991 |
| magnitude | 48/488 | 52.57% | 52.59% | 52.64% | 0.96% | 22.13% | 18.97% | 111 |
| hessian | 48/488 | 63.27% | 63.04% | 63.12% | 7.76% | 34.53% | 31.87% | 111 |

Paired Hessian minus magnitude test accuracy: +10.48 percentage points. Per-seed values are in `paired.json`. Three seeds on one random split cannot establish general superiority.

FLOPs count linear arithmetic only. `summary.csv` includes measured dense and sparse wall-clock time; sparse Python loops can be slower. The random row split can mix nearby forest cells across splits, so it may overstate performance on a new geographical area. This 90% comparison does not replace the full sparsity sweep required for submission.
