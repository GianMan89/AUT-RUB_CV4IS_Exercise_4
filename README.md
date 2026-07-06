# CV4IS Exercise 4 - Robustness under Distribution Shift

This repository contains a practical exercise for the course
**Computer Vision for Industrial Systems** at Ruhr University Bochum.

The exercise follows Lecture 5, *Drift, Shift, and Robustness in Industrial
Computer Vision*, and uses the evaluation logic introduced by Hendrycks and
Dietterich: apply controlled corruption families at increasing severity,
measure degradation, and report robustness profiles rather than a single
clean-test number.

The corruption implementations in this exercise are intentionally lightweight
teaching approximations. They reproduce the **structured evaluation idea**
of common-corruption benchmarks, but they are not exact replicas of CIFAR-10-C.

## Learning goals

Students will:

1. distinguish clean evaluation, robustness testing, and live drift monitoring,
2. implement severity-controlled image corruptions,
3. evaluate a fixed classifier across corruption families and severities,
4. measure accuracy, macro-F1, confidence, entropy, and performance drop,
5. create robustness curves, heatmaps, and scalar summaries,
6. inspect corruption-induced failure cases,
7. monitor a simulated image stream with window-level statistics and KS tests,
8. implement a persistent drift alarm and discuss its limitations.

## Dataset

The notebooks use **CIFAR-10**, a real public image dataset with 60,000 RGB
images in ten classes. The first notebook downloads it through
`torchvision.datasets.CIFAR10(..., download=True)`.

Official sources:

- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- TorchVision dataset interface:
  https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html

CIFAR-10 is not an industrial inspection dataset. It is used here because it
is public, compact, reproducible, and suitable for a two-hour CPU-friendly
robustness lab. The engineering workflow transfers directly to defect
classification, OCR, detection, segmentation, and other industrial tasks.

## Notebooks

| Notebook | Main activity |
|---|---|
| `notebooks/01_common_corruptions_student.ipynb` | severity-controlled stress testing |
| `notebooks/02_stream_monitoring_student.ipynb` |  window-level drift monitoring |
| `notebooks/solutions/01_common_corruptions_solution.ipynb` |  solution reference |
| `notebooks/solutions/02_stream_monitoring_solution.ipynb` |  solution reference |

## Important engineering caveats

- Corruption severity is a test design choice, not a universal physical unit.
- A corruption must preserve the intended label for the evaluation to remain valid.
- Clean accuracy and corruption robustness may rank models differently.
- Confidence can remain high while accuracy collapses.
- A drift alarm indicates that investigation is needed. It does not identify
  the cause and does not prove that performance has degraded.
- Acceptance criteria should be defined from operational risk before examining
  final test results.
- Keep validation choices separate from final test reporting.

## References

- Hendrycks, D., Dietterich, T. *Benchmarking Neural Network Robustness to
  Common Corruptions and Perturbations*. ICLR 2019.
  https://arxiv.org/abs/1903.12261
- Krizhevsky, A. *Learning Multiple Layers of Features from Tiny Images*, 2009.
- Quiñonero-Candela et al. *Dataset Shift in Machine Learning*, 2009.
