# Teacher notes - Exercise 4

## Purpose

This exercise turns Lecture 5 into a compact engineering workflow:

1. establish a fixed clean baseline,
2. define deployment-relevant perturbation families,
3. evaluate a family-by-severity grid,
4. aggregate degradation without hiding the worst family,
5. inspect failures,
6. connect pre-deployment stress testing to live monitoring.

The central message is that a clean test score is conditional on the tested
acquisition conditions.

## Recommended introduction

Use 10-15 minutes to recap:

- `P_train(x, y) != P_deploy(x, y)`,
- robustness testing is active and controlled,
- monitoring is passive and window-based,
- `Delta M_(p,s) = M_clean - M_(p,s)`,
- stochastic corruptions may need repeated draws,
- alerts trigger investigation rather than automatic retraining.

## Notebook 1 priorities

All students should complete:

- Gaussian noise,
- blur,
- pixelation,
- predictive entropy,
- the corruption evaluation loop,
- normalized robustness AUC,
- the final interpretation.

The failure-case inspection is valuable if time allows.

## Notebook 2 priorities

All students should complete:

- brightness,
- sharpness,
- KS comparison against a reference window,
- persistent alert logic,
- the monitoring limitations discussion.

## Expected qualitative observations

Exact values vary because the model is trained during the session.

Typical patterns:

- performance decreases with severity,
- different corruption families degrade at different rates,
- pixelation and strong blur often damage small 32x32 images quickly,
- confidence does not necessarily decrease as fast as accuracy,
- entropy is useful but is not a guaranteed failure detector,
- raw brightness may miss blur, while sharpness catches it,
- an input drift signal can appear before or without a large measured accuracy drop,
- a single signal rarely explains the root cause.

## Common student difficulties

- applying normalization before the PIL corruption,
- forgetting to clip noisy pixels to `[0, 1]`,
- interpreting severity as a physically calibrated quantity,
- averaging all corruption families and hiding one severe weakness,
- treating lower confidence as equivalent to lower accuracy,
- using current labels in monitoring without acknowledging that labels may be delayed,
- defining an alert threshold after looking at the incident period.

## Runtime controls

Defaults are designed for ordinary CPU machines:

- 15,000 training images,
- 2,000 validation images,
- 2,000 test images,
- five epochs,
- 800 stress-test images,
- two Gaussian-noise draws and one draw for deterministic corruptions.

If runtime is tight:

- set `TRAIN_SAMPLES = 8000`,
- set `EPOCHS = 3`,
- set `STRESS_TEST_SAMPLES = 400`,
- keep only three corruption families,
- treat Notebook 2 as an instructor-guided group exercise.

## Discussion prompts

1. Which corruption family is most plausible for a specific industrial camera setup?
2. Which severity values are still label-preserving?
3. Would missed defects or false rejects dominate the risk?
4. Is average robustness sufficient, or is worst-family robustness more meaningful?
5. What would need to change for detection or segmentation?
6. What can be monitored without labels?
7. When would retraining be the wrong response?
