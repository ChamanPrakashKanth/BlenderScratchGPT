# Adaptive activation-router experiment

This report contains results from an actual CPU run of `activation_router_experiment_numpy.py`. The repository's ignored Blender documentation corpus was unavailable, so the frozen dataset was built from the 13 tracked Python/Markdown files that existed before this experiment (28035 characters, 96-character vocabulary). Results therefore test optimization on repository text; they do **not** establish Blender code quality.

## Controlled setup

- Seed 814; 160 AdamW steps; learning rate 0.002; weight decay 0.01
- Context 24; batch 6; embedding 24; heads 4; layers 2; FFN width 96
- Same initial tensors, pre-generated train/validation batches, and exact parameter count (20,352); router-shaped parameters are dormant in the baseline so timing measures the router's real compute overhead
- Training speed is the median of 5 complete runs with alternating execution order
- Baseline output: existing ComplexReLU behavior (`ReLU(real)` + `ReLU(imag)`, algebraically ReLU over the concatenated hidden vector)
- Adaptive output: per-value soft selection among linear, signed_log, stabilized_exp using learned per-hidden-dimension affine router logits
- Stabilized exponential: `sign(z) * expm1(tanh(abs(z)))`, bounded to prevent exponential overflow

## Results

| Metric | ComplexReLU | Adaptive router |
|---|---:|---:|
| Initial validation loss | 5.3784 | 5.4147 |
| Final train loss | 2.9571 | 2.9396 |
| Mean final-20 train loss | 2.8580 | 2.8599 |
| Final validation loss | 3.1509 | 3.1110 |
| Mean gradient L2 | 1.1071 | 1.1180 |
| Peak gradient L2 | 4.1955 | 3.8663 |
| Peak gradient element | 0.7375 | 0.7162 |
| Finite steps | 100.0% | 100.0% |
| CPU training time | 0.60 s | 1.58 s |
| Steps/second | 266.12 | 101.13 |

Adaptive validation-loss change versus baseline: **-1.27%** (negative is better). Adaptive wall-time change: **+163.16%**.

## Adaptive routing frequencies

Soft frequencies are mean probability mass. The final column gives hard argmax frequencies in `linear / signed_log / stabilized_exp` order.

| Layer | Linear (soft) | Signed-log (soft) | Stabilized-exp (soft) | Hard argmax frequencies |
|---:|---:|---:|---:|---:|
| 1 | 33.59% | 32.94% | 33.47% | 38.10% / 39.05% / 22.84% |
| 2 | 33.19% | 33.51% | 33.30% | 28.31% / 46.11% / 25.58% |

## Interpretation limits

This is a smoke-scale, single-seed experiment on a substitute corpus and a reduced model. A difference here is evidence that the implementations trained differently under matched conditions, not evidence that one activation is generally superior. A publishable conclusion needs the actual Blender corpus, multiple seeds, a PyTorch replication at the repository's 128-wide/4-layer configuration, and confidence intervals.
