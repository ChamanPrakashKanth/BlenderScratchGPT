# Adaptive activation-router experiment

This report contains results from an actual CPU run of `activation_router_experiment_numpy.py`. The repository's ignored Blender documentation corpus was unavailable, so the frozen dataset was built from the 13 tracked Python/Markdown files that existed before this experiment (28035 characters, 96-character vocabulary). Results therefore test optimization on repository text; they do **not** establish Blender code quality.

## Controlled setup

- Seed 814; 160 AdamW steps; learning rate 0.002; weight decay 0.01
- Context 24; batch 6; embedding 24; heads 4; layers 2; FFN width 96
- Same initial tensors, pre-generated train/validation batches, and exact parameter count (20,352); router-shaped parameters are dormant in the baseline so timing measures the router's real compute overhead
- Training speed is the median of 5 complete runs with alternating execution order
- Baseline output: existing ComplexReLU behavior (`ReLU(real)` + `ReLU(imag)`, algebraically ReLU over the concatenated hidden vector)
- Adaptive output: per-value soft selection among linear, signed_log, stabilized_exp using learned per-hidden-dimension affine router logits
- Composed output: adaptive routing first, followed by ComplexReLU on the routed hidden vector
- Stabilized exponential: `sign(z) * expm1(tanh(abs(z)))`, bounded to prevent exponential overflow

## Results

| Metric | ComplexReLU | Adaptive router | ComplexReLU over router |
|---|---:|---:|---:|
| Initial validation loss | 5.3784 | 5.4147 | 5.3998 |
| Final train loss | 2.9571 | 2.9396 | 2.9577 |
| Mean final-20 train loss | 2.8580 | 2.8599 | 2.8622 |
| Final validation loss | 3.1509 | 3.1110 | 3.1474 |
| Mean gradient L2 | 1.1071 | 1.1180 | 1.0550 |
| Peak gradient L2 | 4.1955 | 3.8663 | 4.2634 |
| Peak gradient element | 0.7375 | 0.7162 | 0.7556 |
| Finite steps | 100.0% | 100.0% | 100.0% |
| CPU training time | 0.61 s | 1.60 s | 1.60 s |
| Steps/second | 262.58 | 99.85 | 99.86 |

Adaptive validation-loss change versus baseline: **-1.27%** (negative is better). Adaptive wall-time change: **+162.97%**. ComplexReLU-over-router validation-loss change: **-0.11%**.

## Adaptive routing frequencies

Soft frequencies are mean probability mass for the composed model's adaptive path. The final column gives hard argmax frequencies in `linear / signed_log / stabilized_exp` order.

| Layer | Linear (soft) | Signed-log (soft) | Stabilized-exp (soft) | Hard argmax frequencies |
|---:|---:|---:|---:|---:|
| 1 | 33.76% | 32.75% | 33.49% | 38.10% / 37.03% / 24.87% |
| 2 | 33.55% | 33.01% | 33.44% | 31.27% / 41.34% / 27.39% |

## Interpretation limits

This is a smoke-scale, single-seed experiment on a substitute corpus and a reduced model. A difference here is evidence that the implementations trained differently under matched conditions, not evidence that one activation is generally superior. A publishable conclusion needs the actual Blender corpus, multiple seeds, a PyTorch replication at the repository's 128-wide/4-layer configuration, and confidence intervals.
