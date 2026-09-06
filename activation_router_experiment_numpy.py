"""Controlled ComplexReLU vs adaptive activation-router experiment.

This file is deliberately self-contained.  The Work runner used for the first
experiment did not provide PyTorch, so a small NumPy reverse-mode autodiff
engine is included.  The model mirrors the repository's decoder-only design:
character embeddings, Fourier positions, causal multi-head self-attention,
pre-norm residual blocks, a 4x feed-forward layer, and an LM head.

Run:
    python activation_router_experiment_numpy.py

Outputs:
    experiment_results/activation_router_results.json
    experiment_results/activation_router_report.md
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiment_results"
SEED = 814
BLOCK_SIZE = 24
BATCH_SIZE = 6
N_EMBED = 24
N_HEAD = 4
N_LAYER = 2
STEPS = 160
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 0.01
EVAL_BATCHES = 16
ROUTER_TEMPERATURE = 1.0
TIMING_REPEATS = 5


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """Minimal vectorized reverse-mode tensor for this experiment."""

    def __init__(self, data, requires_grad=False, name="", children=()):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.name = name
        self._prev = tuple(children)
        self._backward = lambda: None

    def _add_grad(self, grad):
        if self.requires_grad:
            self.grad = grad if self.grad is None else self.grad + grad

    def __add__(self, other):
        other = as_tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad, children=(self, other))
        def backward():
            if out.grad is None: return
            self._add_grad(_unbroadcast(out.grad, self.data.shape))
            other._add_grad(_unbroadcast(out.grad, other.data.shape))
        out._backward = backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-as_tensor(other))

    def __rsub__(self, other):
        return as_tensor(other) - self

    def __mul__(self, other):
        other = as_tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad, children=(self, other))
        def backward():
            if out.grad is None: return
            self._add_grad(_unbroadcast(out.grad * other.data, self.data.shape))
            other._add_grad(_unbroadcast(out.grad * self.data, other.data.shape))
        out._backward = backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * as_tensor(other).pow(-1.0)

    def __matmul__(self, other):
        other = as_tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad, children=(self, other))
        def backward():
            if out.grad is None: return
            self._add_grad(_unbroadcast(out.grad @ np.swapaxes(other.data, -1, -2), self.data.shape))
            other._add_grad(_unbroadcast(np.swapaxes(self.data, -1, -2) @ out.grad, other.data.shape))
        out._backward = backward
        return out

    def pow(self, exponent):
        out = Tensor(self.data ** exponent, self.requires_grad, children=(self,))
        def backward():
            if out.grad is not None:
                self._add_grad(out.grad * exponent * (self.data ** (exponent - 1)))
        out._backward = backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, children=(self,))
        def backward():
            if out.grad is None: return
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                for ax in sorted((a % self.data.ndim for a in axes)):
                    grad = np.expand_dims(grad, ax)
            self._add_grad(np.broadcast_to(grad, self.data.shape))
        out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            denom = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            denom = math.prod(self.data.shape[a] for a in axes)
        return self.sum(axis=axis, keepdims=keepdims) / float(denom)

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), self.requires_grad, children=(self,))
        def backward():
            if out.grad is not None: self._add_grad(out.grad.reshape(self.data.shape))
        out._backward = backward
        return out

    def transpose(self, axes):
        out = Tensor(self.data.transpose(axes), self.requires_grad, children=(self,))
        inv = np.argsort(axes)
        def backward():
            if out.grad is not None: self._add_grad(out.grad.transpose(inv))
        out._backward = backward
        return out

    def __getitem__(self, index):
        out = Tensor(self.data[index], self.requires_grad, children=(self,))
        def backward():
            if out.grad is not None:
                grad = np.zeros_like(self.data)
                np.add.at(grad, index, out.grad)
                self._add_grad(grad)
        out._backward = backward
        return out

    def exp(self):
        clipped = np.clip(self.data, -30.0, 30.0)
        value = np.exp(clipped)
        out = Tensor(value, self.requires_grad, children=(self,))
        def backward():
            if out.grad is not None:
                active = (self.data >= -30.0) & (self.data <= 30.0)
                self._add_grad(out.grad * value * active)
        out._backward = backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), self.requires_grad, children=(self,))
        def backward():
            if out.grad is not None: self._add_grad(out.grad / self.data)
        out._backward = backward
        return out

    def log1p(self):
        out = Tensor(np.log1p(self.data), self.requires_grad, children=(self,))
        def backward():
            if out.grad is not None: self._add_grad(out.grad / (1.0 + self.data))
        out._backward = backward
        return out

    def tanh(self):
        value = np.tanh(self.data)
        out = Tensor(value, self.requires_grad, children=(self,))
        def backward():
            if out.grad is not None: self._add_grad(out.grad * (1.0 - value * value))
        out._backward = backward
        return out

    def abs(self):
        out = Tensor(np.abs(self.data), self.requires_grad, children=(self,))
        def backward():
            if out.grad is not None: self._add_grad(out.grad * np.sign(self.data))
        out._backward = backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, children=(self,))
        def backward():
            if out.grad is not None: self._add_grad(out.grad * (self.data > 0.0))
        out._backward = backward
        return out

    def backward(self):
        topo, seen = [], set()
        def build(node):
            if id(node) in seen: return
            seen.add(id(node))
            for child in node._prev: build(child)
            topo.append(node)
        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo): node._backward()


def as_tensor(value):
    return value if isinstance(value, Tensor) else Tensor(value)


def concat(tensors, axis=-1):
    tensors = [as_tensor(t) for t in tensors]
    out = Tensor(np.concatenate([t.data for t in tensors], axis=axis), any(t.requires_grad for t in tensors), children=tensors)
    widths = [t.data.shape[axis] for t in tensors]
    def backward():
        if out.grad is None: return
        start = 0
        for tensor, width in zip(tensors, widths):
            index = [slice(None)] * out.grad.ndim
            index[axis] = slice(start, start + width)
            tensor._add_grad(out.grad[tuple(index)])
            start += width
    out._backward = backward
    return out


def softmax(x, axis=-1):
    shifted = x - Tensor(np.max(x.data, axis=axis, keepdims=True))
    ex = shifted.exp()
    return ex / ex.sum(axis=axis, keepdims=True)


def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean).pow(2.0)).mean(axis=-1, keepdims=True)
    return (x - mean) / (variance + eps).pow(0.5) * weight + bias


class TinyGPTNumpy:
    def __init__(self, vocab_size, rng, variant):
        self.variant = variant
        self.params = {}
        self.routing_sums = np.zeros((N_LAYER, 3), dtype=np.float64)
        self.routing_hard = np.zeros((N_LAYER, 3), dtype=np.int64)
        self.routing_count = np.zeros(N_LAYER, dtype=np.int64)

        def param(name, shape, scale=0.02, zeros=False):
            data = np.zeros(shape, np.float32) if zeros else rng.normal(0, scale, shape).astype(np.float32)
            self.params[name] = Tensor(data, True, name=name)
        param("token_embedding", (vocab_size, N_EMBED))
        param("lm_head.w", (N_EMBED, vocab_size), scale=1 / math.sqrt(N_EMBED))
        param("lm_head.b", (vocab_size,), zeros=True)
        for layer in range(N_LAYER):
            p = f"blocks.{layer}"
            for ln in ("ln1", "ln2"):
                self.params[f"{p}.{ln}.w"] = Tensor(np.ones(N_EMBED, np.float32), True, name=f"{p}.{ln}.w")
                param(f"{p}.{ln}.b", (N_EMBED,), zeros=True)
            for name in ("q", "k", "v", "proj"):
                param(f"{p}.attn.{name}.w", (N_EMBED, N_EMBED), scale=1 / math.sqrt(N_EMBED))
                param(f"{p}.attn.{name}.b", (N_EMBED,), zeros=True)
            param(f"{p}.ff.fc1.w", (N_EMBED, 4 * N_EMBED), scale=1 / math.sqrt(N_EMBED))
            param(f"{p}.ff.fc1.b", (4 * N_EMBED,), zeros=True)
            param(f"{p}.ff.fc2.w", (4 * N_EMBED, N_EMBED), scale=1 / math.sqrt(4 * N_EMBED))
            param(f"{p}.ff.fc2.b", (N_EMBED,), zeros=True)
            # Per-hidden-dimension, input-conditioned logits: z_d*w_dc+b_dc.
            param(f"{p}.router.w", (4 * N_EMBED, 3), scale=0.01)
            param(f"{p}.router.b", (4 * N_EMBED, 3), zeros=True)
        self.params["ln.w"] = Tensor(np.ones(N_EMBED, np.float32), True, name="ln.w")
        param("ln.b", (N_EMBED,), zeros=True)

        position = np.arange(BLOCK_SIZE, dtype=np.float32)[:, None]
        div = np.exp(np.arange(0, N_EMBED, 2, dtype=np.float32) * (-math.log(10000.0) / N_EMBED))
        pe = np.zeros((BLOCK_SIZE, N_EMBED), np.float32)
        pe[:, 0::2] = np.sin(position * div)
        pe[:, 1::2] = np.cos(position * div)
        self.pe = Tensor(pe[None, :, :])

    def clone(self, variant):
        copied = object.__new__(TinyGPTNumpy)
        copied.variant = variant
        copied.params = {k: Tensor(v.data.copy(), True, name=k) for k, v in self.params.items()}
        copied.routing_sums = np.zeros_like(self.routing_sums)
        copied.routing_hard = np.zeros_like(self.routing_hard)
        copied.routing_count = np.zeros_like(self.routing_count)
        copied.pe = Tensor(self.pe.data.copy())
        return copied

    def linear(self, x, prefix):
        return x @ self.params[prefix + ".w"] + self.params[prefix + ".b"]

    def activation(self, z, layer, collect):
        if self.variant == "complex_relu":
            # Router-shaped parameters remain in the model for exact parameter
            # matching, but the baseline executes only the repository behavior.
            return z.relu()
        p = f"blocks.{layer}.router"
        logits = (z.reshape(*z.data.shape, 1) * self.params[p + ".w"] + self.params[p + ".b"]) / ROUTER_TEMPERATURE
        weights = softmax(logits, axis=-1)
        signed_log = Tensor(np.sign(z.data)) * z.abs().log1p()
        # Smoothly bounded exp: expm1(tanh(|z|)); finite output and derivative.
        stabilized_exp = Tensor(np.sign(z.data)) * (z.abs().tanh().exp() - 1.0)
        choices = concat([
            z.reshape(*z.data.shape, 1),
            signed_log.reshape(*z.data.shape, 1),
            stabilized_exp.reshape(*z.data.shape, 1),
        ], axis=-1)
        routed = (weights * choices).sum(axis=-1)
        if collect:
            flat = weights.data.reshape(-1, 3)
            self.routing_sums[layer] += flat.sum(axis=0)
            self.routing_hard[layer] += np.bincount(np.argmax(flat, axis=1), minlength=3)
            self.routing_count[layer] += flat.shape[0]
        return routed

    def forward(self, idx, targets=None, collect=False):
        batch, tokens = idx.shape
        x = self.params["token_embedding"][idx] + self.pe[:, :tokens, :]
        mask = np.triu(np.full((1, 1, tokens, tokens), -1e9, np.float32), 1)
        for layer in range(N_LAYER):
            p = f"blocks.{layer}"
            h = layer_norm(x, self.params[p + ".ln1.w"], self.params[p + ".ln1.b"])
            q = self.linear(h, p + ".attn.q").reshape(batch, tokens, N_HEAD, N_EMBED // N_HEAD).transpose((0, 2, 1, 3))
            k = self.linear(h, p + ".attn.k").reshape(batch, tokens, N_HEAD, N_EMBED // N_HEAD).transpose((0, 2, 1, 3))
            v = self.linear(h, p + ".attn.v").reshape(batch, tokens, N_HEAD, N_EMBED // N_HEAD).transpose((0, 2, 1, 3))
            scores = (q @ k.transpose((0, 1, 3, 2))) / math.sqrt(N_EMBED // N_HEAD) + Tensor(mask)
            attn = softmax(scores, axis=-1)
            context = (attn @ v).transpose((0, 2, 1, 3)).reshape(batch, tokens, N_EMBED)
            x = x + self.linear(context, p + ".attn.proj")
            h = layer_norm(x, self.params[p + ".ln2.w"], self.params[p + ".ln2.b"])
            z = self.linear(h, p + ".ff.fc1")
            x = x + self.linear(self.activation(z, layer, collect), p + ".ff.fc2")
        x = layer_norm(x, self.params["ln.w"], self.params["ln.b"])
        logits = self.linear(x, "lm_head")
        if targets is None:
            return logits, None
        flat = logits.reshape(batch * tokens, -1)
        probs = softmax(flat, axis=-1)
        selected = probs[np.arange(batch * tokens), targets.reshape(-1)]
        loss = -(selected + 1e-9).log().mean()
        return logits, loss


class AdamW:
    def __init__(self, params):
        self.params = list(params.values())
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            grad = p.grad
            self.m[i] = 0.9 * self.m[i] + 0.1 * grad
            self.v[i] = 0.999 * self.v[i] + 0.001 * grad * grad
            corrected_m = self.m[i] / (1.0 - 0.9 ** self.t)
            corrected_v = self.v[i] / (1.0 - 0.999 ** self.t)
            p.data -= LEARNING_RATE * (corrected_m / (np.sqrt(corrected_v) + 1e-8) + WEIGHT_DECAY * p.data)
            p.grad = None


def build_corpus():
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    selected = [p for p in tracked if p.endswith((".py", ".md")) and p != Path(__file__).name]
    chunks = []
    for relative in selected:
        chunks.append(f"\n# FILE: {relative}\n" + (ROOT / relative).read_text(encoding="utf-8", errors="replace"))
    text = "".join(chunks)
    chars = sorted(set(text))
    stoi = {char: i for i, char in enumerate(chars)}
    ids = np.asarray([stoi[char] for char in text], dtype=np.int64)
    split = int(0.9 * len(ids))
    return ids[:split], ids[split:], chars, selected, len(text)


def make_batches(tokens, count, rng):
    starts = rng.integers(0, len(tokens) - BLOCK_SIZE - 1, size=(count, BATCH_SIZE))
    x = np.stack([[tokens[s:s + BLOCK_SIZE] for s in row] for row in starts])
    y = np.stack([[tokens[s + 1:s + BLOCK_SIZE + 1] for s in row] for row in starts])
    return list(zip(x, y))


def evaluate(model, batches, collect=False):
    losses = []
    for x, y in batches:
        _, loss = model.forward(x, y, collect=collect)
        losses.append(float(loss.data))
    return float(np.mean(losses))


def train_one(model, train_batches, val_batches, verbose=True):
    optimizer = AdamW(model.params)
    losses, grad_norms, grad_maxes = [], [], []
    finite_steps = 0
    initial_val = evaluate(model, val_batches)
    started = time.perf_counter()
    for step, (x, y) in enumerate(train_batches, 1):
        _, loss = model.forward(x, y, collect=False)
        loss.backward()
        grads = [p.grad for p in model.params.values() if p.grad is not None]
        norm = math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads))
        max_abs = max(float(np.max(np.abs(g))) for g in grads)
        is_finite = np.isfinite(loss.data).all() and np.isfinite(norm) and np.isfinite(max_abs)
        finite_steps += int(is_finite)
        losses.append(float(loss.data))
        grad_norms.append(norm)
        grad_maxes.append(max_abs)
        if not is_finite:
            raise FloatingPointError(f"non-finite value at step {step}")
        optimizer.step()
        if verbose and (step == 1 or step % 40 == 0):
            print(f"{model.variant:16s} step={step:3d} loss={losses[-1]:.4f} grad_l2={norm:.3f}", flush=True)
    seconds = time.perf_counter() - started
    final_val = evaluate(model, val_batches, collect=True)
    if model.variant == "adaptive_router":
        soft = model.routing_sums / model.routing_count[:, None]
        hard = model.routing_hard / model.routing_count[:, None]
    else:
        soft = np.zeros((N_LAYER, 3), dtype=np.float64)
        hard = np.zeros((N_LAYER, 3), dtype=np.float64)
    return {
        "initial_validation_loss": initial_val,
        "final_train_loss": losses[-1],
        "mean_last_20_train_loss": float(np.mean(losses[-20:])),
        "final_validation_loss": final_val,
        "gradient_l2_mean": float(np.mean(grad_norms)),
        "gradient_l2_max": float(np.max(grad_norms)),
        "gradient_abs_max": float(np.max(grad_maxes)),
        "finite_step_fraction": finite_steps / STEPS,
        "training_seconds": seconds,
        "steps_per_second": STEPS / seconds,
        "routing_soft_frequencies_by_layer": soft.tolist(),
        "routing_hard_frequencies_by_layer": hard.tolist(),
        "loss_curve": losses,
        "gradient_l2_curve": grad_norms,
    }


def render_report(result):
    b = result["variants"]["complex_relu"]
    a = result["variants"]["adaptive_router"]
    labels = ["linear", "signed_log", "stabilized_exp"]
    rows = []
    for layer, freq in enumerate(a["routing_soft_frequencies_by_layer"], 1):
        hard = a["routing_hard_frequencies_by_layer"][layer - 1]
        rows.append("| " + str(layer) + " | " + " | ".join(f"{100*x:.2f}%" for x in freq) + " | " + " / ".join(f"{100*x:.2f}%" for x in hard) + " |")
    delta = 100 * (a["final_validation_loss"] - b["final_validation_loss"]) / b["final_validation_loss"]
    speed = 100 * (a["training_seconds"] - b["training_seconds"]) / b["training_seconds"]
    return f"""# Adaptive activation-router experiment

This report contains results from an actual CPU run of `activation_router_experiment_numpy.py`. The repository's ignored Blender documentation corpus was unavailable, so the frozen dataset was built from the {result['data']['tracked_files']} tracked Python/Markdown files that existed before this experiment ({result['data']['characters']} characters, {result['data']['vocab_size']}-character vocabulary). Results therefore test optimization on repository text; they do **not** establish Blender code quality.

## Controlled setup

- Seed {SEED}; {STEPS} AdamW steps; learning rate {LEARNING_RATE}; weight decay {WEIGHT_DECAY}
- Context {BLOCK_SIZE}; batch {BATCH_SIZE}; embedding {N_EMBED}; heads {N_HEAD}; layers {N_LAYER}; FFN width {4*N_EMBED}
- Same initial tensors, pre-generated train/validation batches, and exact parameter count ({result['model']['parameter_count']:,}); router-shaped parameters are dormant in the baseline so timing measures the router's real compute overhead
- Training speed is the median of {TIMING_REPEATS} complete runs with alternating execution order
- Baseline output: existing ComplexReLU behavior (`ReLU(real)` + `ReLU(imag)`, algebraically ReLU over the concatenated hidden vector)
- Adaptive output: per-value soft selection among {', '.join(labels)} using learned per-hidden-dimension affine router logits
- Stabilized exponential: `sign(z) * expm1(tanh(abs(z)))`, bounded to prevent exponential overflow

## Results

| Metric | ComplexReLU | Adaptive router |
|---|---:|---:|
| Initial validation loss | {b['initial_validation_loss']:.4f} | {a['initial_validation_loss']:.4f} |
| Final train loss | {b['final_train_loss']:.4f} | {a['final_train_loss']:.4f} |
| Mean final-20 train loss | {b['mean_last_20_train_loss']:.4f} | {a['mean_last_20_train_loss']:.4f} |
| Final validation loss | {b['final_validation_loss']:.4f} | {a['final_validation_loss']:.4f} |
| Mean gradient L2 | {b['gradient_l2_mean']:.4f} | {a['gradient_l2_mean']:.4f} |
| Peak gradient L2 | {b['gradient_l2_max']:.4f} | {a['gradient_l2_max']:.4f} |
| Peak gradient element | {b['gradient_abs_max']:.4f} | {a['gradient_abs_max']:.4f} |
| Finite steps | {100*b['finite_step_fraction']:.1f}% | {100*a['finite_step_fraction']:.1f}% |
| CPU training time | {b['training_seconds']:.2f} s | {a['training_seconds']:.2f} s |
| Steps/second | {b['steps_per_second']:.2f} | {a['steps_per_second']:.2f} |

Adaptive validation-loss change versus baseline: **{delta:+.2f}%** (negative is better). Adaptive wall-time change: **{speed:+.2f}%**.

## Adaptive routing frequencies

Soft frequencies are mean probability mass. The final column gives hard argmax frequencies in `linear / signed_log / stabilized_exp` order.

| Layer | Linear (soft) | Signed-log (soft) | Stabilized-exp (soft) | Hard argmax frequencies |
|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Interpretation limits

This is a smoke-scale, single-seed experiment on a substitute corpus and a reduced model. A difference here is evidence that the implementations trained differently under matched conditions, not evidence that one activation is generally superior. A publishable conclusion needs the actual Blender corpus, multiple seeds, a PyTorch replication at the repository's 128-wide/4-layer configuration, and confidence intervals.
"""


def main():
    train_tokens, val_tokens, chars, files, character_count = build_corpus()
    batch_rng = np.random.default_rng(SEED + 1)
    train_batches = make_batches(train_tokens, STEPS, batch_rng)
    val_batches = make_batches(val_tokens, EVAL_BATCHES, batch_rng)
    initialized = TinyGPTNumpy(len(chars), np.random.default_rng(SEED), "complex_relu")
    baseline = initialized.clone("complex_relu")
    adaptive = initialized.clone("adaptive_router")
    count_b = sum(p.data.size for p in baseline.params.values())
    count_a = sum(p.data.size for p in adaptive.params.values())
    assert count_b == count_a
    results = {
        "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": "NumPy self-contained reverse-mode autodiff on CPU",
        "data": {"characters": character_count, "train_tokens": len(train_tokens), "validation_tokens": len(val_tokens), "vocab_size": len(chars), "tracked_files": len(files), "files": files},
        "model": {"parameter_count": count_b, "block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE, "n_embed": N_EMBED, "n_head": N_HEAD, "n_layer": N_LAYER, "steps": STEPS, "seed": SEED},
        "variants": {},
    }
    for model in (baseline, adaptive):
        results["variants"][model.variant] = train_one(model, train_batches, val_batches)
    timing = {
        "complex_relu": [results["variants"]["complex_relu"]["training_seconds"]],
        "adaptive_router": [results["variants"]["adaptive_router"]["training_seconds"]],
    }
    # Add four fresh complete runs per arm and alternate order to reduce warm-up
    # and order bias. Losses are deterministic; only timing is aggregated.
    for repeat in range(1, TIMING_REPEATS):
        order = ("adaptive_router", "complex_relu") if repeat % 2 else ("complex_relu", "adaptive_router")
        for variant in order:
            fresh = initialized.clone(variant)
            rerun = train_one(fresh, train_batches, val_batches, verbose=False)
            timing[variant].append(rerun["training_seconds"])
            reference = results["variants"][variant]
            if not np.isclose(rerun["final_validation_loss"], reference["final_validation_loss"], rtol=0, atol=1e-6):
                raise AssertionError("deterministic validation loss did not reproduce")
    for variant, samples in timing.items():
        median = float(np.median(samples))
        results["variants"][variant]["training_seconds_repeats"] = samples
        results["variants"][variant]["training_seconds"] = median
        results["variants"][variant]["steps_per_second"] = STEPS / median
    OUT.mkdir(exist_ok=True)
    (OUT / "activation_router_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "activation_router_report.md").write_text(render_report(results), encoding="utf-8")
    print(f"wrote {OUT / 'activation_router_results.json'}")
    print(f"wrote {OUT / 'activation_router_report.md'}")


if __name__ == "__main__":
    main()
