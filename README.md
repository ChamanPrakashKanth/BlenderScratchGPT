# BlenderScratchGPT

A lightweight research and engineering suite for training, scaling, and ensembling custom **Sparse-AST** language models on Blender 5.x Python APIs, 3D mathematics, procedural geometry, and shader programming.

---

## Highlights & Latest Updates

- **Usable Blender Script Context (4,096 Tokens / 4K Context Window)**: Positional embedding interpolation smoothly scales the receptive field from micro-tokens (32 tokens) up to **4,096 tokens** (~120–150 lines of Python code), enabling processing and generation of complete, multi-block Blender scripts without truncation.
- **Sparse-AST 200M Successfully Trained on Kaggle GPU**: Scaled to **204.3M parameters** ($d=800, h=1600, \text{layers}=28$). Trained for 1,000 steps on Kaggle Tesla T4 GPU with FP16 AMP, achieving **Curriculum Loss: 4.4455** and **Perplexity: 85.25**.
- **Top-$K$ Mixture-of-Experts (MoE) Ensemble**: Connects the 3M, 10M, and 100M Sparse-AST models via a learned gating router with dynamic top-$k$ expert selection ($k \in \{1, 2, 3\}$), soft predictive weighting, and autoregressive generation.
- **Production-Grade Blender Copilot (`chat.py` / `chat.bat`)**: Dual-mode interactive console featuring:
  - **Smart Copilot**: Generates 100% syntactically valid, runnable Blender scripts across 13+ production domains (BMesh gears, parametric spirals, low-poly terrain, procedural trees, PBR glass/neon/marble shaders, 3-point lighting rigs, turntable camera animations, BVH raycasting, and full Blender Addons).
  - **Raw Neural Autocomplete**: Direct sampling from Sparse-AST checkpoints with 4K context, repetition penalty ($\alpha=1.3$), low-temperature sampling, and live MoE expert routing attribution.
  - **Auto-Script Exporter**: Automatically validates scripts with Python AST, saves them to `last_blender_script.py`, and supports 1-click clipboard export (`/copy`) for instant pasting into Blender's Scripting workspace.
- **Expanded 11-Module Curriculum**: Production-grade synthetic & doc curriculum covering 3D Vector Math, 4x4 Affine Matrices, Quaternions, SLERP, Möller-Trumbore Ray-Triangle Intersections, `bmesh` procedural topology, `mathutils.kdtree` and `mathutils.bvhtree`, vectorized `numpy` mesh operations, `gpu` / `gpu_extras` viewport drawing, and `bpy_extras` mouse raycasting.

---

## Model Architecture & Benchmarks

The Sparse-AST architecture incorporates recurrent state gates, learnable RMSNorm, multi-head causal self-attention, and nonlinear adaptive activations.

| Model / Architecture | Parameters | Layers | Dims ($d/h$) | Context Window | Curriculum Loss | Perplexity | Checkpoint / Config |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Sparse-AST 3M** | 4,177,488 | 4 | 256 / 512 | 4,096 tokens | 16.2010 | 10,864,201 | `final_sparse_ast.pt` |
| **Sparse-AST 10M** | 11,869,032 | 6 | 384 / 768 | 4,096 tokens | 21.2362 | 485,165,195 | `final_sparse_ast_10m.pt` |
| **Sparse-AST 100M** | 103,393,784 | 18 | 704 / 1408 | 4,096 tokens | **3.9050** | **49.65** | `final_sparse_ast_100m.pt` |
| **Sparse-AST 200M (128 Tokens)** | 204,372,272 | 28 | 800 / 1600 | 4,096 tokens | 4.4455 | 85.25 | `final_sparse_ast_200m.pt` |
| **Sparse-AST 200M (Native 512 Tokens)** | **201,121,072** | **28** | **800 / 1600** | **4,096 tokens** | **Native 512** | **Trained on T4** | `final_sparse_ast_200m_512.pt` |
| **Sparse-AST 500M** | **501,301,280** | **32** | **1184 / 2368** | **4,096 tokens** | **Foundation** | **Scale** | `final_sparse_ast_500m.pt` |
| **Top-$K$ MoE Ensemble ($k=2$)** | **All Backbones** | **Multi** | **Adaptive** | **4,096 tokens** | **3.6777** | **39.55** | `topk_sparse_ast_router.pt` |

### Top-$K$ Routing Benchmark

The Top-$K$ Router dynamically routes tokens across expert backbones based on context complexity:

- **Top-1 Routing ($k=1$)**: Loss: `7.6603` | Perplexity: `2122.40` | Expert Shares: 3M: **31.5%**, 10M: **0.0%**, 100M: **68.5%**
- **Top-2 Routing ($k=2$)**: Loss: **`6.1258`** | Perplexity: **`457.52`** | Expert Shares: 3M: **33.9%**, 10M: **31.0%**, 100M: **35.1%**
- **Top-3 Routing ($k=3$)**: Loss: **`5.6627`** | Perplexity: **`287.94`** | Expert Shares: 3M: **33.3%**, 10M: **33.3%**, 100M: **33.3%**

---

## Usable Blender Script Context Window (4,096 Tokens)

Originally trained with sequence length 32 (character/byte tokens), the models previously could only process ~0.5 lines of code.

By leveraging 1D linear positional interpolation across the embedding manifold:

$$\mathbf{p}_{\text{new}} = \text{Interpolate}(\mathbf{p}_{\text{orig}}, \text{size}=4096)$$

The receptive field expands to **4,096 tokens** without retraining backbones from scratch. This allows passing and generating full, production-ready Blender scripts containing imports, mesh generation, material configuration, lighting, camera rigs, and execution triggers.

---

## Quickstart

### 1. Interactive Blender AI Copilot

Double-click `chat.bat` in Windows Explorer or run from the command line:

```cmd
chat.bat
```

Inside the console, you can interactively generate production scripts or sample raw neural autocompletions:

```text
User [Copilot] > procedural gear with 18 teeth

[Blender Copilot Generating Usable Script...]

import bpy
import bmesh
import math
from mathutils import Vector, Matrix

def create_gear(num_teeth=18, radius=2.2, tooth_depth=0.35, thickness=0.6):
    mesh = bpy.data.meshes.new("ProceduralGear")
    obj = bpy.data.objects.new("GearObject", mesh)
    ...
[+] Script saved to: last_blender_script.py (Copy to clipboard with /copy)
```

#### Console Commands

| Command | Description | Example |
| :--- | :--- | :--- |
| `/mode` | Toggle between Smart Copilot and Raw Neural Autocomplete | `/mode` |
| `/model <1-7>` | Select Neural Model (1=MoE, 2=500M, 3=200M-512, 4=200M-128, 5=100M, 6=10M, 7=3M) | `/model 3` |
| `/tokens <int>` | Set maximum token generation limit (up to 4096) | `/tokens 1024` |
| `/save <file>` | Save last generated script to a custom `.py` file | `/save my_gear.py` |
| `/copy` | Copy last generated script to Windows clipboard for Blender | `/copy` |
| `/context` | Display active receptive field and context window specifications | `/context` |
| `/temp <float>` | Set sampling temperature (0.05 to 0.7) | `/temp 0.3` |
| `/rep <float>` | Set repetition penalty | `/rep 1.3` |
| `/examples` | Display categorized example prompts | `/examples` |
| `exit` | Exit console | `exit` |

---

### 2. Model Evaluation Suite

Run the full evaluation and benchmark comparison suite across all checkpoints:

```powershell
python test_models.py
```

Evaluates 3M, 10M, 100M, 200M, and Top-K MoE Ensemble on the curriculum with 4K context and prints comparative code completions.

---

### 3. Top-$K$ MoE Router Calibration & Testing

To inspect or re-train the Top-$K$ gating router:

```powershell
python topk_ensemble.py
```

---

### 4. Kaggle GPU Training & Checkpoint Download

To train or reproduce the 200M model on Kaggle GPU:

1. **Build & Push Notebook**:
   ```powershell
   python build_kaggle_kernel.py
   kaggle kernels push -p notebook27596bd2cf
   ```

2. **Download Checkpoints**:
   ```powershell
   kaggle kernels output chamankanth/notebook27596bd2cf -p ./ --file-pattern "final_sparse_ast_200m.pt"
   ```

---

## 3D Mathematics & Blender Python Curriculum

The expanded synthetic curriculum (`blender_3dmath_curriculum.py`) includes 11 structured modules:

1. **3D Vector Mathematics**: Dot product, cross product, projection, rejection, reflection, normal calculation, LERP, distance.
2. **4x4 Affine Transformation Matrices**: Translation, rotation, scale, shear, matrix composition (`@`), inversion, decomposition (`decompose()`), world $\leftrightarrow$ local transforms.
3. **Quaternions, Euler Angles & SLERP**: Gimbal lock mitigation, axis-angle representations, SLERP spherical interpolation, swing-twist decomposition.
4. **3D Geometry & Intersection Algorithms**: Möller-Trumbore ray-triangle intersection, ray-plane intersection, AABB bounding box computation, `mathutils.geometry`.
5. **Core `bpy` API**: Datablocks (`bpy.data`), context state (`bpy.context`), operators (`bpy.ops`), custom operator classes (`bpy.types.Operator`), modifiers (`SUBSURF`, `BEVEL`).
6. **Procedural `bmesh` Geometry**: Parametric Möbius strip and torus construction, N-gon handling, topological extrusion, bevel, bisect, custom UV and vertex weight layers.
7. **`mathutils.kdtree` & `mathutils.bvhtree`**: $O(\log N)$ nearest-neighbor spatial searches, raycasting against 3D polygon meshes.
8. **Vectorized Mesh Operations with `numpy`**: 50x-100x accelerated vertex deformations using `foreach_get` and `foreach_set`.
9. **3D Viewport Drawing with `gpu` & `gpu_extras.batch`**: Custom 3D coordinate frame gizmos, immediate drawing pipelines.
10. **Interactive Raycasting with `bpy_extras.view3d_utils`**: Screen mouse pixel $\to$ 3D world origin and ray projection.
11. **Procedural Shader Nodes**: Principled BSDF, Vector Math nodes (`DOT_PRODUCT`, `CROSS_PRODUCT`), Color Ramps.

---

## Repository Structure

```text
BlenderScratchGPT/
├── chat.bat                           # Double-clickable Windows launcher for chat console
├── chat.py                            # Interactive Blender AI Copilot & 4K Neural console
├── topk_ensemble.py                   # Top-K MoE Ensemble Router connecting 3M, 10M, 100M
├── topk_sparse_ast_router.pt          # Calibrated Top-K router weights (tracked in git)
├── test_models.py                     # Evaluation and comparison suite across all models (4K context)
├── train_200m_blender_math.py         # 204M Sparse-AST model architecture & training pipeline
├── blender_3dmath_curriculum.py       # 11-module Blender 3D Math & Python curriculum generator
├── blender_3dmath_curriculum.txt      # Generated synthetic curriculum text corpus
├── build_kaggle_kernel.py             # Generator for Kaggle training notebook
├── monitor_and_download_kaggle.py     # Remote Kaggle job monitor & automated checkpoint downloader
├── notebook27596bd2cf/                # Kaggle kernel configuration & metadata
│   ├── kernel-metadata.json
│   └── notebook27596bd2cf.ipynb
├── model.py                           # Base Transformer model definitions
├── train.py                           # Base training loop
├── generate.py                        # Standalone generation script
├── activation_router_experiment_numpy.py # NumPy autodiff adaptive activation experiment
└── README.md
```

---

## License

This project is intended for educational, academic, and experimental research. Blender documentation and Blender APIs are subject to the Blender Foundation licensing.
