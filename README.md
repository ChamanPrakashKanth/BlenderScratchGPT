# BlenderScratchGPT

A lightweight research and engineering suite for training, scaling, and ensembling custom **Sparse-AST** language models on Blender 5.x Python APIs, 3D mathematics, procedural geometry, and shader programming.

---

## Highlights & Latest Updates

- **Top-$K$ Mixture-of-Experts (MoE) Ensemble**: Connects the 3M, 10M, and 100M Sparse-AST models via a learned gating router with dynamic top-$k$ expert selection ($k \in \{1, 2, 3\}$), soft predictive weighting, and autoregressive generation.
- **Model Scaling Suite**: Scaled architectures from 3M to 10M, 100M, and **200M** parameters ($d=800, h=1600, \text{layers}=28$).
- **Expanded 11-Module Curriculum**: Production-grade synthetic & doc curriculum covering 3D Vector Math, 4x4 Affine Matrices, Quaternions, SLERP, Möller-Trumbore Ray-Triangle Intersections, `bmesh` procedural topology, `mathutils.kdtree` and `mathutils.bvhtree`, vectorized `numpy` mesh operations, `gpu` / `gpu_extras` viewport drawing, and `bpy_extras` mouse raycasting.
- **Kaggle GPU Training Pipeline**: Fully automated Kaggle CLI integration with GPU acceleration (Tesla T4) and automatic checkpoint synchronization.
- **Interactive Chat Console**: Launchable directly via `chat.bat` or `python chat.py` with real-time token streaming and dynamic expert routing attribution.

---

## Model Architecture & Benchmarks

The Sparse-AST architecture incorporates recurrent state gates, learnable RMSNorm, multi-head causal self-attention, and nonlinear adaptive activations.

| Model / Architecture | Parameters | Layers | Dims ($d/h$) | Curriculum Loss | Perplexity | Checkpoint / Config |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Sparse-AST 3M** | 3,145,296 | 4 | 256 / 512 | 15.5198 | 5,497,504 | `final_sparse_ast.pt` |
| **Sparse-AST 10M** | 10,320,744 | 6 | 384 / 768 | 21.1346 | 485,165,195 | `final_sparse_ast_10m.pt` |
| **Sparse-AST 100M** | 100,532,728 | 18 | 704 / 1408 | 3.7621 | 43.04 | `final_sparse_ast_100m.pt` |
| **Top-$K$ MoE Ensemble ($k=2$)** | **113,998,768** | **4+6+18** | **256 / 384 / 704** | **6.1258** | **457.52** | `topk_sparse_ast_router.pt` |
| **Sparse-AST 200M** | **201,121,072** | **28** | **800 / 1600** | *Training on Kaggle* | *Training on Kaggle* | `final_sparse_ast_200m.pt` |

### Top-$K$ Routing Benchmark

The Top-$K$ Router dynamically routes tokens across the 3 expert backbones based on context complexity:

- **Top-1 Routing ($k=1$)**: Loss: `7.6603` | Perplexity: `2122.40` | Expert Shares: 3M: **31.5%**, 10M: **0.0%**, 100M: **68.5%**
- **Top-2 Routing ($k=2$)**: Loss: **`6.1258`** | Perplexity: **`457.52`** | Expert Shares: 3M: **33.9%**, 10M: **31.0%**, 100M: **35.1%**
- **Top-3 Routing ($k=3$)**: Loss: **`5.6627`** | Perplexity: **`287.94`** | Expert Shares: 3M: **33.3%**, 10M: **33.3%**, 100M: **33.3%**

---

## Repository Structure

```text
BlenderScratchGPT/
├── chat.bat                           # Double-clickable Windows launcher for chat console
├── chat.py                            # Interactive terminal chat & code generation interface
├── topk_ensemble.py                   # Top-K MoE Ensemble Router connecting 3M, 10M, 100M
├── topk_sparse_ast_router.pt          # Calibrated Top-K router weights (tracked in git)
├── test_models.py                     # Evaluation and comparison suite across all models
├── train_200m_blender_math.py         # 201M Sparse-AST model architecture & training pipeline
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

## Quickstart

### 1. Interactive Chat Console

Double-click `chat.bat` in Windows Explorer or run from the command line:

```cmd
chat.bat
```

Inside the console, you can interactively generate code and inspect live MoE expert routing:

```text
User [Top-K MoE Ensemble (3M+10M+100M)] > import mathutils
from mathutils import Vector
v1 = Vector((1.0, 2.0, 3.0))

Assistant: 
v2 = Vector((4.0, 5.0, 6.0))
dot_prod = v1.dot(v2)
cross_prod = v1.cross(v2)
  [MoE Routing: 3M: 8 tokens, 100M: 27 tokens]
```

#### Built-in Interactive Commands

| Command | Description | Example |
| :--- | :--- | :--- |
| `/model <1-5>` | Switch active model (Ensemble, 100M, 200M, 10M, 3M) | `/model 2` |
| `/k <1\|2\|3>` | Change Top-$K$ routing depth | `/k 2` |
| `/temp <float>` | Set sampling temperature (0.05 to 1.5) | `/temp 0.6` |
| `/tokens <int>` | Set maximum generated tokens | `/tokens 50` |
| `/help` | Display sample prompts | `/help` |
| `exit` | Exit console | `exit` |

---

### 2. Model Evaluation Suite

Run the full evaluation and benchmark comparison suite across all checkpoints:

```powershell
python test_models.py
```

Outputs metrics table and comparative code completions for standard 3D math and Blender API prompts.

---

### 3. Top-$K$ MoE Router Calibration & Testing

To inspect or re-train the Top-$K$ gating router:

```powershell
python topk_ensemble.py
```

---

### 4. Kaggle GPU Training & Checkpoint Download

To train the 200M model on Kaggle GPU:

1. **Build & Push Notebook**:
   ```powershell
   python build_kaggle_kernel.py
   kaggle kernels push -p notebook27596bd2cf
   ```

2. **Monitor & Auto-Download Checkpoints**:
   ```powershell
   python monitor_and_download_kaggle.py chamankanth/notebook27596bd2cf ./
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

## License

This project is intended for educational, academic, and experimental research. Blender documentation and Blender APIs are subject to the Blender Foundation licensing.
