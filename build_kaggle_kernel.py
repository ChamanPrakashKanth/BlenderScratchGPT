"""
Builds notebook27596bd2cf.ipynb configured for Sparse-AST 500M Training on Kaggle GPU.
Exact Parameter Target: 501,301,280 parameters (~501.3M)
"""

import json
import os

def build_notebook():
    kaggle_dir = r"c:\Users\user\Downloads\checkpoint\notebook27596bd2cf"
    out_ipynb = os.path.join(kaggle_dir, "notebook27596bd2cf.ipynb")
    
    code = r'''# Stabilized Sparse-AST 500M Training on Blender Dataset & 3D Math Curriculum
import os, glob, time, subprocess, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Total VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB')

# -------------------------------------------------------------
# 1. Dataset Preparation: JSONL, Blender Manual & 3D Math Curriculum
# -------------------------------------------------------------
texts = []
for p in glob.glob('/kaggle/input/**/*.jsonl', recursive=True):
    try:
        print(f'Loading input dataset: {p}')
        texts.append(Path(p).read_text(errors='ignore'))
    except Exception as e:
        print(f'Error reading {p}: {e}')

# Clone blender manual into /tmp to avoid git hook artifacts in /kaggle/working
repo = Path('/tmp/blender-manual')
if not repo.exists():
    print('Cloning Blender documentation to /tmp/blender-manual...')
    try:
        subprocess.run(['git', 'clone', '--depth=1', 'https://projects.blender.org/blender/blender-manual.git', str(repo)], check=False, timeout=180)
    except Exception as e:
        print(f'Git clone error: {e}')

for root in [str(repo), '/opt/conda/lib/python3.12']:
    if os.path.exists(root):
        for p in (glob.glob(root + '/**/*.rst', recursive=True) + glob.glob(root + '/**/*.py', recursive=True)):
            try:
                if os.path.getsize(p) < 500000:
                    texts.append(Path(p).read_text(errors='ignore'))
            except Exception:
                pass

# Rich 3D Math & Python Blender Curriculum Text
curriculum_snippets = [
    """
# 3D Vector Mathematics in Blender Python using mathutils
import math, mathutils
from mathutils import Vector, Matrix, Quaternion, Euler
v1 = Vector((1.0, 2.0, 3.0))
v2 = Vector((4.0, 5.0, 6.0))
v_add = v1 + v2
v_sub = v2 - v1
length = v1.length
v_unit = v1.normalized()
dot_val = v1.dot(v2)
normal_vec = v1.cross(v2)
proj = (v1.dot(v2) / (v2.length**2)) * v2
dist = (v2 - v1).length
""",
    """
# 4x4 Affine Transformation Matrices
from mathutils import Matrix, Vector, Euler
mat_ident = Matrix.Identity(4)
mat_trans = Matrix.Translation(Vector((3.0, -2.0, 5.0)))
mat_rot = Matrix.Rotation(math.radians(45.0), 4, 'Z')
mat_scale = Matrix.Diagonal(Vector((2.0, 2.0, 0.5, 1.0)))
mat_combined = mat_trans @ mat_rot @ mat_scale
loc, rot_quat, scale = mat_combined.decompose()
mat_inv = mat_combined.inverted()
""",
    """
# Quaternions, Euler Angles & SLERP
from mathutils import Quaternion, Euler, Vector
euler_rot = Euler((math.radians(30), math.radians(45), math.radians(60)), 'XYZ')
quat = euler_rot.to_quaternion()
axis = Vector((0.0, 0.0, 1.0)).normalized()
q_axis_angle = Quaternion(axis, math.radians(90.0))
q_combined = quat @ q_axis_angle
v_rotated = q_combined @ Vector((1.0, 0.0, 0.0))
q_slerp = quat.slerp(q_axis_angle, 0.5)
""",
    """
# High-Performance Procedural Geometry with bmesh
import bpy, bmesh
from mathutils import Vector, Matrix
mesh = bpy.data.meshes.new("ProceduralTorus")
obj = bpy.data.objects.new("TorusObj", mesh)
bpy.context.collection.objects.link(obj)
bm = bmesh.new()
bmesh.ops.create_cube(bm, size=2.0)
bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=2, use_grid_fill=True)
bmesh.ops.bevel(bm, geom=bm.edges, offset=0.1, segments=2)
bm.normal_update()
bm.to_mesh(mesh)
bm.free()
""",
    """
# mathutils.kdtree and bvhtree Spatial Searching
import bpy, mathutils
from mathutils import Vector, kdtree, bvhtree
kd = kdtree.KDTree(1000)
for i, vert in enumerate(bpy.context.active_object.data.vertices):
    kd.insert(vert.co, i)
kd.balance()
co, index, dist = kd.find(Vector((0, 0, 0)))
bvh = bvhtree.BVHTree.FromPolygons([v.co for v in bpy.context.active_object.data.vertices],
                                   [f.vertices for f in bpy.context.active_object.data.polygons])
hit_loc, hit_norm, hit_idx, hit_dist = bvh.ray_cast(Vector((0,0,5)), Vector((0,0,-1)))
""",
    """
# Vectorized Vertex Manipulations with numpy in Blender
import bpy, numpy as np
mesh = bpy.context.active_object.data
n_verts = len(mesh.vertices)
coords = np.empty(n_verts * 3, dtype=np.float32)
mesh.vertices.foreach_get('co', coords)
coords = coords.reshape((n_verts, 3))
coords[:, 2] += 0.5 * np.sin(coords[:, 0] * 3.0)
mesh.vertices.foreach_set('co', coords.ravel())
mesh.update()
""",
    """
# 3D Viewport Drawing with gpu and gpu_extras
import bpy, gpu
from gpu_extras.batch import batch_for_shader
shader = gpu.shader.from_builtin('UNIFORM_COLOR')
batch = batch_for_shader(shader, 'LINES', {"pos": [(0,0,0), (1,1,1)]})
shader.bind()
shader.uniform_float("color", (1.0, 0.5, 0.0, 1.0))
batch.draw(shader)
"""
]

texts += curriculum_snippets * 1500
texts += ['matrix transform quaternion dot product cross product Blender bpy Python function calculus algebra geometry bmesh mathutils.'] * 3000

raw_text = '\n'.join(texts)
data = torch.tensor(list(raw_text.encode('utf8', 'ignore')), dtype=torch.long)
print(f'Total training dataset size: {len(data):,} bytes')

# -------------------------------------------------------------
# 2. Stabilized Sparse-AST 500M Architecture (d=1184, h=2368, layers=32, heads=16)
# -------------------------------------------------------------
class N(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.w = nn.Parameter(torch.ones(d))
    def forward(s, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype) * s.w

class B(nn.Module):
    def __init__(s, d=1184, h=2368):
        super().__init__()
        s.n1 = N(d)
        s.n2 = N(d)
        s.a = nn.MultiheadAttention(d, 16, batch_first=True)
        s.up = nn.Linear(d, h)
        s.r = nn.Linear(h, 3)
        s.down = nn.Linear(2*h, d)
        s.al = nn.Sequential(nn.Linear(3*h+3, 24), nn.SiLU(), nn.Linear(24, 1))
        s.m = nn.Linear(d, d, bias=False)
    def forward(s, x):
        t = x.shape[1]
        q = s.n1(x)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        x = torch.clamp(x + s.a(q, q, q, attn_mask=mask, need_weights=False)[0], -50, 50)
        z = torch.clamp(s.up(s.n2(x)), -12, 12)
        g = F.gumbel_softmax(s.r(z), tau=2.0, hard=True, dim=-1)
        e = z.sign() * torch.expm1(z.abs().clamp(max=2.0))
        u = (torch.stack((z, z.sign() * torch.log1p(z.abs()), e), -1) * g.unsqueeze(-2)).sum(-1).clamp(-8, 8)
        p, n = F.relu(u), F.relu(-u)
        d = torch.exp(-F.softplus(s.al(torch.cat((p, n, u, g), -1))).squeeze(-1)).clamp(1e-4, 0.9999)
        y = torch.clamp(x + s.down(torch.cat((p, n), -1)), -50, 50)
        state = torch.zeros_like(y[:, 0])
        o = []
        for j in range(t):
            state = torch.clamp(state * d[:, j:j+1] + y[:, j] * (1 - d[:, j:j+1]), -50, 50)
            o.append(torch.clamp(y[:, j] + s.m(state), -50, 50))
        return torch.stack(o, 1)

class M(nn.Module):
    def __init__(s, d=1184, h=2368, layers=32, seq_len=1024):
        super().__init__()
        s.d = d
        s.seq_len = seq_len
        s.e = nn.Embedding(512, d)
        s.p = nn.Embedding(seq_len, d)
        s.b = nn.ModuleList([B(d, h) for _ in range(layers)])
        s.n = N(d)
        s.h = nn.Linear(d, 512, bias=False)
        s.h.weight = s.e.weight
    def forward(s, i):
        if i.shape[1] > s.seq_len:
            i = i[:, -s.seq_len:]
        x = s.e(i) + s.p(torch.arange(i.shape[1], device=i.device))[None]
        for b in s.b:
            x = b(x)
        return s.h(s.n(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = M(d=1184, h=2368, layers=32, seq_len=1024).to(device)
param_count = sum(p.numel() for p in m.parameters())
print(f'Model Sparse-AST-500M successfully built: {param_count:,} parameters ({param_count/1e6:.2f}M)')

# -------------------------------------------------------------
# 3. Training Loop with Mixed Precision & Gradient Accumulation
# -------------------------------------------------------------
opt = torch.optim.AdamW(m.parameters(), lr=8e-5, weight_decay=0.01)
scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

BATCH, GRAD_ACCUM, SEQ, START, STEPS = 1, 4, 512, 0, 1000
last = START
t0 = time.time()
print(f'Beginning training Sparse-AST 500M from step 1 to {STEPS} (effective batch {BATCH*GRAD_ACCUM})...')

opt.zero_grad(set_to_none=True)
running_loss = 0.0

for offset in range(1, STEPS + 1):
    step = START + offset
    
    # Gradient accumulation loop
    step_loss = 0.0
    for accum_i in range(GRAD_ACCUM):
        ix = torch.randint(0, len(data) - SEQ - 1, (BATCH,))
        x = torch.stack([data[j:j+SEQ] for j in ix]).to(device)
        y = torch.stack([data[j+1:j+SEQ+1] for j in ix]).to(device)
        
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = m(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten()) / GRAD_ACCUM
            if not torch.isfinite(loss):
                print(f'STOP non-finite loss at step {step}')
                break
            scaler.scale(loss).backward()
            step_loss += loss.item() * GRAD_ACCUM
        else:
            logits = m(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten()) / GRAD_ACCUM
            loss.backward()
            step_loss += loss.item() * GRAD_ACCUM
            
    if torch.cuda.is_available():
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 0.3)
        scaler.step(opt)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(m.parameters(), 0.3)
        opt.step()
    opt.zero_grad(set_to_none=True)
    
    last = step
    if step % 100 == 0:
        elapsed = time.time() - t0
        path = f'/kaggle/working/checkpoint_500m_{step:04d}.pt'
        loss_val = float(step_loss)
        # Save model weights in FP16 to keep checkpoints compact (~1.0 GB)
        half_state = {k: v.half() if v.is_floating_point() else v for k, v in m.state_dict().items()}
        torch.save({
            'model': half_state,
            'step': step,
            'loss': loss_val,
            'config': 'sparse-AST-500M-stabilized',
            'params': param_count,
            'd': 1184,
            'h': 2368,
            'layers': 32,
            'seq_len': 1024
        }, path)
        print(f'Step {step:04d}/{STEPS} | Loss: {loss_val:.4f} | Elapsed: {elapsed:.1f}s | Saved: {path}')

if last == START + STEPS:
    final_path = '/kaggle/working/final_sparse_ast_500m.pt'
    loss_val = float(step_loss)
    half_state = {k: v.half() if v.is_floating_point() else v for k, v in m.state_dict().items()}
    torch.save({
        'model': half_state,
        'step': last,
        'loss': loss_val,
        'config': 'sparse-AST-500M-stabilized',
        'params': param_count,
        'd': 1184,
        'h': 2368,
        'layers': 32,
        'seq_len': 1024
    }, final_path)
    print(f'SUCCESS: 500M Training finished at step {last}. Saved {final_path}')
'''

    nb_data = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            },
            "accelerator": "GPU",
            "isGpuEnabled": True
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            {
                "cell_type": "code",
                "metadata": {"trusted": True},
                "execution_count": None,
                "source": code,
                "outputs": []
            }
        ]
    }
    
    with open(out_ipynb, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)
        
    print(f"Generated {out_ipynb} successfully!")

if __name__ == '__main__':
    build_notebook()
