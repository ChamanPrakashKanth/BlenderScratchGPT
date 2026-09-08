"""
Sparse-AST 200M Parameter Model Architecture & Training Script
Domain: Blender 3D Mathematics & Blender Python Libraries (bpy, mathutils, bmesh, numpy, gpu)
Target Architecture: d=800, h=1600, layers=28, seq_len=32, vocab=512 -> 201,121,072 parameters (~200M)
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------
# 1. Architecture Definition for Sparse-AST 200M
# -------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.w

class SparseASTBlock(nn.Module):
    def __init__(self, d=800, h=1600, al_hidden=24, n_heads=8):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.a = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.up = nn.Linear(d, h)
        self.r = nn.Linear(h, 3)
        self.down = nn.Linear(2*h, d)
        self.al = nn.Sequential(nn.Linear(3*h + 3, al_hidden), nn.SiLU(), nn.Linear(al_hidden, 1))
        self.m = nn.Linear(d, d, bias=False)
        
    def forward(self, x):
        t = x.shape[1]
        q = self.n1(x)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        x = torch.clamp(x + self.a(q, q, q, attn_mask=mask, need_weights=False)[0], -50, 50)
        z = torch.clamp(self.up(self.n2(x)), -12, 12)
        g = F.gumbel_softmax(self.r(z), tau=2.0, hard=True, dim=-1)
        e = z.sign() * torch.expm1(z.abs().clamp(max=2.0))
        u = (torch.stack((z, z.sign() * torch.log1p(z.abs()), e), -1) * g.unsqueeze(-2)).sum(-1).clamp(-8, 8)
        p, n = F.relu(u), F.relu(-u)
        d = torch.exp(-F.softplus(self.al(torch.cat((p, n, u, g), -1))).squeeze(-1)).clamp(1e-4, 0.9999)
        y = torch.clamp(x + self.down(torch.cat((p, n), -1)), -50, 50)
        state = torch.zeros_like(y[:, 0])
        o = []
        for j in range(t):
            state = torch.clamp(state * d[:, j:j+1] + y[:, j] * (1 - d[:, j:j+1]), -50, 50)
            o.append(torch.clamp(y[:, j] + self.m(state), -50, 50))
        return torch.stack(o, 1)

class SparseAST200M(nn.Module):
    def __init__(self, d=800, h=1600, layers=28, seq_len=32, vocab=512, al_hidden=24):
        super().__init__()
        self.d = d
        self.h = h
        self.layers = layers
        self.seq_len = seq_len
        self.vocab = vocab
        
        self.e = nn.Embedding(vocab, d)
        self.p = nn.Embedding(seq_len, d)
        self.b = nn.ModuleList([SparseASTBlock(d=d, h=h, al_hidden=al_hidden, n_heads=8) for _ in range(layers)])
        self.n = RMSNorm(d)
        self.h_out = nn.Linear(d, vocab, bias=False)
        self.h_out.weight = self.e.weight  # Weight tying

    def forward(self, i):
        seq = i.shape[1]
        pos = torch.arange(seq, device=i.device)[None]
        x = self.e(i) + self.p(pos)
        for b in self.b:
            x = b(x)
        return self.h_out(self.n(x))

def train(curriculum_path, output_dir, steps=300, batch_size=1, grad_accum=4, lr=1e-4, seq_len=32, resume_from=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[200M Training] Using execution device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"[200M Training] GPU Name: {torch.cuda.get_device_name(0)} | VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB", flush=True)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Instantiate 200M Model
    model = SparseAST200M(d=800, h=1600, layers=28, seq_len=seq_len, vocab=512)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[200M Training] Model initialized with {param_count:,} parameters ({round(param_count/1e6, 2)}M).", flush=True)
    
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        print(f"[200M Training] Resuming from checkpoint {resume_from}...", flush=True)
        ck = torch.load(resume_from, map_location='cpu')
        model.load_state_dict(ck['model'])
        start_step = ck.get('step', 0)
        print(f"[200M Training] Resumed at step {start_step}.", flush=True)
        
    model.to(device)
    
    # 2. Load and Prepare Curriculum Data
    print(f"[200M Training] Reading curriculum from {curriculum_path}...", flush=True)
    with open(curriculum_path, 'r', encoding='utf-8') as f:
        text = f.read()
    data = torch.tensor(list(text.encode('utf-8', 'ignore')), dtype=torch.long)
    print(f"[200M Training] Loaded curriculum dataset: {len(data):,} bytes.", flush=True)
    
    # 3. Optimization Setup
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    t0 = time.time()
    print(f"[200M Training] Starting training from step {start_step+1} to {start_step+steps}...", flush=True)
    print(f"[200M Training] Batch size: {batch_size}, Grad Accumulation: {grad_accum} (Effective Batch = {batch_size * grad_accum})", flush=True)
    
    model.train()
    opt.zero_grad(set_to_none=True)
    accum_loss = 0.0
    
    for offset in range(1, steps + 1):
        step = start_step + offset
        
        for accum_idx in range(grad_accum):
            ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
            x = torch.stack([data[j:j+seq_len] for j in ix]).to(device)
            y = torch.stack([data[j+1:j+seq_len+1] for j in ix]).to(device)
            
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten()) / grad_accum
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten()) / grad_accum
                loss.backward()
                
            accum_loss += float(loss) * grad_accum
            
        if torch.cuda.is_available():
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            opt.step()
            
        opt.zero_grad(set_to_none=True)
        cur_loss = accum_loss / grad_accum
        accum_loss = 0.0
        
        # Periodic Logging and Checkpoint
        if step % 25 == 0 or step == start_step + steps:
            elapsed = time.time() - t0
            speed = (offset * batch_size * grad_accum) / max(elapsed, 0.01)
            print(f"Step {step:04d}/{start_step+steps} | Loss: {cur_loss:.4f} | Speed: {speed:.1f} seqs/s | Elapsed: {elapsed:.1f}s", flush=True)
            
        if step % 50 == 0 or step == start_step + steps:
            chk_path = os.path.join(output_dir, f"checkpoint_200m_{step:04d}.pt")
            torch.save({
                'model': model.state_dict(),
                'step': step,
                'loss': cur_loss,
                'config': 'sparse-AST-200M',
                'd': 800,
                'h': 1600,
                'layers': 28,
                'params': param_count,
                'seq_len': seq_len,
                'vocab': 512
            }, chk_path)
            print(f"[200M Training] Checkpoint saved -> {chk_path}", flush=True)
            
    # Final model export
    final_path = os.path.join(output_dir, "final_sparse_ast_200m.pt")
    torch.save({
        'model': model.state_dict(),
        'step': start_step + steps,
        'loss': cur_loss,
        'config': 'sparse-AST-200M',
        'd': 800,
        'h': 1600,
        'layers': 28,
        'params': param_count,
        'seq_len': seq_len,
        'vocab': 512
    }, final_path)
    print(f"\n[+] SUCCESS! 200M Model training complete. Saved final model to: {final_path}", flush=True)
    return final_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Sparse-AST 200M on Blender 3D Math & Python")
    parser.add_argument('--steps', type=int, default=100, help="Number of training steps")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=1, help="Per-step micro batch size")
    parser.add_argument('--grad_accum', type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument('--curriculum', type=str, default=r"c:\Users\user\Downloads\checkpoint\blender_3dmath_curriculum.txt")
    parser.add_argument('--output_dir', type=str, default=r"c:\Users\user\Downloads\checkpoint")
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    train(
        curriculum_path=args.curriculum,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        resume_from=args.resume
    )
