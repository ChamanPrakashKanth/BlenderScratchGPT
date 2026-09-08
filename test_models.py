"""
Model Evaluation & Comparison Suite for Sparse-AST Architectures:
Tests 3M, 10M, and 100M models on Blender Python & 3D Math Curriculum.
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class N(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.w = nn.Parameter(torch.ones(d))
    def forward(s, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype) * s.w

class B(nn.Module):
    def __init__(s, d, h, al_hidden=24):
        super().__init__()
        s.n1 = N(d)
        s.n2 = N(d)
        s.a = nn.MultiheadAttention(d, 8, batch_first=True)
        s.up = nn.Linear(d, h)
        s.r = nn.Linear(h, 3)
        s.down = nn.Linear(2*h, d)
        s.al = nn.Sequential(nn.Linear(3*h+3, al_hidden), nn.SiLU(), nn.Linear(al_hidden, 1))
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

class SparseAST(nn.Module):
    def __init__(s, d, h, layers, al_hidden=24, seq_len=4096, vocab=512):
        super().__init__()
        s.d = d
        s.seq_len = seq_len
        s.e = nn.Embedding(vocab, d)
        s.p = nn.Embedding(seq_len, d)
        s.b = nn.ModuleList([B(d, h, al_hidden) for _ in range(layers)])
        s.n = N(d)
        s.h = nn.Linear(d, vocab, bias=False)
        s.h.weight = s.e.weight

    def forward(s, i):
        if i.shape[1] > s.seq_len:
            i = i[:, -s.seq_len:]
        seq = i.shape[1]
        pos = torch.arange(seq, device=i.device)[None]
        x = s.e(i) + s.p(pos)
        for b in s.b:
            x = b(x)
        return s.h(s.n(x))

def auto_load_model(checkpoint_path, target_seq_len=4096):
    ck = torch.load(checkpoint_path, map_location='cpu')
    state = ck['model']
    
    # Infer architecture dynamically from tensor shapes
    vocab, d = state['e.weight'].shape
    orig_seq_len, _ = state['p.weight'].shape
    h, _ = state['b.0.up.weight'].shape
    al_hidden = state['b.0.al.0.weight'].shape[0]
    layers = len([k for k in state.keys() if 'b.' in k and '.n1.w' in k])
    
    # Dynamically expand positional embeddings to usable Blender script context (up to 4096 tokens)
    effective_seq_len = max(orig_seq_len, target_seq_len)
    if effective_seq_len > orig_seq_len:
        old_p = state['p.weight'] # [orig_seq_len, d]
        new_p = F.interpolate(old_p.T.unsqueeze(0), size=effective_seq_len, mode='linear', align_corners=True).squeeze(0).T
        state['p.weight'] = new_p
    
    model = SparseAST(d=d, h=h, layers=layers, al_hidden=al_hidden, seq_len=effective_seq_len, vocab=vocab)
    model.load_state_dict(state)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    return model, {
        'name': os.path.basename(checkpoint_path),
        'config': ck.get('config', 'unknown'),
        'step': ck.get('step', 0),
        'loss': ck.get('loss'),
        'd': d,
        'h': h,
        'layers': layers,
        'al_hidden': al_hidden,
        'seq_len': effective_seq_len,
        'orig_seq_len': orig_seq_len,
        'params': params
    }

def evaluate_on_curriculum(model, text_data, seq_len=1024, num_samples=20):
    losses = []
    with torch.no_grad():
        max_start = len(text_data) - seq_len - 1
        if max_start <= 0:
            seq_len = len(text_data) - 2
            max_start = 1
        for i in range(min(num_samples, max_start)):
            idx = (i * 317) % max_start
            x = text_data[idx : idx + seq_len].unsqueeze(0)
            y = text_data[idx + 1 : idx + seq_len + 1].unsqueeze(0)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            losses.append(float(loss))
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl

def generate_completion(model, prompt_text, max_new_tokens=100, temperature=0.5):
    model.eval()
    encoded = list(prompt_text.encode('utf-8', 'ignore'))
    seq_max = min(model.seq_len - 1, 4095)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            cur_input = encoded[-seq_max:]
            x = torch.tensor([cur_input], dtype=torch.long)
            logits = model(x)[:, -1, :]
            if temperature <= 0.05:
                next_token = int(logits.argmax(dim=-1).item())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())
            encoded.append(next_token)
            
    res = bytes([t for t in encoded if t < 256]).decode('utf-8', errors='replace')
    return res

def run_tests():
    curriculum_path = r"c:\Users\user\Downloads\checkpoint\blender_3dmath_curriculum.txt"
    with open(curriculum_path, 'r', encoding='utf-8') as f:
        curr_text = f.read()
    curr_data = torch.tensor(list(curr_text.encode('utf-8', 'ignore')), dtype=torch.long)
    
    models_to_test = [
        ("3M Model", r"c:\Users\user\Downloads\checkpoint\final_sparse_ast.pt"),
        ("10M Model", r"c:\Users\user\Downloads\checkpoint\final_sparse_ast_10m.pt"),
        ("100M Model", r"c:\Users\user\Downloads\checkpoint\final_sparse_ast_100m.pt"),
        ("200M Model", r"c:\Users\user\Downloads\checkpoint\final_sparse_ast_200m.pt")
    ]
    
    prompts = [
        "import mathutils\nfrom mathutils import Vector\nv1 = Vector((",
        "import bpy\n# Create primitive cube\nbpy.ops.mesh.",
        "import bmesh\nbm = bmesh.new()\n",
        "# 3D Dot product calculation\ndef dot_product("
    ]
    
    results = []
    print("="*85)
    print("      SPARSE-AST MODEL TEST SUITE (3M vs 10M vs 100M vs 200M & Top-K MoE)")
    print("="*85)
    
    for label, path in models_to_test:
        if not os.path.exists(path):
            # Check for interim checkpoint if final not yet downloaded
            if "200M" in label:
                alt_paths = sorted([p for p in os.listdir(r"c:\Users\user\Downloads\checkpoint") if p.startswith("checkpoint_200m_") and p.endswith(".pt")])
                if alt_paths:
                    path = os.path.join(r"c:\Users\user\Downloads\checkpoint", alt_paths[-1])
                    label = f"200M Model ({alt_paths[-1]})"
                else:
                    print(f"Skipping {label}: {path} not found (training in progress)")
                    continue
            else:
                print(f"Skipping {label}: file not found at {path}")
                continue
                
        print(f"\n[+] Testing {label} ({os.path.basename(path)})...")
        model, info = auto_load_model(path)
        t0 = time.time()
        loss, ppl = evaluate_on_curriculum(model, curr_data, seq_len=30, num_samples=25)
        eval_time = time.time() - t0
        
        info['curr_loss'] = loss
        info['curr_ppl'] = ppl
        info['eval_time'] = eval_time
        info['model_obj'] = model
        results.append(info)
        
        print(f"    Params: {info['params']:,} | Layers: {info['layers']} (d={info['d']}, h={info['h']})")
        print(f"    Curriculum Loss: {loss:.4f} | Perplexity: {ppl:.2f} | Eval Time: {eval_time:.2f}s")

    # Evaluate Top-K MoE Ensemble (Connecting 3M, 10M, 100M)
    router_path = r"c:\Users\user\Downloads\checkpoint\topk_sparse_ast_router.pt"
    topk_ensemble_obj = None
    if os.path.exists(router_path):
        print(f"\n[+] Testing Top-K MoE Ensemble (Connecting 3M + 10M + 100M)...")
        from topk_ensemble import TopKSparseASTEnsemble
        ensemble = TopKSparseASTEnsemble(k=2)
        router_data = torch.load(router_path, map_location=ensemble.device)
        ensemble.router.load_state_dict(router_data['router_state'])
        
        t0 = time.time()
        ens_loss, ens_ppl, expert_dist = ensemble.evaluate(curr_data, seq_len=30, num_samples=25, k=2)
        eval_time = time.time() - t0
        
        total_ens_params = sum(r['params'] for r in results if r['name'] in ['final_sparse_ast.pt', 'final_sparse_ast_10m.pt', 'final_sparse_ast_100m.pt'])
        ens_info = {
            'name': 'Top-K MoE Ensemble (k=2)',
            'params': total_ens_params,
            'layers': '4+6+18',
            'd': '256/384/704',
            'h': '512/768/1408',
            'curr_loss': ens_loss,
            'curr_ppl': ens_ppl,
            'eval_time': eval_time,
            'expert_dist': expert_dist,
            'is_ensemble': True,
            'model_obj': ensemble
        }
        results.append(ens_info)
        print(f"    Total Combined Params: {total_ens_params:,} | Gating: Top-2 MoE")
        print(f"    Curriculum Loss: {ens_loss:.4f} | Perplexity: {ens_ppl:.2f} | Eval Time: {eval_time:.2f}s")
        print(f"    Expert Utilization: 3M={expert_dist[0]:.1f}%, 10M={expert_dist[1]:.1f}%, 100M={expert_dist[2]:.1f}%")

    print("\n" + "="*85)
    print("               ARCHITECTURE & METRICS COMPARISON TABLE")
    print("="*85)
    print(f"{'Model / Architecture':<27} | {'Params':<12} | {'Layers':<7} | {'Dims (d/h)':<13} | {'Loss':<8} | {'Perplexity':<10}")
    print("-" * 85)
    for r in results:
        dh_str = f"{r['d']}/{r['h']}"
        print(f"{r['name']:<27} | {r['params']:<12,} | {str(r['layers']):<7} | {dh_str:<13} | {r['curr_loss']:<8.4f} | {r['curr_ppl']:<10.2f}")

    print("\n" + "="*85)
    print("                AUTOREGRESSIVE CODE GENERATION SAMPLES")
    print("="*85)
    for p_idx, prompt in enumerate(prompts, 1):
        print(f"\n>>> [PROMPT {p_idx}]: \"{repr(prompt)}\"")
        for r in results:
            if r.get('is_ensemble'):
                completion, routes = r['model_obj'].generate(prompt, max_new_tokens=35, k=2, temperature=0.6)
                new_text = completion[len(prompt):]
                print(f"[{r['name']}] -> {repr(new_text)} (routes: {routes})")
            else:
                completion = generate_completion(r['model_obj'], prompt, max_new_tokens=35, temperature=0.6)
                new_text = completion[len(prompt):]
                print(f"[{r['name']} ({r['params']//1_000_000}M)] -> {repr(new_text)}")

if __name__ == '__main__':
    run_tests()
