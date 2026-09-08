"""
Top-K Mixture-of-Experts (MoE) Ensemble Router for Sparse-AST Models:
Connects 3M, 10M, and 100M models with dynamic Top-K routing (k=1, 2, 3)
and top-k sampling for Blender Python & 3D Math Code Generation.
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_models import auto_load_model

class TopKRouter(nn.Module):
    """
    Lightweight Gating Router Network.
    Computes routing logits over the 3 Sparse-AST expert models:
    Expert 0: 3M Model  (Fast, lightweight syntax drafting)
    Expert 1: 10M Model (Intermediate architectural capacity)
    Expert 2: 100M Model (Deep 3D math, geometry, and bmesh reasoning)
    """
    def __init__(self, vocab_size=512, hidden_dim=64, num_experts=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
    def forward(self, x):
        """
        Input x: (Batch, Seq) token IDs
        Output: (Batch, Seq, NumExperts)
        """
        emb = self.norm(self.embed(x)) # (B, T, hidden_dim)
        router_logits = self.mlp(emb)   # (B, T, num_experts)
        return router_logits

class TopKSparseASTEnsemble(nn.Module):
    """
    Unified MoE Ensemble connecting 3 pretrained Sparse-AST checkpoints with Top-K routing.
    Supports k in {1, 2, 3}.
    """
    def __init__(self, model_paths=None, device=None, k=2, tau=1.0):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.k = k
        self.tau = tau
        self.num_experts = 3
        
        default_paths = [
            r"c:\Users\user\Downloads\checkpoint\final_sparse_ast.pt",      # Expert 0: 3M
            r"c:\Users\user\Downloads\checkpoint\final_sparse_ast_10m.pt",  # Expert 1: 10M
            r"c:\Users\user\Downloads\checkpoint\final_sparse_ast_100m.pt"  # Expert 2: 100M
        ]
        self.paths = model_paths or default_paths
        
        # Load the three models
        self.experts = nn.ModuleList()
        self.expert_infos = []
        for p in self.paths:
            print(f"[Top-K Ensemble] Loading expert: {os.path.basename(p)}...", flush=True)
            m, info = auto_load_model(p, target_seq_len=4096)
            m.to(device)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False  # Freeze pretrained backbones
            self.experts.append(m)
            self.expert_infos.append(info)
            
        self.min_seq_len = min(info['seq_len'] for info in self.expert_infos)
        
        # Learnable Router
        self.router = TopKRouter(vocab_size=512, hidden_dim=64, num_experts=self.num_experts).to(device)

    def forward_routing(self, x, k=None):
        """
        Computes Top-K gated logits for sequence x (Batch, Seq).
        If k=1, runs only the highest-scoring expert per token.
        If k=2 or 3, evaluates top-k experts and linearly combines their logits.
        """
        if k is None:
            k = self.k
        k = max(1, min(k, self.num_experts))
        
        # Window sequence if it exceeds the minimum expert capacity (32 tokens)
        if x.shape[1] > self.min_seq_len:
            x_input = x[:, -self.min_seq_len:]
        else:
            x_input = x
            
        b, t = x_input.shape
        
        # Router scores: (B, T, 3)
        router_logits = self.router(x_input)
        
        # Top-K selection
        topk_scores, topk_indices = torch.topk(router_logits, k=k, dim=-1) # (B, T, k)
        gate_weights = F.softmax(topk_scores / self.tau, dim=-1)            # (B, T, k)
        
        # Collect expert predictions with no_grad on frozen models
        unique_experts = torch.unique(topk_indices).tolist()
        expert_logits = {}
        with torch.no_grad():
            for exp_idx in unique_experts:
                expert_logits[exp_idx] = self.experts[exp_idx](x_input).detach() # (B, T, 512)
            
        # Combine logits weighted by gating scores
        combined_logits = torch.zeros((b, t, 512), device=self.device, dtype=torch.float32)
        for i in range(k):
            idx_i = topk_indices[:, :, i]      # (B, T)
            weight_i = gate_weights[:, :, i:i+1] # (B, T, 1)
            for exp_idx in unique_experts:
                mask = (idx_i == exp_idx).unsqueeze(-1) # (B, T, 1)
                if mask.any():
                    combined_logits += weight_i * expert_logits[exp_idx] * mask.float()
                    
        return combined_logits, router_logits, topk_indices, gate_weights

    def evaluate(self, text_data, seq_len=30, num_samples=25, k=None):
        """
        Evaluates cross-entropy loss and perplexity on test curriculum text.
        """
        self.eval()
        losses = []
        expert_counts = {0: 0, 1: 0, 2: 0}
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(min(num_samples, len(text_data) - seq_len - 1)):
                idx = (i * 137) % (len(text_data) - seq_len - 1)
                x = text_data[idx : idx + seq_len].unsqueeze(0).to(self.device)
                y = text_data[idx + 1 : idx + seq_len + 1].unsqueeze(0).to(self.device)
                logits, _, topk_indices, _ = self.forward_routing(x, k=k)
                
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                losses.append(float(loss))
                
                # Track expert routing stats
                for exp_idx in range(self.num_experts):
                    expert_counts[exp_idx] += int((topk_indices == exp_idx).sum().item())
                total_tokens += topk_indices.numel()
                
        avg_loss = sum(losses) / len(losses)
        ppl = math.exp(min(avg_loss, 20.0))
        expert_dist = {exp: (expert_counts[exp] / max(total_tokens, 1)) * 100 for exp in expert_counts}
        return avg_loss, ppl, expert_dist

    def generate(self, prompt_text, max_new_tokens=35, k=2, temperature=0.7, top_k_tokens=40):
        """
        Autoregressive code generation with Top-K expert model routing & Top-K token sampling.
        """
        self.eval()
        encoded = list(prompt_text.encode('utf-8', 'ignore'))
        expert_names = ["3M", "10M", "100M"]
        token_routes = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                cur_input = encoded[-self.min_seq_len:]
                x = torch.tensor([cur_input], dtype=torch.long, device=self.device)
                
                logits, _, topk_indices, gate_weights = self.forward_routing(x, k=k)
                last_logits = logits[:, -1, :] # (1, 512)
                
                # Log top-1 chosen expert for this token
                best_expert = int(topk_indices[0, -1, 0].item())
                token_routes.append(best_expert)
                
                # Temperature & Top-K vocabulary sampling
                if temperature <= 0.05:
                    next_token = int(last_logits.argmax(dim=-1).item())
                else:
                    scaled_logits = last_logits / temperature
                    if top_k_tokens > 0:
                        v, _ = torch.topk(scaled_logits, min(top_k_tokens, scaled_logits.size(-1)))
                        scaled_logits[scaled_logits < v[:, [-1]]] = -float('inf')
                    probs = F.softmax(scaled_logits, dim=-1)
                    next_token = int(torch.multinomial(probs, num_samples=1).item())
                    
                encoded.append(next_token)
                
        res = bytes([t for t in encoded if t < 256]).decode('utf-8', errors='replace')
        route_summary = {expert_names[i]: token_routes.count(i) for i in range(3)}
        return res, route_summary

def train_topk_router(ensemble, text_data, steps=60, lr=2e-3, seq_len=30, batch_size=2):
    """
    Calibrates the Top-K Gating Router on the Blender 3D Math curriculum.
    Backbones remain frozen, enabling rapid convergence.
    """
    print(f"\n[Top-K Router Training] Training router on curriculum ({steps} steps)...", flush=True)
    opt = torch.optim.AdamW(ensemble.router.parameters(), lr=lr, weight_decay=0.01)
    ensemble.router.train()
    t0 = time.time()
    
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(text_data) - seq_len - 1, (batch_size,))
        x = torch.stack([text_data[j:j+seq_len] for j in ix]).to(ensemble.device)
        y = torch.stack([text_data[j+1:j+seq_len+1] for j in ix]).to(ensemble.device)
        
        opt.zero_grad()
        logits, router_logits, _, gate_weights = ensemble.forward_routing(x, k=ensemble.k)
        
        # Primary Task Loss (Predictive Cross Entropy)
        ce_loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        
        # Load Balancing Regularizer (Encourages utilization across all 3 experts)
        router_probs = F.softmax(router_logits, dim=-1).mean(dim=[0, 1]) # (3,)
        entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-6))
        balance_loss = -0.05 * entropy
        
        total_loss = ce_loss + balance_loss
        total_loss.backward()
        opt.step()
        
        if step % 20 == 0 or step == steps:
            print(f"Step {step:03d}/{steps} | CrossEntropy: {float(ce_loss):.4f} | Prob Dist: {[round(float(p), 3) for p in router_probs]}", flush=True)
            
    print(f"[Top-K Router Training] Completed in {time.time()-t0:.1f}s.", flush=True)

def main():
    curriculum_path = r"c:\Users\user\Downloads\checkpoint\blender_3dmath_curriculum.txt"
    with open(curriculum_path, 'r', encoding='utf-8') as f:
        curr_text = f.read()
    curr_data = torch.tensor(list(curr_text.encode('utf-8', 'ignore')), dtype=torch.long)
    
    print("="*85, flush=True)
    print("      TOP-K MIXTURE-OF-EXPERTS (MoE) ENSEMBLE ROUTER INITIALIZATION", flush=True)
    print("="*85, flush=True)
    
    ensemble = TopKSparseASTEnsemble(k=2)
    
    # Train/calibrate router for 40 steps
    train_topk_router(ensemble, curr_data, steps=40, lr=2e-3)
    
    # Save trained router
    save_path = r"c:\Users\user\Downloads\checkpoint\topk_sparse_ast_router.pt"
    torch.save({
        'router_state': ensemble.router.state_dict(),
        'k': ensemble.k,
        'tau': ensemble.tau,
        'num_experts': ensemble.num_experts,
        'config': 'TopK-MoE-SparseAST-3M-10M-100M'
    }, save_path)
    print(f"\n[+] Saved trained Top-K MoE router checkpoint to {save_path}!", flush=True)
    
    # Evaluate at different k values
    print("\n" + "="*85, flush=True)
    print("             TOP-K ROUTING BENCHMARK (k=1 vs k=2 vs k=3)", flush=True)
    print("="*85, flush=True)
    for test_k in [1, 2, 3]:
        loss, ppl, dist = ensemble.evaluate(curr_data, seq_len=30, num_samples=20, k=test_k)
        print(f"Top-{test_k} Routing | Loss: {loss:.4f} | Perplexity: {ppl:.2f} | Expert Shares: 3M={dist[0]:.1f}%, 10M={dist[1]:.1f}%, 100M={dist[2]:.1f}%", flush=True)

    # Autoregressive generation demonstration
    print("\n" + "="*85, flush=True)
    print("            TOP-K AUTOREGRESSIVE CODE GENERATION SAMPLES", flush=True)
    print("="*85, flush=True)
    prompts = [
        "import mathutils\nfrom mathutils import Vector, Matrix\n# Dot product of two vectors\n",
        "import bpy\nimport bmesh\n# Create parametric mesh with bmesh\ndef create_mesh():\n    bm = bmesh.",
        "# Möller-Trumbore ray-triangle intersection\ndef intersect_ray_triangle("
    ]
    
    for p_idx, prompt in enumerate(prompts, 1):
        print(f"\n>>> [PROMPT {p_idx}]: {repr(prompt)}", flush=True)
        res, routes = ensemble.generate(prompt, max_new_tokens=35, k=2, temperature=0.6)
        new_text = res[len(prompt):]
        print(f"Generated ({routes}):\n{repr(new_text)}", flush=True)

if __name__ == '__main__':
    main()
