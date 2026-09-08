"""
Top-K Mixture-of-Experts (MoE) Ensemble Router for Sparse-AST Models:
Connects 3M, 10M, 100M, 200M, and 500M models with dynamic Top-K routing (k=1..N)
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
    Computes routing logits over all available Sparse-AST expert models:
    Expert 0: 3M Model   (Fast, ultra-low latency syntax drafting)
    Expert 1: 10M Model  (Intermediate structural capacity)
    Expert 2: 100M Model (Deep 3D math, geometry, and bmesh reasoning)
    Expert 3: 200M Model (Advanced multi-block procedural modeling)
    Expert 4: 500M Model (High-capacity 500M parameter foundation reasoning)
    """
    def __init__(self, vocab_size=512, hidden_dim=64, num_experts=5):
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
    Unified MoE Ensemble connecting all pretrained Sparse-AST checkpoints with Top-K routing.
    Dynamically discovers 3M, 10M, 100M, 200M, and 500M models.
    """
    def __init__(self, model_paths=None, device=None, k=2, tau=1.0):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.k = k
        self.tau = tau
        
        default_expert_configs = [
            ("3M", r"c:\Users\user\Downloads\checkpoint\final_sparse_ast.pt"),
            ("10M", r"c:\Users\user\Downloads\checkpoint\final_sparse_ast_10m.pt"),
            ("100M", r"c:\Users\user\Downloads\checkpoint\final_sparse_ast_100m.pt"),
            ("200M", r"c:\Users\user\Downloads\checkpoint\final_sparse_ast_200m.pt"),
            ("500M", r"c:\Users\user\Downloads\checkpoint\final_sparse_ast_500m.pt"),
        ]
        
        self.experts = nn.ModuleList()
        self.expert_infos = []
        self.expert_names = []
        
        if model_paths:
            configs_to_try = [(f"Expert_{i}", p) for i, p in enumerate(model_paths)]
        else:
            configs_to_try = default_expert_configs
            
        for name, path in configs_to_try:
            if os.path.exists(path):
                print(f"[Top-K Ensemble] Loading expert [{name}]: {os.path.basename(path)}...", flush=True)
                m, info = auto_load_model(path, target_seq_len=4096)
                m.to(device)
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False  # Freeze pretrained backbones
                self.experts.append(m)
                self.expert_infos.append(info)
                self.expert_names.append(name)
            else:
                print(f"[Top-K Ensemble] Expert [{name}] not found at {path} (pending download).", flush=True)
                
        self.num_experts = len(self.experts)
        if self.num_experts == 0:
            raise RuntimeError("No Sparse-AST checkpoints found to build Top-K Ensemble!")
            
        self.k = min(self.k, self.num_experts)
        self.min_seq_len = min(info['seq_len'] for info in self.expert_infos)
        
        # Learnable Router
        self.router = TopKRouter(vocab_size=512, hidden_dim=64, num_experts=self.num_experts).to(device)
        print(f"[Top-K Ensemble] Initialized with {self.num_experts} active experts: {self.expert_names} (k={self.k})", flush=True)

    def forward_routing(self, x, k=None):
        """
        Computes Top-K gated logits for sequence x (Batch, Seq).
        """
        if k is None:
            k = self.k
        k = max(1, min(k, self.num_experts))
        
        if x.shape[1] > self.min_seq_len:
            x_input = x[:, -self.min_seq_len:]
        else:
            x_input = x
            
        b, t = x_input.shape
        router_logits = self.router(x_input) # (B, T, num_experts)
        
        topk_scores, topk_indices = torch.topk(router_logits, k=k, dim=-1) # (B, T, k)
        gate_weights = F.softmax(topk_scores / self.tau, dim=-1)            # (B, T, k)
        
        unique_experts = torch.unique(topk_indices).tolist()
        expert_logits = {}
        with torch.no_grad():
            for exp_idx in unique_experts:
                expert_logits[exp_idx] = self.experts[exp_idx](x_input).detach() # (B, T, 512)
            
        combined_logits = torch.zeros((b, t, 512), device=self.device, dtype=torch.float32)
        for i in range(k):
            idx_i = topk_indices[:, :, i]        # (B, T)
            weight_i = gate_weights[:, :, i:i+1] # (B, T, 1)
            for exp_idx in unique_experts:
                mask = (idx_i == exp_idx).unsqueeze(-1) # (B, T, 1)
                if mask.any():
                    combined_logits += weight_i * expert_logits[exp_idx] * mask.float()
                    
        return combined_logits, router_logits, topk_indices, gate_weights

    def evaluate(self, text_data, seq_len=1024, num_samples=20, k=None):
        """
        Evaluates cross-entropy loss and perplexity on test curriculum text.
        """
        self.eval()
        losses = []
        expert_counts = {i: 0 for i in range(self.num_experts)}
        total_tokens = 0
        
        with torch.no_grad():
            max_start = len(text_data) - seq_len - 1
            if max_start <= 0:
                seq_len = len(text_data) - 2
                max_start = 1
                
            for i in range(min(num_samples, max_start)):
                idx = (i * 317) % max_start
                x = text_data[idx : idx + seq_len].unsqueeze(0).to(self.device)
                y = text_data[idx + 1 : idx + seq_len + 1].unsqueeze(0).to(self.device)
                logits, _, topk_indices, _ = self.forward_routing(x, k=k)
                
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                losses.append(float(loss))
                
                for exp_idx in range(self.num_experts):
                    expert_counts[exp_idx] += int((topk_indices == exp_idx).sum().item())
                total_tokens += topk_indices.numel()
                
        avg_loss = sum(losses) / len(losses)
        ppl = math.exp(min(avg_loss, 20.0))
        expert_dist = {exp: (expert_counts[exp] / max(total_tokens, 1)) * 100 for exp in expert_counts}
        return avg_loss, ppl, expert_dist

    def generate(self, prompt_text, max_new_tokens=40, k=2, temperature=0.7, top_k_tokens=40):
        """
        Autoregressive code generation with Top-K expert model routing & Top-K token sampling.
        """
        self.eval()
        encoded = list(prompt_text.encode('utf-8', 'ignore'))
        token_routes = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                cur_input = encoded[-self.min_seq_len:]
                x = torch.tensor([cur_input], dtype=torch.long, device=self.device)
                
                logits, _, topk_indices, gate_weights = self.forward_routing(x, k=k)
                last_logits = logits[:, -1, :] # (1, 512)
                
                best_expert = int(topk_indices[0, -1, 0].item())
                token_routes.append(best_expert)
                
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
        route_summary = {self.expert_names[i]: token_routes.count(i) for i in range(self.num_experts) if token_routes.count(i) > 0}
        return res, route_summary

def train_topk_router(ensemble, text_data, steps=60, lr=2e-3, seq_len=64, batch_size=2):
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
        
        ce_loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        
        # Load balancing regularizer across all active experts
        router_probs = F.softmax(router_logits, dim=-1).mean(dim=[0, 1]) # (num_experts,)
        entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-6))
        balance_loss = -0.05 * entropy
        
        total_loss = ce_loss + balance_loss
        total_loss.backward()
        opt.step()
        
        if step % 20 == 0 or step == steps:
            dist_str = ", ".join([f"{ensemble.expert_names[i]}:{float(router_probs[i]):.3f}" for i in range(ensemble.num_experts)])
            print(f"Step {step:03d}/{steps} | CrossEntropy: {float(ce_loss):.4f} | Dist: [{dist_str}]", flush=True)
            
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
    
    # Train router on curriculum
    train_topk_router(ensemble, curr_data, steps=40, lr=2e-3)
    
    # Save calibrated router checkpoint
    save_path = r"c:\Users\user\Downloads\checkpoint\topk_sparse_ast_router.pt"
    torch.save({
        'router_state': ensemble.router.state_dict(),
        'k': ensemble.k,
        'tau': ensemble.tau,
        'num_experts': ensemble.num_experts,
        'expert_names': ensemble.expert_names,
        'config': f'TopK-MoE-SparseAST-{"-".join(ensemble.expert_names)}'
    }, save_path)
    print(f"\n[+] Saved trained Top-K MoE router checkpoint to {save_path}!", flush=True)

if __name__ == '__main__':
    main()
