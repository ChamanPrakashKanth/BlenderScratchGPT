"""
Interactive Terminal Chat Interface for Sparse-AST Models & Top-K MoE Ensemble
Supports 3M, 10M, 100M, 200M models and the Top-K MoE Ensemble Router.
"""

import os
import sys
import math
import time
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F

from test_models import auto_load_model
from topk_ensemble import TopKSparseASTEnsemble

class ChatApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = r"c:\Users\user\Downloads\checkpoint"
        self.router_path = os.path.join(self.checkpoint_dir, "topk_sparse_ast_router.pt")
        
        self.available_models = {
            "1": ("Top-K MoE Ensemble (3M+10M+100M)", "ensemble"),
            "2": ("Sparse-AST 100M", os.path.join(self.checkpoint_dir, "final_sparse_ast_100m.pt")),
            "3": ("Sparse-AST 200M", os.path.join(self.checkpoint_dir, "final_sparse_ast_200m.pt")),
            "4": ("Sparse-AST 10M", os.path.join(self.checkpoint_dir, "final_sparse_ast_10m.pt")),
            "5": ("Sparse-AST 3M", os.path.join(self.checkpoint_dir, "final_sparse_ast.pt"))
        }
        
        self.current_model_key = "1"
        self.loaded_models = {}
        self.k = 2
        self.temperature = 0.6
        self.max_tokens = 45
        self.top_k_tokens = 40
        
    def load_model(self, key):
        if key in self.loaded_models:
            return self.loaded_models[key]
            
        name, path = self.available_models[key]
        print(f"\n[+] Loading {name} onto {self.device}...", flush=True)
        
        if path == "ensemble":
            ensemble = TopKSparseASTEnsemble(device=self.device, k=self.k)
            if os.path.exists(self.router_path):
                router_data = torch.load(self.router_path, map_location=self.device)
                ensemble.router.load_state_dict(router_data['router_state'])
                print(f"[+] Loaded calibrated Top-K router weights from {os.path.basename(self.router_path)}.", flush=True)
            self.loaded_models[key] = (ensemble, {'name': name, 'is_ensemble': True})
            return self.loaded_models[key]
            
        # Check if 200M final exists or interim checkpoint
        if not os.path.exists(path):
            if "200M" in name:
                alt = sorted([p for p in os.listdir(self.checkpoint_dir) if p.startswith("checkpoint_200m_") and p.endswith(".pt")])
                if alt:
                    path = os.path.join(self.checkpoint_dir, alt[-1])
                    print(f"[*] Using latest available 200M checkpoint: {alt[-1]}", flush=True)
                else:
                    print(f"[-] 200M checkpoint not found locally yet (training on Kaggle). Defaulting to Top-K Ensemble.", flush=True)
                    self.current_model_key = "1"
                    return self.load_model("1")
            else:
                print(f"[-] Checkpoint file {path} not found.", flush=True)
                self.current_model_key = "1"
                return self.load_model("1")
                
        model, info = auto_load_model(path)
        model.to(self.device)
        model.eval()
        self.loaded_models[key] = (model, info)
        return self.loaded_models[key]

    def stream_generate(self, prompt_text):
        model_obj, info = self.load_model(self.current_model_key)
        encoded = list(prompt_text.encode('utf-8', 'ignore'))
        is_ensemble = info.get('is_ensemble', False)
        seq_limit = 31 if is_ensemble else min(getattr(model_obj, 'seq_len', 32) - 1, 31)
        
        expert_names = ["3M", "10M", "100M"]
        token_routes = []
        new_tokens = []
        
        sys.stdout.write("\nAssistant: ")
        sys.stdout.flush()
        
        with torch.no_grad():
            for _ in range(self.max_tokens):
                cur_input = encoded[-seq_limit:]
                x = torch.tensor([cur_input], dtype=torch.long, device=self.device)
                
                if is_ensemble:
                    logits, _, topk_indices, _ = model_obj.forward_routing(x, k=self.k)
                    last_logits = logits[:, -1, :]
                    best_expert = int(topk_indices[0, -1, 0].item())
                    token_routes.append(best_expert)
                else:
                    last_logits = model_obj(x)[:, -1, :]
                    
                if self.temperature <= 0.05:
                    next_token = int(last_logits.argmax(dim=-1).item())
                else:
                    scaled = last_logits / self.temperature
                    if self.top_k_tokens > 0:
                        v, _ = torch.topk(scaled, min(self.top_k_tokens, scaled.size(-1)))
                        scaled[scaled < v[:, [-1]]] = -float('inf')
                    probs = F.softmax(scaled, dim=-1)
                    next_token = int(torch.multinomial(probs, num_samples=1).item())
                    
                encoded.append(next_token)
                new_tokens.append(next_token)
                
                # Stream char in real-time
                if next_token < 256:
                    char = bytes([next_token]).decode('utf-8', errors='replace')
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    
        print()
        if is_ensemble and token_routes:
            summary = {expert_names[i]: token_routes.count(i) for i in range(3) if token_routes.count(i) > 0}
            print(f"  [MoE Routing: {', '.join(f'{k}: {v} tokens' for k, v in summary.items())}]", flush=True)

    def print_banner(self):
        print("="*75)
        print("     SPARSE-AST INTERACTIVE CHAT & CODE GENERATION CONSOLE")
        print("="*75)
        print("Models Available:")
        for k, (name, _) in self.available_models.items():
            marker = " [ACTIVE]" if k == self.current_model_key else ""
            print(f"  [{k}] {name}{marker}")
        print("\nCommands:")
        print("  /model <1-5>   - Switch active model")
        print("  /k <1|2|3>     - Change Top-K MoE routing depth")
        print("  /temp <float>  - Set sampling temperature (e.g. 0.2 to 1.0)")
        print("  /tokens <int>  - Set max output tokens (e.g. 35, 60)")
        print("  /help          - Show sample prompts (Blender vectors, bmesh, shaders)")
        print("  exit / quit    - Exit console")
        print("="*75)

    def run(self):
        self.print_banner()
        self.load_model(self.current_model_key)
        
        while True:
            try:
                cur_name = self.available_models[self.current_model_key][0]
                user_input = input(f"\nUser [{cur_name}] > ").strip()
                if not user_input:
                    continue
                    
                if user_input.lower() in ['exit', 'quit', ':q']:
                    print("Goodbye!")
                    break
                    
                if user_input.startswith("/"):
                    parts = user_input.split()
                    cmd = parts[0].lower()
                    
                    if cmd == "/model" and len(parts) > 1:
                        target = parts[1]
                        if target in self.available_models:
                            self.current_model_key = target
                            self.load_model(target)
                            print(f"[+] Switched active model to: {self.available_models[target][0]}")
                        else:
                            print(f"[-] Invalid model choice. Select 1 to 5.")
                    elif cmd == "/k" and len(parts) > 1:
                        try:
                            val = int(parts[1])
                            if 1 <= val <= 3:
                                self.k = val
                                if self.current_model_key in self.loaded_models:
                                    self.loaded_models[self.current_model_key][0].k = val
                                print(f"[+] Top-K set to: k={self.k}")
                            else:
                                print("[-] k must be 1, 2, or 3.")
                        except ValueError:
                            print("[-] Invalid number for k.")
                    elif cmd == "/temp" and len(parts) > 1:
                        try:
                            self.temperature = max(0.01, min(2.0, float(parts[1])))
                            print(f"[+] Temperature set to: {self.temperature:.2f}")
                        except ValueError:
                            print("[-] Invalid float for temperature.")
                    elif cmd == "/tokens" and len(parts) > 1:
                        try:
                            self.max_tokens = max(10, min(200, int(parts[1])))
                            print(f"[+] Max tokens set to: {self.max_tokens}")
                        except ValueError:
                            print("[-] Invalid int for tokens.")
                    elif cmd == "/help":
                        print("\nSample Prompts to Try:")
                        print("  1. import mathutils\n     from mathutils import Vector\n     v1 = Vector((")
                        print("  2. import bpy\n     import bmesh\n     bm = bmesh.new()")
                        print("  3. # Calculate 3D dot product\n     def dot_product(v1, v2):")
                        print("  4. # 4x4 matrix transformation\n     mat = Matrix.Rotation(")
                    else:
                        print("[-] Unknown command. Type /help or enter code prompt.")
                    continue
                    
                self.stream_generate(user_input)
                
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except Exception as e:
                print(f"\n[-] Error: {e}")

if __name__ == '__main__':
    app = ChatApp()
    app.run()
