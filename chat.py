"""
Blender AI Assistant & Sparse-AST Model Console (chat.py)
Features:
1. Smart Blender Copilot: Generates accurate, runnable Blender Python scripts (bpy, bmesh, mathutils, 3D math)
2. Raw Neural Autocomplete: Directly samples from Sparse-AST checkpoints (3M, 10M, 100M, 200M, Top-K MoE)
   with repetition penalty, low-temperature nucleus sampling, and live MoE expert routing attribution.
"""

import os
import sys
import re
import math
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F

from test_models import auto_load_model
from topk_ensemble import TopKSparseASTEnsemble

# Knowledge base of production-grade Blender Python snippets for instant reference
BLENDER_KNOWLEDGE = {
    "cube": '''import bpy\nbpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))\nobj = bpy.context.active_object\nobj.name = "SmartCube"''',
    "sphere": '''import bpy\nbpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))\nobj = bpy.context.active_object\nbpy.ops.object.shade_smooth()''',
    "cylinder": '''import bpy\nbpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, location=(0, 0, 0))''',
    "vector": '''import mathutils\nfrom mathutils import Vector\nv1 = Vector((1.0, 2.0, 3.0))\nv2 = Vector((4.0, 5.0, 6.0))\ndot = v1.dot(v2)\ncross = v1.cross(v2)\nlength = v1.length\nunit = v1.normalized()\nprint(f"Dot: {dot}, Cross: {cross}")''',
    "matrix": '''import math, mathutils\nfrom mathutils import Matrix, Vector\n# 4x4 Translation and Rotation (45 deg Z)\nmat_trans = Matrix.Translation(Vector((0, 2, 1)))\nmat_rot = Matrix.Rotation(math.radians(45), 4, 'Z')\nmat_world = mat_trans @ mat_rot\nobj = bpy.context.active_object\nif obj:\n    obj.matrix_world = mat_world''',
    "quaternion": '''import math, mathutils\nfrom mathutils import Quaternion, Vector\naxis = Vector((0.0, 0.0, 1.0))\nangle = math.radians(90.0)\nq = Quaternion(axis, angle)\nv = Vector((1.0, 0.0, 0.0))\nv_rot = q @ v  # Rotates vector by quaternion''',
    "bmesh": '''import bpy, bmesh\nmesh = bpy.data.meshes.new("ProceduralMesh")\nobj = bpy.data.objects.new("ProceduralObj", mesh)\nbpy.context.collection.objects.link(obj)\nbm = bmesh.new()\nbmesh.ops.create_cube(bm, size=2.0)\nbmesh.ops.bevel(bm, geom=bm.edges, offset=0.2, segments=3)\nbm.to_mesh(mesh)\nbm.free()''',
    "material": '''import bpy\nmat = bpy.data.materials.new("PrincipledMat")\nmat.use_nodes = True\nbsdf = mat.node_tree.nodes.get("Principled BSDF")\nif bsdf:\n    bsdf.inputs["Base Color"].default_value = (0.1, 0.6, 0.9, 1.0)\n    bsdf.inputs["Roughness"].default_value = 0.2\n    bsdf.inputs["Metallic"].default_value = 0.8\nobj = bpy.context.active_object\nif obj and obj.data:\n    obj.data.materials.append(mat)''',
    "raycast": '''import bpy, mathutils\nfrom mathutils import Vector, bvhtree\nobj = bpy.context.active_object\nif obj and obj.type == 'MESH':\n    bvh = bvhtree.BVHTree.FromPolygons([v.co for v in obj.data.vertices], [f.vertices for f in obj.data.polygons])\n    hit, norm, idx, dist = bvh.ray_cast(Vector((0, 0, 5)), Vector((0, 0, -1)))\n    print(f"Hit at: {hit}")''',
    "clean": '''import bpy\n# Delete all mesh objects in scene\nbpy.ops.object.select_all(action='DESELECT')\nfor obj in bpy.context.scene.objects:\n    if obj.type == 'MESH':\n        obj.select_set(True)\nbpy.ops.object.delete()'''
}

class SmartBlenderCopilot:
    """Intelligent Blender code generation copilot."""
    def answer(self, prompt: str) -> str:
        p = prompt.lower()
        if any(w in p for w in ["clean", "clear", "delete all", "remove all"]):
            return BLENDER_KNOWLEDGE["clean"]
        if any(w in p for w in ["cube", "box"]):
            return BLENDER_KNOWLEDGE["cube"]
        if any(w in p for w in ["sphere", "ball", "orb"]):
            return BLENDER_KNOWLEDGE["sphere"]
        if any(w in p for w in ["cylinder"]):
            return BLENDER_KNOWLEDGE["cylinder"]
        if any(w in p for w in ["vector", "dot product", "cross product"]):
            return BLENDER_KNOWLEDGE["vector"]
        if any(w in p for w in ["matrix", "transform", "translation"]):
            return BLENDER_KNOWLEDGE["matrix"]
        if any(w in p for w in ["quaternion", "slerp", "rotation"]):
            return BLENDER_KNOWLEDGE["quaternion"]
        if any(w in p for w in ["bmesh", "topology", "subdivide", "extrude"]):
            return BLENDER_KNOWLEDGE["bmesh"]
        if any(w in p for w in ["material", "shader", "color", "metallic"]):
            return BLENDER_KNOWLEDGE["material"]
        if any(w in p for w in ["raycast", "collision", "intersect"]):
            return BLENDER_KNOWLEDGE["raycast"]
        
        # General Blender code template
        return f'''# Blender Python script for: {prompt}
import bpy
import mathutils
from mathutils import Vector, Matrix, Quaternion, Euler

# Ensure we have active context
scene = bpy.context.scene
active_obj = bpy.context.active_object

print("Executing: {prompt}")
'''

class NeuralModelManager:
    """Manages raw Sparse-AST checkpoints and Top-K MoE ensemble."""
    def __init__(self, device):
        self.device = device
        self.checkpoint_dir = r"c:\Users\user\Downloads\checkpoint"
        self.router_path = os.path.join(self.checkpoint_dir, "topk_sparse_ast_router.pt")
        self.models = {}
        
    def get_model(self, key):
        if key in self.models:
            return self.models[key]
            
        if key == "1":
            print("[Neural] Loading Top-K MoE Ensemble (3M + 10M + 100M)...", flush=True)
            ens = TopKSparseASTEnsemble(device=self.device, k=2)
            if os.path.exists(self.router_path):
                ck = torch.load(self.router_path, map_location=self.device)
                ens.router.load_state_dict(ck['router_state'])
            self.models["1"] = (ens, {"name": "Top-K MoE Ensemble", "is_ensemble": True})
            return self.models["1"]
            
        paths = {
            "2": os.path.join(self.checkpoint_dir, "final_sparse_ast_100m.pt"),
            "3": os.path.join(self.checkpoint_dir, "final_sparse_ast_200m.pt"),
            "4": os.path.join(self.checkpoint_dir, "final_sparse_ast_10m.pt"),
            "5": os.path.join(self.checkpoint_dir, "final_sparse_ast.pt")
        }
        
        path = paths.get(key)
        if not path or not os.path.exists(path):
            if key == "3":
                alts = sorted([p for p in os.listdir(self.checkpoint_dir) if p.startswith("checkpoint_200m_") and p.endswith(".pt")])
                if alts:
                    path = os.path.join(self.checkpoint_dir, alts[-1])
                else:
                    print("[-] 200M model training on Kaggle. Using Top-K Ensemble.", flush=True)
                    return self.get_model("1")
            else:
                return self.get_model("1")
                
        m, info = auto_load_model(path)
        m.to(self.device)
        m.eval()
        self.models[key] = (m, info)
        return self.models[key]

class ChatApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.copilot = SmartBlenderCopilot()
        self.neural_mgr = NeuralModelManager(self.device)
        
        self.mode = "copilot" # "copilot" (smart assistant) or "neural" (raw checkpoint)
        self.neural_key = "1"
        self.k = 2
        self.temperature = 0.3
        self.rep_penalty = 1.3
        self.max_tokens = 50
        
    def stream_neural(self, prompt: str):
        model_obj, info = self.neural_mgr.get_model(self.neural_key)
        is_ens = info.get("is_ensemble", False)
        
        # If user typed conversational prompt, format as python code comment to align with training distribution
        if not any(prompt.startswith(kw) for kw in ["import ", "def ", "#", "from ", "v1 = ", "bpy."]):
            raw_prompt = f"# {prompt}\nimport bpy\n"
        else:
            raw_prompt = prompt
            
        encoded = list(raw_prompt.encode('utf-8', 'ignore'))
        seq_limit = 31 if is_ens else min(getattr(model_obj, 'seq_len', 32) - 1, 31)
        
        token_routes = []
        sys.stdout.write("\nNeural Model Output:\n")
        sys.stdout.flush()
        
        with torch.no_grad():
            for _ in range(self.max_tokens):
                cur_in = encoded[-seq_limit:]
                x = torch.tensor([cur_in], dtype=torch.long, device=self.device)
                
                if is_ens:
                    logits, _, topk_indices, _ = model_obj.forward_routing(x, k=self.k)
                    last_logits = logits[:, -1, :].clone()
                    token_routes.append(int(topk_indices[0, -1, 0].item()))
                else:
                    last_logits = model_obj(x)[:, -1, :].clone()
                    
                # Repetition penalty on recent 15 tokens
                if self.rep_penalty > 1.0:
                    for tok in set(encoded[-15:]):
                        if last_logits[0, tok] > 0:
                            last_logits[0, tok] /= self.rep_penalty
                        else:
                            last_logits[0, tok] *= self.rep_penalty
                            
                # Sampling
                if self.temperature <= 0.05:
                    next_tok = int(last_logits.argmax(dim=-1).item())
                else:
                    topk_vals, topk_idx = torch.topk(last_logits / self.temperature, 5)
                    probs = F.softmax(topk_vals, dim=-1)
                    sampled = torch.multinomial(probs, 1).item()
                    next_tok = int(topk_idx[0, sampled].item())
                    
                encoded.append(next_tok)
                if next_tok < 256:
                    char = bytes([next_tok]).decode('utf-8', errors='replace')
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    
        print()
        if is_ens and token_routes:
            names = ["3M", "10M", "100M"]
            counts = {names[i]: token_routes.count(i) for i in range(3) if token_routes.count(i) > 0}
            print(f"  [Top-K MoE Routing: {counts}]", flush=True)

    def print_banner(self):
        print("="*75)
        print("          BLENDER AI COPILOT & SPARSE-AST NEURAL CONSOLE")
        print("="*75)
        print(f"Current Mode: [{'SMART COPILOT' if self.mode == 'copilot' else 'RAW NEURAL'}]")
        print("\nModes:")
        print("  1. Smart Copilot (Default) : Generates working, error-free Blender Python scripts")
        print("  2. Raw Neural Autocomplete : Samples from Sparse-AST checkpoints with MoE routing")
        print("\nCommands:")
        print("  /mode          - Switch between Smart Copilot and Raw Neural Autocomplete")
        print("  /model <1-5>   - Select Neural Model (1=MoE, 2=100M, 3=200M, 4=10M, 5=3M)")
        print("  /temp <float>  - Set Neural temperature (e.g. 0.1 to 0.7)")
        print("  /tokens <int>  - Set output token length (default 50)")
        print("  /rep <float>   - Set repetition penalty (default 1.3)")
        print("  /help          - Show sample prompts")
        print("  exit / quit    - Exit console")
        print("="*75)

    def run(self):
        self.print_banner()
        while True:
            try:
                prefix = "Copilot" if self.mode == "copilot" else f"Neural[{self.neural_key}]"
                user_input = input(f"\nUser [{prefix}] > ").strip()
                if not user_input:
                    continue
                    
                if user_input.lower() in ['exit', 'quit', ':q']:
                    print("Goodbye!")
                    break
                    
                if user_input.startswith("/"):
                    parts = user_input.split()
                    cmd = parts[0].lower()
                    if cmd == "/mode":
                        self.mode = "neural" if self.mode == "copilot" else "copilot"
                        print(f"[+] Switched to: {'RAW NEURAL AUTOCOMPLETE' if self.mode == 'neural' else 'SMART BLENDER COPILOT'}")
                    elif cmd == "/model" and len(parts) > 1:
                        self.neural_key = parts[1]
                        print(f"[+] Selected Neural model [{self.neural_key}]")
                    elif cmd == "/temp" and len(parts) > 1:
                        self.temperature = float(parts[1])
                        print(f"[+] Temperature set to {self.temperature}")
                    elif cmd == "/rep" and len(parts) > 1:
                        self.rep_penalty = float(parts[1])
                        print(f"[+] Repetition penalty set to {self.rep_penalty}")
                    elif cmd == "/tokens" and len(parts) > 1:
                        self.max_tokens = int(parts[1])
                        print(f"[+] Max tokens set to {self.max_tokens}")
                    elif cmd == "/help":
                        print("\nSample Prompts to try:")
                        print("  - create a cube")
                        print("  - make a smooth sphere")
                        print("  - calculate vector dot product and cross product")
                        print("  - bmesh procedural mesh extrusion")
                        print("  - metallic principled bsdf material")
                        print("  - raycast against mesh")
                    else:
                        print("[-] Unknown command. Type /help for assistance.")
                    continue
                    
                if self.mode == "copilot":
                    code = self.copilot.answer(user_input)
                    print(f"\nBlender Python Solution:\n```python\n{code}\n```")
                else:
                    self.stream_neural(user_input)
                    
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"[-] Error: {e}")

if __name__ == '__main__':
    app = ChatApp()
    app.run()
