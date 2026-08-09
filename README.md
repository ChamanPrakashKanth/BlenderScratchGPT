# BlenderScratchGPT

A small experimental GPT-style language model trained on the Blender 5.1 Python API documentation.

The project explores how a custom Transformer can learn Blender Python API patterns from documentation, with the longer-term goal of specializing a pretrained coding model such as Qwen2.5-Coder-3B for Blender-specific coding assistance.

## Project Status

**Experimental / Research Project**

Current progress:

- Blender 5.1 Python API documentation obtained as offline HTML
- Documentation converted into a clean text corpus
- Custom tokenizer and PyTorch dataset pipeline
- Custom TinyGPT implemented in PyTorch
- Custom Fourier positional encoding
- Custom `ComplexReLU` activation
- 4 Transformer blocks with self-attention
- Model trained successfully for 2,000 steps
- Approximately 853K trainable parameters
- Training loss reduced from **5.7320 → 0.3795**

The next stage is evaluation and Blender code generation, followed by Qwen2.5-Coder-3B LoRA/QLoRA specialization.

## Architecture

```text
Token IDs
    ↓
Token Embedding
    ↓
Fourier Positional Encoding
    ↓
Transformer Block × 4
    │
    ├── LayerNorm
    ├── Self-Attention
    ├── LayerNorm
    └── Complex Feed Forward
           │
           ├── Linear
           ├── ComplexReLU
           └── Linear
    ↓
LayerNorm
    ↓
Language Model Head
    ↓
Next-token logits
```

## Current Model

| Parameter | Value |
|---|---:|
| Parameters | ~852,968 |
| Vocabulary size | 232 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Context length | 128 |
| Batch size | 32 |
| Learning rate | 3e-4 |
| Training steps | 2,000 |

## Training Result

The first training run produced:

```text
Step       Train Loss
---------------------
1          5.7320
100        2.3949
500        1.1262
1000       0.8207
1500       0.5107
2000       0.3795
```

The overall training loss decreased substantially, showing that the model learned statistical patterns from the Blender API corpus.

**Important:** training loss alone does not demonstrate that the model understands Blender or generates correct Python. Validation loss, held-out examples, and actual code execution are required for meaningful evaluation.

## Dataset

The corpus is derived from the official Blender 5.1 Python API documentation.

Relevant API areas include:

- `bpy`
- `bpy.types`
- `bpy.ops`
- `bpy.data`
- `bpy.context`
- `bpy.props`
- `bmesh`
- `mathutils`
- `gpu`
- `bpy_extras`

The documentation contains Blender classes, methods, properties, functions, parameters, descriptions, and Python examples.

The offline HTML documentation is parsed and converted into text before tokenization.

## Project Structure

```text
BlenderScratchGPT/
│
├── dataset.py
├── model.py
├── train.py
├── tokenizer.py
├── README.md
├── requirements.txt
│
├── data/
│   └── tokenized.pt
│
├── checkpoints/
│   └── blender_gpt.pt
│
└── blender_api_5_1/
    └── extracted Blender documentation
```

Large generated files and the local Python environment should not be committed to GitHub.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install torch
pip install beautifulsoup4
```

Or, if `requirements.txt` is provided:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Extract the Blender 5.1 offline HTML documentation and place it in the project directory.

Then run:

```bash
python dataset.py
```

This prepares the documentation for training and produces the tokenized dataset.

## Training

Run:

```bash
python train.py
```

The current training configuration uses:

```python
BLOCK_SIZE = 128
BATCH_SIZE = 32

N_EMBED = 128
N_HEAD = 4
N_LAYER = 4

LEARNING_RATE = 3e-4
STEPS = 2000
```

The trained checkpoint is saved as:

```text
checkpoints/blender_gpt.pt
```

## Intended Capabilities

The eventual Blender coding assistant should be able to help with tasks such as:

```text
Create a cube and move it 2 meters along X.
```

Potential output:

```python
import bpy

obj = bpy.context.object
obj.location.x += 2
```

Potential capabilities include:

- Blender Python code generation
- `bpy` API assistance
- API lookup
- Procedural modeling
- Scene manipulation
- Blender automation
- Add-on development
- Python debugging

## Evaluation Plan

Future evaluation will include:

1. Training loss
2. Validation loss
3. Held-out Blender API prompts
4. Code generation quality
5. Python syntax correctness
6. Blender execution tests
7. API accuracy
8. Comparison between TinyGPT and Qwen2.5-Coder-3B
9. LoRA/QLoRA performance

The most important test is not simply whether the model produces plausible code, but whether the generated code actually executes correctly inside Blender.

## Why TinyGPT?

The custom model is intentionally small.

The goal is to understand the complete pipeline:

```text
Documentation
    ↓
Tokenization
    ↓
Dataset
    ↓
Embeddings
    ↓
Attention
    ↓
Transformer blocks
    ↓
Loss
    ↓
Backpropagation
    ↓
Generation
```

This makes the project useful as a controlled experiment before moving to a much larger pretrained coding model.

## Disclaimer

This is an experimental educational/research project.

A low training loss does not necessarily mean that the model has learned reliable Blender reasoning or that generated code is correct.

Blender documentation and Blender itself are subject to their respective licenses. Refer to the official Blender project for licensing information.

## Future Work

- Add validation loss
- Implement text generation
- Test generated code in Blender
- Improve dataset structure
- Generate instruction-response examples
- Fine-tune Qwen2.5-Coder-3B
- Compare full fine-tuning vs LoRA/QLoRA
- Build a Blender-side coding assistant
- Evaluate generated scripts automatically

---

**BlenderScratchGPT — from Blender documentation to a custom Transformer, and eventually to a Blender-specialized coding model.**
