# BlenderScratchGPT

A lightweight research project for training a small GPT-style language model on Blender 5.1 Python API documentation.

The goal is to build a compact Blender-focused code assistant that can learn patterns from the official API docs and generate sensible Blender Python snippets, with the longer-term possibility of extending the model with parameter-efficient fine-tuning such as LoRA or QLoRA.

## What this project does

- Scrapes Blender API reference pages from local HTML documentation
- Cleans and normalizes the extracted text
- Builds a custom character-level vocabulary and tokenized dataset
- Trains a compact PyTorch Transformer model from scratch
- Generates Blender-related text and Python code from a prompt
- Includes exploratory scientific ML research experiments alongside the Blender model work

This is an experimental project rather than a production Blender plugin or an officially supported AI tool.

## Research experiments

The repository also contains a separate exploratory script, `test.py`, which demonstrates a Physics-Informed Neural Network (PINN) for a mass-spring-damper system.

This research prototype:

- defines a neural network model for the displacement $x(t)$
- computes the residual of the ODE $m \ddot{x} + c \dot{x} + kx = 0$
- enforces the initial conditions via an additional loss term
- trains the model with automatic differentiation in PyTorch
- plots the predicted trajectory over time

This experiment is not part of the Blender documentation model itself, but it shows a parallel research direction focused on scientific machine learning and differentiable physics. It is useful as a reference for testing optimization, autodiff, and PINN-style training workflows in the same project environment.

### Adaptive activation routing

`activation_router_experiment_numpy.py` contains a controlled, self-contained
language-model experiment derived from the repository's decoder architecture.
It tests three feed-forward activation paths:

1. the existing ComplexReLU baseline
2. an adaptive per-hidden-dimension router over linear, signed-log, and
   stabilized-exponential transformations
3. the composed path requested by the experiment, where routing happens first
   and ComplexReLU activates the routed value:

$$
z \rightarrow R(z) \rightarrow \operatorname{ComplexReLU}(R(z))
$$

The arms use identical initialized model tensors, parameter count, seed,
pre-generated batches, optimizer, and training budget. The baseline carries
dormant router-shaped parameters to keep the total parameter count matched.

#### Executed smoke-test results

| Activation path | Final train loss | Validation loss | Median steps/s |
|---|---:|---:|---:|
| ComplexReLU | 2.9571 | 3.1509 | 262.58 |
| Adaptive router | **2.9396** | **3.1110** | 99.85 |
| ComplexReLU over router | 2.9577 | 3.1474 | 99.86 |

All three arms completed 160/160 steps with finite losses and gradients. Pure
adaptive routing reduced validation loss by 1.27% relative to ComplexReLU, while
ComplexReLU over the router reduced it by only 0.11%. The composed model's soft
router probabilities remained close to uniform; hard selections mildly favored
signed-log, particularly in the second layer.

These numbers are a single-seed CPU smoke test, not a general performance claim.
The ignored Blender documentation corpus and `data/tokenized.pt` were not
available in the execution environment, so the frozen character corpus was
built from the repository's original tracked Python and Markdown files. The
test also uses a reduced 20,352-parameter NumPy/autodiff decoder. A substantive
conclusion requires the real Blender corpus, multiple seeds, confidence
intervals, and replication in the full PyTorch model.

Run the reproducible experiment with:

```bash
python activation_router_experiment_numpy.py
```

The generated report and raw loss/gradient curves are stored under
`experiment_results/`.

## Project status

Current progress includes:

- Blender 5.1 Python API documentation collected locally
- HTML docs converted into a clean text corpus
- Character-level tokenizer and dataset pipeline implemented
- Tiny GPT-style Transformer implemented in PyTorch
- Fourier positional encoding and custom activation logic added
- Training pipeline working with a small model and checkpoint saving
- Generation loop implemented for interactive prompting

The current model is intentionally small and designed as a research baseline rather than a fully robust Blender assistant.

## Model overview

The training setup uses a compact decoder-only Transformer with:

- vocabulary size from the extracted Blender corpus
- context length of 128
- embedding size of 128
- 4 attention heads
- 4 transformer blocks
- AdamW optimizer
- cross-entropy language modeling objective

## Repository structure

```text
BlenderScratchGPT/
├── README.md
├── .gitignore
├── activation_router_experiment_numpy.py  # Matched activation-routing experiment
├── experiment_results/         # Executed report and raw JSON metrics
├── test.py                    # PINN research experiment for a mass-spring-damper system
├── scrape_blender.py          # Extract text from Blender HTML docs
├── clean_text.py              # Cleaning and normalization
├── blender_tokenizer.py       # Character-level tokenization pipeline
├── dataset.py                 # Dataset and dataloader setup
├── model.py                   # Tiny GPT model definition
├── train.py                   # Training loop and checkpoint export
├── generate.py                # Text generation from a trained model
├── chat.py                    # Interactive prompt loop
├── checkpoint.py              # Checkpoint utilities
├── evaluate.py                # Evaluation utilities
├── dataloader.py              # Optional loader helpers
├── blender_api_5_1/           # Offline Blender API docs (HTML)
├── data/                      # Tokenized data and generated corpora
├── checkpoints/               # Saved model checkpoints
├── .venv/                     # Local virtual environment (ignored by Git)
└── blender_api_5_1.txt        # Generated combined text corpus
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install torch beautifulsoup4
```

If a requirements file is added later, use:

```powershell
pip install -r requirements.txt
```

## Data pipeline

### 1) Collect the Blender API docs

Place the local HTML reference files under the project folder, for example:

```text
blender_api_5_1/
```

### 2) Scrape the docs

```powershell
python scrape_blender.py
```

This script walks the HTML folder and extracts readable content into a text corpus.

### 3) Clean and normalize the corpus

```powershell
python clean_text.py
```

### 4) Tokenize the corpus

```powershell
python blender_tokenizer.py
```

This creates a tokenized dataset in `data/tokenized.pt`.

## Training

Launch training with:

```powershell
python train.py
```

The training loop saves checkpoints into the `checkpoints/` directory. A typical checkpoint path is:

```text
checkpoints/blender_gpt.pt
```

## Generation

After training, generate text or Blender-like code with:

```powershell
python generate.py
```

The script loads the saved model and samples from the learned distribution. You can also use:

```powershell
python chat.py
```

for an interactive prompt loop.

## Example use

A typical prompt might look like:

```text
Prompt: create a cube and move it along the x axis
```

The model may output Blender Python such as:

```python
import bpy

obj = bpy.context.active_object
obj.location.x += 2
```

This is meant as a research prototype and is not guaranteed to be correct or production-safe.

## Important limitations

- This project is trained on documentation text, not on verified Blender execution traces.
- Training loss is not the same as functional correctness.
- Generated code may be syntactically valid but semantically wrong.
- The model is small and designed for experimentation, not as a production coding assistant.

## Roadmap

Possible next steps:

1. Improve corpus cleaning and filtering
2. Add validation metrics and held-out evaluation
3. Improve generation quality with better sampling and decoding
4. Add support for code-focused prompts and Blender scripting examples
5. Explore LoRA/QLoRA fine-tuning from a stronger base model

## License

This project is intended for academic and experimental use. Add or review the repository license before public or commercial use.

## Contributing

Contributions are welcome for:

- data-cleaning improvements
- model architecture experiments
- generation quality tuning
- documentation and reproducibility enhancements

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
