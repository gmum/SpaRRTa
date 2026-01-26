---
title: Getting Started
description: Installation guide and quick start for SpaRRTa benchmark
---

# Getting Started

This guide will help you set up SpaRRTa for evaluating Visual Foundation Models on spatial reasoning tasks.

!!! warning "Coming Soon"
    
    This section is currently under development. Full documentation will be available soon.

<!--
## Prerequisites

Before installing SpaRRTa, ensure you have the following:

- **Python** 3.9 or higher
- **PyTorch** 2.0 or higher with CUDA support
- **Git** for cloning the repository
- **NVIDIA GPU** with at least 11GB VRAM (for evaluation)

!!! info "Optional: Unreal Engine 5"
    
    If you want to generate new synthetic data, you'll also need:
    
    - **Unreal Engine 5.5** (for scene generation)
    - **Windows** operating system (UE5 requirement)
    - Additional 24GB+ VRAM recommended for rendering

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/gmum/SpaRRTa.git
cd SpaRRTa
```

### Step 2: Create Virtual Environment

=== "conda"

    ```bash
    conda create -n sparrta python=3.10
    conda activate sparrta
    ```

=== "venv"

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # or
    .\venv\Scripts\activate  # Windows
    ```

### Step 3: Install Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install SpaRRTa
pip install -e .
```

### Step 4: Download Pre-generated Dataset

```bash
# Download the benchmark dataset
python scripts/download_dataset.py --output data/

# Verify installation
python -c "from sparrta import SpaRRTaDataset; print('Installation successful!')"
```

## Quick Start

### Evaluate a Single Model

```python
from sparrta import SpaRRTaEvaluator, load_vfm

# Load a Visual Foundation Model
model = load_vfm("dinov2_vitb14")

# Initialize evaluator
evaluator = SpaRRTaEvaluator(
    data_path="data/sparrta",
    probe_type="efficient",  # or "linear", "abmilp"
)

# Run evaluation
results = evaluator.evaluate(
    model=model,
    environments=["forest", "desert"],  # or "all"
    tasks=["ego", "allo"],
)

# Print results
print(results.summary())
```

### Run Full Benchmark

```bash
# Evaluate all models with default settings
python scripts/run_benchmark.py --config configs/default.yaml

# Evaluate specific model
python scripts/run_benchmark.py --model dinov2_vitb14 --probe efficient

# Evaluate on specific environment
python scripts/run_benchmark.py --model vggt --env city --task allo
```

## Unreal Engine 5 Setup

If you want to generate custom synthetic data, follow these additional steps.

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10/11 | Windows 11 |
| CPU | 6-core | 8+ cores |
| RAM | 32GB | 64GB |
| GPU | RTX 2080 (11GB) | RTX 4090 (24GB) |
| Storage | 100GB SSD | 500GB NVMe SSD |

### Step 1: Install Unreal Engine 5.5

1. Download and install [Epic Games Launcher](https://www.unrealengine.com/download)
2. Install **Unreal Engine 5.5** from the launcher
3. Enable **Python Editor Script Plugin**:
   - Edit → Plugins → Search "Python"
   - Enable "Python Editor Script Plugin"
   - Restart the editor

### Step 2: Install UnrealCV Plugin

```bash
# Clone UnrealCV
git clone https://github.com/unrealcv/unrealcv.git

# Copy to UE5 plugins folder
cp -r unrealcv/Plugins/UnrealCV /path/to/UE5/Engine/Plugins/
```

### Step 3: Configure SpaRRTa Environments

```bash
# Download SpaRRTa UE5 project
python scripts/download_ue5_project.py --output ue5_project/

# Open in Unreal Engine
# File → Open Project → Select ue5_project/SpaRRTa.uproject
```

### Step 4: Generate Custom Data

```python
from sparrta.generation import SceneGenerator

# Initialize generator
generator = SceneGenerator(
    ue5_project_path="ue5_project/",
    output_path="data/custom/",
)

# Generate scenes
generator.generate(
    environment="forest",
    num_images=1000,
    task="ego",  # or "allo"
    objects=["bear", "tree", "human"],
)
```

## Configuration

### Default Configuration File

Create `configs/my_config.yaml`:

```yaml
# Data settings
data:
  path: "data/sparrta"
  environments: ["forest", "desert", "winter_town", "bridge", "city"]
  tasks: ["ego", "allo"]
  
# Model settings
model:
  name: "dinov2_vitb14"
  checkpoint: null  # Use default pretrained weights
  
# Probe settings
probe:
  type: "efficient"  # linear, abmilp, efficient
  num_queries: 4  # For efficient probing
  dropout: 0.4
  
# Training settings
training:
  batch_size: 256
  learning_rate: 0.001
  epochs: 500
  warmup_steps: 100
  weight_decay: 0.001
  
# Evaluation settings
evaluation:
  seeds: [42, 123]
  triples_per_env: 3
```

### Environment Variables

```bash
# Set data path
export SPARRTA_DATA_PATH=/path/to/data

# Set cache directory for model weights
export SPARRTA_CACHE_DIR=/path/to/cache

# Enable CUDA (default: auto-detect)
export SPARRTA_DEVICE=cuda:0
```

## Project Structure

```
SpaRRTa/
├── sparrta/
│   ├── __init__.py
│   ├── dataset.py          # Dataset classes
│   ├── evaluator.py        # Main evaluation logic
│   ├── models/             # VFM loaders
│   │   ├── dino.py
│   │   ├── clip.py
│   │   ├── mae.py
│   │   └── ...
│   ├── probes/             # Probing heads
│   │   ├── linear.py
│   │   ├── abmilp.py
│   │   └── efficient.py
│   └── generation/         # UE5 data generation
│       ├── scene_generator.py
│       └── utils.py
├── configs/
│   └── default.yaml
├── scripts/
│   ├── download_dataset.py
│   ├── run_benchmark.py
│   └── download_ue5_project.py
├── data/                   # Dataset storage
├── docs/                   # Documentation
├── tests/                  # Unit tests
└── README.md
```

## Troubleshooting

??? question "CUDA out of memory error"
    
    Try reducing batch size:
    ```bash
    python scripts/run_benchmark.py --batch-size 128
    ```
    
    Or use gradient checkpointing:
    ```python
    evaluator = SpaRRTaEvaluator(gradient_checkpointing=True)
    ```

??? question "Model not found error"
    
    Ensure the model is available:
    ```python
    from sparrta.models import list_available_models
    print(list_available_models())
    ```
    
    Or download manually:
    ```bash
    python scripts/download_models.py --model dinov2_vitb14
    ```

??? question "UnrealCV connection failed"
    
    1. Ensure UE5 project is running
    2. Check firewall settings
    3. Verify UnrealCV plugin is enabled
    4. Try restarting the UE5 editor

## Next Steps

<div class="feature-grid">
  <a href="../user-guide/" class="feature-card" style="text-decoration: none;">
    <div class="feature-icon">📖</div>
    <div class="feature-title">User Guide</div>
    <div class="feature-description">
      Detailed documentation on using SpaRRTa for your research.
    </div>
  </a>
  
  <a href="../examples/" class="feature-card" style="text-decoration: none;">
    <div class="feature-icon">💡</div>
    <div class="feature-title">Examples</div>
    <div class="feature-description">
      Code examples and Jupyter notebooks for common use cases.
    </div>
  </a>
  
  <a href="../results/" class="feature-card" style="text-decoration: none;">
    <div class="feature-icon">📊</div>
    <div class="feature-title">Results</div>
    <div class="feature-description">
      Full benchmark results and leaderboards.
    </div>
  </a>
</div>

-->
