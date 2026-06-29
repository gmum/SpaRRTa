---
title: Getting Started
description: Evaluation-first setup guide for SpaRRTa, with optional Unreal scene generation
---

# Getting Started

SpaRRTa currently works as a **two-repository workflow**:

- **This repository** contains the project website and the [`unreal-scene-gen/`](unreal-scene-generation.md) pipeline for generating synthetic scenes in Unreal Engine 5.
- **[`turhancan97/SpaRRTa`](https://github.com/turhancan97/SpaRRTa)** contains the main evaluation code for probing Visual Foundation Models, running transfer experiments, lego evaluation, and attention analysis.

For most users, the fastest path is:

1. Use the **published Hugging Face datasets**
2. Run the **evaluation repo**
3. Use Unreal generation only if you need **custom scenes**

## Choose Your Path

=== "Run the benchmark now"

    Use the published synthetic dataset and the main evaluation repository.
    
    This is the recommended path if you want to reproduce the paper's probing workflow or start experimenting quickly.

=== "Generate custom scenes"

    Use `unreal-scene-gen/` in this repository to render RGB images, save scene metadata, and optionally derive segmentation masks.
    
    This path is useful if you want custom environments, new scene samples, or additional synthetic data.

## Orientation

SpaRRTa is a 4-way classification benchmark with two task variants:

- **SpaRRTa-ego**: classify the target object's direction from the **camera's** viewpoint
- **SpaRRTa-allo**: classify the target object's direction from a **human figure's** viewpoint

The main evaluation code trains lightweight probe heads on top of frozen backbones such as DINO, DINOv2, DINOv3, MAE, VGGT, CroCo, SPA, and CLIP.

The first training command below runs one benchmark configuration:

- backbone: `dino_b16`
- dataset: synthetic Unreal data
- probe: `EfficientProbing`
- perspective: `camera` (egocentric)
- environment: `forest`

Results are written by the evaluation repo under its default output directory, `result/`.

## Quick Start: Evaluation Repo

### 1. Clone the evaluation code

```bash
git clone https://github.com/turhancan97/SpaRRTa.git
cd SpaRRTa
```

### 2. Create an environment and install dependencies

```bash
conda create -n sparrta python=3.9 --yes
conda activate sparrta

# Install PyTorch for your CUDA version (example from the upstream README)
conda install pytorch=2.2.1 torchvision=0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -e .
```

Optional, only if you want to use `maskfeat_vitb16`:

```bash
pip install -U openmim && mim install mmcv mmcls "mmselfsup>=1.0.0rc0"
```

### 3. Download or point to the datasets

The evaluation repo expects datasets to live outside the codebase and be referenced through environment variables.

- Synthetic Unreal dataset: https://huggingface.co/datasets/turhancan97/SpaRRTa
- Lego real-world split: https://huggingface.co/datasets/turhancan97/SpaRRTa-Lego
- Attention-analysis split: https://huggingface.co/datasets/turhancan97/SpaRRTa-Attention

Set the paths:

```bash
export SPARRTA_DATA_ROOT=/path/to/sparrta/unreal
export SPARRTA_LEGO_ROOT=/path/to/sparrta/lego
export SPARRTA_ANALYSIS_ROOT=/path/to/sparrta/attn
export SPARRTA_CACHE_DIR=./cache
export SPARRTA_MODELS_DIR=~/.cache/sparrta/models
```

If you download the lego split directly from Hugging Face, the upstream repo documents this setup:

```bash
huggingface-cli download turhancan97/SpaRRTa-Lego --repo-type dataset --local-dir ./hf_SpaRRTa-Lego
export SPARRTA_LEGO_ROOT=$(pwd)/hf_SpaRRTa-Lego/train
```

### 4. Run your first experiment

Train an `EfficientProbing` head on DINO features for the egocentric task in the forest environment:

```bash
python train.py \
  backbone=dino_b16 \
  dataset=unreal_position \
  probe=classifier probe._target_=sparrta.models.probes.EfficientProbing \
  dataset.perspective=camera \
  environment=forest
```

To switch to the allocentric task, change the perspective:

```bash
python train.py \
  backbone=dino_b16 \
  dataset=unreal_position \
  probe=classifier probe._target_=sparrta.models.probes.EfficientProbing \
  dataset.perspective=human \
  environment=forest
```

### 5. Inspect the resolved Hydra config

If you want to confirm exactly what will run before launching training:

```bash
python train.py --cfg job backbone=dino_b16 dataset=unreal_position
```

## Expected Evaluation Inputs

The upstream evaluation repo documents the synthetic Unreal data layout like this:

```text
$SPARRTA_DATA_ROOT/
  forest/mid-objects/
    img_0001.jpg
    params_0001.json
    ...
  desert/mid-objects/
  winter_town/mid-objects/
  bridge/mid-objects/
  city/mid-objects/
```

Each `params_*.json` stores the scene geometry used by the benchmark pipeline, including camera and actor positions.

## Quick Start: Unreal Scene Generation

This repository contains the Unreal Engine generation pipeline under [`unreal-scene-gen/`](unreal-scene-generation.md).

### Prerequisites

- **Unreal Engine 5.5**
- **Python Editor Script Plugin** enabled in Unreal
- **Python 3.10+**
- **PyYAML**

Optional, only for mask generation:

- **UnrealCV**
- **NumPy**
- **Pillow**

Install the Python dependency:

```bash
pip install pyyaml
```

### 1. Configure the generation target

Edit `unreal-scene-gen/config.yaml` and choose:

- `active_environment`
- `active_triple`
- `num_images`
- `screenshot_resolution`

Example:

```yaml
active_environment: "desert"
active_triple: "desert_1"
output_dir: "output"
num_images: 500
```

### 2. Run scene generation inside Unreal Editor

Open your Unreal project with the desired environment level loaded, then run:

```text
py "path/to/unreal-scene-gen/main.py"
```

The generator will:

- sample object placements
- adapt object height to terrain using line traces
- sample a camera viewpoint
- render RGB images
- serialize camera and actor metadata to JSON

Outputs are written under:

```text
unreal-scene-gen/output/<environment>/<triple_id>/
```

For example:

```text
output/desert/desert_1/
├── img_0000.jpg
├── params_0000.json
├── img_0001.jpg
├── params_0001.json
└── ...
```

### 3. Optionally generate masks with UnrealCV

Mask generation is a **separate post-process**. It is not required for base RGB scene generation.

After generating scenes:

1. make sure UnrealCV is installed and the game is running
2. run the mask-generation script

```bash
cd unreal-scene-gen
python batch_generate_masks.py
```

This reads the active environment and triple from `config.yaml`, reuses the saved `params_*.json`, and writes binary masks under:

```text
images/<environment>/<triple_id>/
```

## Integration Note: Generation vs Evaluation

The local Unreal pipeline produces the core ingredients needed by the benchmark:

- RGB images
- camera metadata
- actor metadata
- optional segmentation masks

However, this page does **not** claim that `unreal-scene-gen` output is automatically drop-in compatible with the evaluation repo's dataset loader as-is. Custom generated data may require **layout or adaptation** to match the evaluation repo's expected on-disk structure.

If your goal is to evaluate models quickly, use the published Hugging Face datasets first. Use the local generator when you need custom synthetic data and are prepared to align the output layout with the evaluation repo.

## Advanced Paths

- **Lego evaluation**: use the published real-world lego split and the upstream scripts for sim-to-real experiments.
- **Attention analysis**: use `SPARRTA_ANALYSIS_ROOT` with the attention dataset and the `sparrta/analysis/` scripts from the evaluation repo.
- **Custom backbones**: the upstream repo documents how to add a thin wrapper plus Hydra config for a new model.
- **Transfer / few-shot**: see the upstream scripts for leave-one-environment-out and few-shot adaptation workflows.

## Troubleshooting

!!! failure "Training fails before the first batch"
    
    Check that `SPARRTA_DATA_ROOT` points to a valid synthetic dataset layout. Missing or wrong dataset paths are the most common setup issue.

!!! failure "A backbone errors with a missing repo or missing weights"
    
    Some backbones in the evaluation repo require external repositories or weights, such as `VGGT_REPO`, `SPA_REPO`, `CROCO_REPO`, or `DINOV3_REPO`. Start with backbones that work out of the box, such as `dino_b16`, `dinov2_*`, `dinov3_timm`, `mae_b16`, or `clip_b16_laion`.

!!! failure "Unreal cannot import a Python module"
    
    Unreal's Python environment can differ from your system Python. The local `unreal-scene-gen/README.md` links to Epic's guide for installing Python modules inside Unreal Engine.

!!! info "Do I need UnrealCV to generate RGB images?"
    
    No. UnrealCV is only needed for `batch_generate_masks.py`. Base RGB generation from `main.py` does not require UnrealCV.

## References

- Evaluation repo: https://github.com/turhancan97/SpaRRTa
- Synthetic dataset: https://huggingface.co/datasets/turhancan97/SpaRRTa
- Lego split: https://huggingface.co/datasets/turhancan97/SpaRRTa-Lego
- Attention split: https://huggingface.co/datasets/turhancan97/SpaRRTa-Attention
- Local Unreal pipeline: [`unreal-scene-gen/README.md`](https://github.com/gmum/SpaRRTa/tree/main/unreal-scene-gen)
