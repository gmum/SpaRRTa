---
title: User Guide
description: Researcher-focused workflows for running and extending SpaRRTa experiments
---

# User Guide

This page is the **day-to-day operations manual** for SpaRRTa after initial setup is complete.

If you have not done your first install or first run yet, start with [Getting Started](getting-started.md). This guide assumes you already understand the basic two-repo layout:

- **This repository**: project website and `unreal-scene-gen/`
- **[`turhancan97/SpaRRTa`](https://github.com/turhancan97/SpaRRTa)**: evaluation, probing, transfer, lego evaluation, and attention analysis

The goal here is to help researchers move from “it runs” to “I know how to use it well.”

## Environment Variables and Data Layout

The evaluation repo relies on a small set of environment variables to find data, cache features, and store model weights.

### Core Variables

```bash
export SPARRTA_DATA_ROOT=/path/to/sparrta/unreal
export SPARRTA_LEGO_ROOT=/path/to/sparrta/lego
export SPARRTA_ANALYSIS_ROOT=/path/to/sparrta/attn
export SPARRTA_CACHE_DIR=./cache
export SPARRTA_MODELS_DIR=~/.cache/sparrta/models
```

### What each variable is for

- `SPARRTA_DATA_ROOT`: synthetic Unreal dataset used for the main benchmark
- `SPARRTA_LEGO_ROOT`: real-world lego dataset used for sim-to-real evaluation
- `SPARRTA_ANALYSIS_ROOT`: attention-analysis dataset with masks
- `SPARRTA_CACHE_DIR`: cache for frozen features computed from backbones
- `SPARRTA_MODELS_DIR`: storage location for downloaded model weights

### Expected synthetic data layout

The upstream evaluation repo documents the Unreal dataset like this:

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

Each sample is an image plus a matching `params_*.json` file. In practice:

- `img_*.jpg` stores the rendered scene
- `params_*.json` stores the camera and actor geometry used by the benchmark

The lego dataset is organized by class labels such as `front`, `back`, `left`, and `right`, while the attention-analysis dataset adds segmentation masks under a `metadata/` folder.

## Common Evaluation Workflows

### Run one backbone on one environment

This is the safest default experimental path:

```bash
python train.py \
  backbone=dino_b16 \
  dataset=unreal_position \
  probe=classifier probe._target_=sparrta.models.probes.EfficientProbing \
  dataset.perspective=camera \
  environment=forest
```

This command means:

- backbone: `dino_b16`
- dataset: synthetic Unreal benchmark
- probe head: `EfficientProbing`
- perspective: camera / egocentric
- environment: forest

### Switch from egocentric to allocentric

To run the same experiment from the human viewpoint:

```bash
python train.py \
  backbone=dino_b16 \
  dataset=unreal_position \
  probe=classifier probe._target_=sparrta.models.probes.EfficientProbing \
  dataset.perspective=human \
  environment=forest
```

Use:

- `dataset.perspective=camera` for **SpaRRTa-ego**
- `dataset.perspective=human` for **SpaRRTa-allo**

### Swap the probe head

The upstream repo exposes three main probing choices:

- `sparrta.models.probes.ClassificationHead`
- `sparrta.models.probes.ABMILPHead`
- `sparrta.models.probes.EfficientProbing`

Example with a simpler baseline probe:

```bash
python train.py \
  backbone=dino_b16 \
  dataset=unreal_position \
  probe=classifier probe._target_=sparrta.models.probes.ClassificationHead \
  dataset.perspective=camera \
  environment=forest
```

### Change the backbone

Once the workflow is stable, you can swap the backbone on the CLI:

```bash
python train.py \
  backbone=mae_b16 \
  dataset=unreal_position \
  probe=classifier probe._target_=sparrta.models.probes.EfficientProbing \
  dataset.perspective=camera \
  environment=forest
```

Other safe starting choices from the upstream README include:

- `dinov2_b14`
- `dinov2_b14_reg`
- `dinov2_l14_reg`
- `dinov3_timm`
- `clip_b16_laion`

### Inspect the resolved Hydra config

Before launching a run, inspect the final resolved config:

```bash
python train.py --cfg job backbone=dino_b16 dataset=unreal_position
```

This is the fastest way to verify:

- selected backbone
- selected probe
- dataset perspective
- environment
- output configuration

### Where results go

The upstream repo writes results under its default output directory, `result/`. Feature caches are written under `SPARRTA_CACHE_DIR`.

## Choosing Backbones and Probes

### Recommended starting backbones

For a reliable first set of experiments, start with backbones that work out of the box:

- `dino_b16`
- `dinov2_b14`
- `dinov2_b14_reg`
- `dinov2_l14_reg`
- `dinov3_timm`
- `mae_b16`
- `clip_b16_laion`

These are easier to use because they do not require separate external code repositories.

### Backbones that need extra setup

Some backbones depend on external repos or weights:

- `vggt_l16` needs `VGGT_REPO`
- `spa_b16`, `spa_l16` need `SPA_REPO`
- `croco_b16`, `crocov2_b16` need `CROCO_REPO`
- `dinov3_b16` needs `DINOV3_REPO` and `DINOV3_WEIGHTS`

If one of these fails immediately, the usual cause is a missing environment variable or missing local checkout.

### Practical probe tradeoffs

- **ClassificationHead**: cheapest and simplest baseline; useful for quick comparisons
- **ABMILPHead**: attention-based pooling; stronger than a plain pooled baseline
- **EfficientProbing**: strongest default choice; best fit when you care about the benchmark’s main result that spatial information lives in patch tokens

Recommended default:

- start with `EfficientProbing` for serious experiments
- use `ClassificationHead` when you want a lightweight baseline

## Advanced Workflows

### Leave-one-environment-out and few-shot transfer

**What it is for:** testing generalization across environments and few-shot adaptation to a held-out domain.

**Relevant scripts:**

- `scripts/run_loto_fewshot.py`
- `scripts/summarize_loto_fewshot.py`

**What you need:**

- `SPARRTA_DATA_ROOT`
- a working evaluation installation

**What to expect:** result CSVs and summary tables/plots for holdout-environment transfer and adaptation performance.

The upstream repo also documents three protocol values used by the training pipeline:

- `default`
- `loto_source_to_target`
- `target_only`

### Lego sim-to-real evaluation

**What it is for:** testing whether probes trained on synthetic SpaRRTa data transfer to a small real-world setup.

**Relevant scripts:**

- `scripts/run_lego_rebuttal.py`
- `scripts/summarize_lego_rebuttal.py`

**What you need:**

- `SPARRTA_LEGO_ROOT`
- a trained or runnable experimental configuration

**What to expect:** comparison results for synthetic-to-real transfer on the lego split.

### Attention analysis

**What it is for:** understanding where frozen backbones attend and how attention flows between objects, background, and CLS/register tokens.

**Relevant scripts:**

- `sparrta/analysis/visualize_patch_masks.py`
- `sparrta/analysis/compute_attention.py`
- `sparrta/analysis/plot_attention.py`
- `sparrta/analysis/aggregate_attention.py`
- `sparrta/analysis/attention_rollout.py`

**What you need:**

- `SPARRTA_ANALYSIS_ROOT`
- the attention-analysis dataset with masks

**What to expect:** per-layer attention CSVs, plots, and rollout visualizations, usually under `result/attention/<environment>/`.

The upstream README shows a minimal command:

```bash
python sparrta/analysis/compute_attention.py environment=winter_town
```

And for config inspection:

```bash
python sparrta/analysis/compute_attention.py --cfg job --resolve
```

## Using Custom Unreal Data

If you want to go beyond the published benchmark data, this repository provides the Unreal generation pipeline in [`unreal-scene-gen`](unreal-scene-generation.md).

That pipeline can generate:

- RGB images
- scene metadata in `params_*.json`
- optional masks via UnrealCV

This is useful for:

- generating more samples
- exploring new scene layouts
- testing custom data curation ideas

However, generated outputs should be treated carefully in relation to the evaluation repo:

- the local generator provides the core ingredients needed by SpaRRTa
- the evaluation repo expects a specific on-disk layout
- custom outputs may require adaptation to match that layout

So the practical recommendation is:

1. start with the published datasets
2. validate your workflow
3. only then move to custom generated data

## Troubleshooting

!!! failure "The run fails immediately with a dataset error"
    
    Check `SPARRTA_DATA_ROOT`, `SPARRTA_LEGO_ROOT`, or `SPARRTA_ANALYSIS_ROOT`. A wrong path or missing dataset is the most common failure mode.

!!! failure "A selected backbone errors before training starts"
    
    Backbones such as VGGT, SPA, CroCo, and local DINOv3 may need external repos or local weights. If you want the most reliable path, use `dino_b16`, `dinov2_*`, `dinov3_timm`, `mae_b16`, or `clip_b16_laion`.

!!! failure "The run completes but expected outputs are missing"
    
    Inspect the resolved Hydra config first and verify the output directory assumptions. Also check whether results are being written under `result/` and features under `SPARRTA_CACHE_DIR`.

!!! info "Do I need UnrealCV for all SpaRRTa workflows?"
    
    No. UnrealCV is only needed for mask generation in the Unreal pipeline. Standard benchmark evaluation on published data does not require UnrealCV.

## References

- [Getting Started](getting-started.md)
- [Evaluation of VFMs](evaluation-vfms.md)
- [Unreal Scene Generation](unreal-scene-generation.md)
- [Spatial evaluation overview](https://github.com/gmum/SpaRRTa/tree/main/spatial-evaluation)
- Upstream evaluation repo: https://github.com/turhancan97/SpaRRTa
