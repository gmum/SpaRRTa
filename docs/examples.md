---
title: Examples
description: Copy-pasteable command recipes for common and advanced SpaRRTa workflows
---

# Examples

This page is a **recipe collection** for SpaRRTa.

Use it when you already have the project set up and want short, runnable command patterns. If you still need installation or first-run help, start with [Getting Started](getting-started.md). If you want the reasoning behind the workflows, see [User Guide](user-guide.md).

## Quick Recipes

### First egocentric run

```bash
python train.py \
  backbone=dino_b16 \
  dataset=unreal_position \
  probe=classifier probe._target_=sparrta.models.probes.EfficientProbing \
  dataset.perspective=camera \
  environment=forest
```

### First allocentric run

```bash
python train.py \
  backbone=dino_b16 \
  dataset=unreal_position \
  probe=classifier probe._target_=sparrta.models.probes.EfficientProbing \
  dataset.perspective=human \
  environment=forest
```

### Switch to a simpler probe head

```bash
python train.py \
  backbone=dino_b16 \
  dataset=unreal_position \
  probe=classifier probe._target_=sparrta.models.probes.ClassificationHead \
  dataset.perspective=camera \
  environment=forest
```

### Swap the backbone

```bash
python train.py \
  backbone=mae_b16 \
  dataset=unreal_position \
  probe=classifier probe._target_=sparrta.models.probes.EfficientProbing \
  dataset.perspective=camera \
  environment=forest
```

### Inspect the resolved Hydra config before running

```bash
python train.py --cfg job backbone=dino_b16 dataset=unreal_position
```

## Data Setup Recipes

### Synthetic benchmark data

```bash
export SPARRTA_DATA_ROOT=/path/to/sparrta/unreal
```

### Lego sim-to-real data

```bash
export SPARRTA_LEGO_ROOT=/path/to/sparrta/lego
```

If you want to download the lego split directly:

```bash
huggingface-cli download turhancan97/SpaRRTa-Lego --repo-type dataset --local-dir ./hf_SpaRRTa-Lego
export SPARRTA_LEGO_ROOT=$(pwd)/hf_SpaRRTa-Lego/train
```

### Attention-analysis data

```bash
export SPARRTA_ANALYSIS_ROOT=/path/to/sparrta/attn
```

### Cache and model directories

```bash
export SPARRTA_CACHE_DIR=./cache
export SPARRTA_MODELS_DIR=~/.cache/sparrta/models
```

## Advanced Workflow Recipes

### Leave-one-environment-out / few-shot transfer

Use this workflow when you want to test cross-environment generalization and adaptation.

```bash
python scripts/run_loto_fewshot.py
```

Expected output: result CSVs for holdout-environment transfer and few-shot adaptation, plus summary tables/plots after post-processing.

### Summarize leave-one-environment-out / few-shot results

```bash
python scripts/summarize_loto_fewshot.py
```

### Lego sim-to-real evaluation

Use this workflow when you want to evaluate synthetic-to-real transfer on the lego split.

```bash
python scripts/run_lego_rebuttal.py
```

Expected output: result CSVs comparing performance on the real-world lego evaluation.

### Summarize lego results

```bash
python scripts/summarize_lego_rebuttal.py
```

### Attention analysis for one environment

Use this workflow when you want to inspect where a frozen backbone attends and how attention flows between objects.

```bash
python sparrta/analysis/compute_attention.py environment=winter_town
```

Expected output: attention CSVs and plots, usually under `result/attention/<environment>/`.

### Inspect the attention-analysis config

```bash
python sparrta/analysis/compute_attention.py --cfg job --resolve
```

## Custom Data Checks

If you generated data locally with [`unreal-scene-gen`](unreal-scene-generation.md), sanity-check it before trying evaluation.

### Check the folder contains paired images and metadata

```text
output/<environment>/<triple_id>/
├── img_0000.jpg
├── params_0000.json
├── img_0001.jpg
├── params_0001.json
└── ...
```

### Check the files you care about

- `img_*.jpg`: rendered scene images
- `params_*.json`: camera and actor geometry

### Quick validation checklist

- image files and `params_*.json` files use matching indices
- each environment/triple folder contains multiple valid pairs
- `params_*.json` includes camera and actor entries
- your final layout matches what the evaluation repo expects, or you have a clear adaptation step

### Important compatibility note

Custom Unreal outputs provide the core ingredients needed by the benchmark, but this page does **not** assume they are automatically drop-in compatible with the evaluation repo’s expected dataset layout.

## Failure Examples

### Missing `SPARRTA_DATA_ROOT`

If the main benchmark run fails immediately, verify:

```bash
echo $SPARRTA_DATA_ROOT
```

### Missing external backbone repo

If a repo-backed model fails, verify the corresponding environment variable, for example:

```bash
echo $VGGT_REPO
echo $SPA_REPO
echo $CROCO_REPO
echo $DINOV3_REPO
```

If you want the safest path, prefer backbones that work out of the box:

- `dino_b16`
- `dinov2_b14`
- `dinov2_b14_reg`
- `dinov2_l14_reg`
- `dinov3_timm`
- `mae_b16`
- `clip_b16_laion`

### Unexpected output location

If results appear to be missing, inspect the run config first:

```bash
python train.py --cfg job backbone=dino_b16 dataset=unreal_position
```

Then check:

- `result/` for experiment outputs
- `SPARRTA_CACHE_DIR` for cached features

## References

- [Getting Started](getting-started.md)
- [User Guide](user-guide.md)
- [Evaluation of VFMs](evaluation-vfms.md)
- [Unreal Scene Generation](unreal-scene-generation.md)
- Upstream evaluation repo: https://github.com/turhancan97/SpaRRTa
