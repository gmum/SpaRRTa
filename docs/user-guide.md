---
title: User Guide
description: Comprehensive guide to using SpaRRTa for spatial reasoning evaluation
---

# User Guide

This guide provides detailed documentation on using SpaRRTa for evaluating spatial reasoning in Visual Foundation Models.

## Dataset Structure

### Directory Layout

```
data/sparrta/
├── forest/
│   ├── ego/
│   │   ├── images/
│   │   │   ├── 00001.jpg
│   │   │   ├── 00002.jpg
│   │   │   └── ...
│   │   ├── masks/
│   │   │   ├── 00001.png
│   │   │   └── ...
│   │   └── metadata.json
│   └── allo/
│       └── ...
├── desert/
│   └── ...
├── winter_town/
│   └── ...
├── bridge/
│   └── ...
└── city/
    └── ...
```

### Metadata Format

Each environment contains a `metadata.json` file:

```json
{
  "images": [
    {
      "id": "00001",
      "filename": "images/00001.jpg",
      "mask": "masks/00001.png",
      "source_object": {
        "class": "tree",
        "position": [10.5, 20.3, 0.0],
        "rotation": [0.0, 0.0, 45.0]
      },
      "target_object": {
        "class": "bear",
        "position": [15.2, 18.7, 0.0],
        "rotation": [0.0, 0.0, 90.0]
      },
      "viewpoint_object": {
        "class": "human",
        "position": [5.0, 25.0, 0.0],
        "rotation": [0.0, 0.0, 180.0]
      },
      "camera": {
        "position": [0.0, 30.0, 2.0],
        "rotation": [0.0, -15.0, 0.0],
        "fov": 53.0
      },
      "label_ego": "right",
      "label_allo": "left"
    }
  ]
}
```

## Loading Data

### Using the Dataset Class

```python
from sparrta import SpaRRTaDataset

# Load specific environment and task
dataset = SpaRRTaDataset(
    data_path="data/sparrta",
    environment="forest",
    task="ego",  # or "allo"
    split="train",  # "train", "val", "test"
    transform=None,  # Optional torchvision transforms
)

# Iterate over samples
for image, label, metadata in dataset:
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")  # 0=front, 1=back, 2=left, 3=right
    print(f"Source: {metadata['source_object']['class']}")
```

### Custom Data Loading

```python
import torch
from torch.utils.data import DataLoader

# Create data loaders
train_loader = DataLoader(
    SpaRRTaDataset(data_path, env, task, split="train"),
    batch_size=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    SpaRRTaDataset(data_path, env, task, split="val"),
    batch_size=256,
    shuffle=False,
)
```

### Filtering by Object Triple

```python
# Load only specific object combinations
dataset = SpaRRTaDataset(
    data_path="data/sparrta",
    environment="forest",
    task="ego",
    object_triples=[
        ("tree", "bear", "human"),
        ("rock", "fox", "human"),
    ],
)
```

## Model Integration

### Supported Models

```python
from sparrta.models import list_available_models, load_vfm

# List all supported models
print(list_available_models())
# ['dino_vitb16', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_reg_vitb14', ...]

# Load a model
model = load_vfm("dinov2_vitb14")
```

### Adding Custom Models

```python
from sparrta.models import register_model, VFMBase

@register_model("my_custom_model")
class MyCustomModel(VFMBase):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.model = load_my_model(checkpoint_path)
        
    def extract_features(self, images):
        """
        Extract patch features from images.
        
        Args:
            images: Tensor of shape [B, 3, H, W]
            
        Returns:
            features: Tensor of shape [B, N, D]
                - N: number of patches
                - D: feature dimension
        """
        return self.model.forward_features(images)
    
    @property
    def feature_dim(self):
        return 768  # Feature dimension
    
    @property
    def num_patches(self):
        return 196  # For 224x224 with 16x16 patches

# Use the custom model
model = load_vfm("my_custom_model", checkpoint_path="path/to/weights.pth")
```

## Probing Heads

### Linear Probing

```python
from sparrta.probes import LinearProbe

probe = LinearProbe(
    input_dim=768,      # VFM feature dimension
    num_classes=4,      # Front, Back, Left, Right
    dropout=0.4,
)

# Training
features = model.extract_features(images)  # [B, N, D]
pooled = features.mean(dim=1)               # [B, D] - Global average pooling
logits = probe(pooled)                      # [B, 4]
```

### AbMILP Probing

```python
from sparrta.probes import AbMILPProbe

probe = AbMILPProbe(
    input_dim=768,
    num_classes=4,
    hidden_dim=256,
    dropout=0.4,
)

# Training
features = model.extract_features(images)  # [B, N, D]
logits, attention = probe(features)        # [B, 4], [B, N]
```

### Efficient Probing

```python
from sparrta.probes import EfficientProbe

probe = EfficientProbe(
    input_dim=768,
    num_classes=4,
    num_queries=4,
    output_dim=96,  # input_dim / 8
    dropout=0.4,
)

# Training
features = model.extract_features(images)  # [B, N, D]
logits, attentions = probe(features)       # [B, 4], [B, Q, N]
```

## Training Pipeline

### Basic Training Loop

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Setup
model = load_vfm("dinov2_vitb14").eval().cuda()
probe = EfficientProbe(input_dim=768, num_classes=4).cuda()

optimizer = AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(500):
    probe.train()
    for images, labels, _ in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        # Extract frozen features
        with torch.no_grad():
            features = model.extract_features(images)
        
        # Forward through probe
        logits, _ = probe(features)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Validation
    probe.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.cuda(), labels.cuda()
            features = model.extract_features(images)
            logits, _ = probe(features)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    
    print(f"Epoch {epoch}: Val Acc = {100*correct/total:.2f}%")
```

### Using the Evaluator

```python
from sparrta import SpaRRTaEvaluator

evaluator = SpaRRTaEvaluator(
    data_path="data/sparrta",
    probe_type="efficient",
    device="cuda",
)

# Full evaluation
results = evaluator.evaluate(
    model=model,
    environments="all",  # or ["forest", "desert"]
    tasks=["ego", "allo"],
    seeds=[42, 123],
    triples_per_env=3,
)

# Access results
print(results.summary())
print(results.to_dataframe())
results.save("results/dinov2_results.json")
```

## Visualization

### Attention Maps

```python
from sparrta.visualization import visualize_attention

# Get attention from probe
features = model.extract_features(image.unsqueeze(0))
_, attention = probe(features)  # [1, Q, N]

# Visualize
fig = visualize_attention(
    image=image,
    attention=attention[0],  # [Q, N]
    patch_size=16,
    queries_to_show=[0, 1, 2, 3],
)
fig.savefig("attention_map.png")
```

### Results Plotting

```python
from sparrta.visualization import plot_results

# Load results
results = Results.load("results/all_models.json")

# Generate plots
plot_results.accuracy_by_environment(results, save_path="figs/env_acc.pdf")
plot_results.probe_comparison(results, save_path="figs/probe_cmp.pdf")
plot_results.ego_vs_allo(results, save_path="figs/ego_allo.pdf")
plot_results.model_ranking(results, save_path="figs/ranking.pdf")
```

## Configuration Reference

### Full Configuration Options

```yaml
# configs/full_config.yaml

# Data configuration
data:
  path: "data/sparrta"
  environments:
    - forest
    - desert
    - winter_town
    - bridge
    - city
  tasks:
    - ego
    - allo
  object_triples: null  # null = use all available
  image_size: 224
  normalize: true
  augmentation: false

# Model configuration
model:
  name: "dinov2_vitb14"
  checkpoint: null
  freeze: true
  layer: -1  # -1 = last layer, or specify layer index

# Probe configuration
probe:
  type: "efficient"  # linear, abmilp, efficient
  
  # Linear probe settings
  linear:
    dropout: 0.4
    
  # AbMILP settings
  abmilp:
    hidden_dim: 256
    dropout: 0.4
    
  # Efficient probe settings
  efficient:
    num_queries: 4
    output_dim: null  # null = input_dim / 8
    dropout: 0.4

# Training configuration
training:
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.001
  epochs: 500
  warmup_steps: 100
  scheduler: cosine
  gradient_clip: 1.0
  mixed_precision: true

# Evaluation configuration
evaluation:
  seeds: [42, 123]
  triples_per_env: 3
  checkpoint_selection: "best_val"  # best_val, last
  
# Logging configuration
logging:
  wandb: false
  project: "sparrta"
  save_dir: "results/"
  save_attention: true
```

## Best Practices

### Memory Optimization

```python
# Use gradient checkpointing for large models
model = load_vfm("dinov2_vitl14", gradient_checkpointing=True)

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    features = model.extract_features(images)
    logits, _ = probe(features)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Multi-GPU Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])

# Wrap probe in DDP (model stays frozen)
probe = DDP(probe, device_ids=[local_rank])
```

### Reproducibility

```python
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

<div style="text-align: center; margin-top: 2rem;">
  <a href="../examples/" class="md-button md-button--primary">View Examples →</a>
  <a href="../results/" class="md-button">See Results</a>
</div>

