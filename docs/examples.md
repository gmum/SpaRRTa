---
title: Examples
description: Code examples and demonstrations for using SpaRRTa
---

# Examples

This page provides practical code examples for using SpaRRTa in your research.

## Quick Examples

### Evaluate a Single Model

```python
from sparrta import SpaRRTaEvaluator, load_vfm

# Load model
model = load_vfm("dinov2_vitb14")

# Create evaluator
evaluator = SpaRRTaEvaluator(
    data_path="data/sparrta",
    probe_type="efficient",
)

# Run evaluation
results = evaluator.evaluate(
    model=model,
    environments=["forest"],
    tasks=["ego"],
)

print(f"Forest Ego Accuracy: {results['forest']['ego']['accuracy']:.2f}%")
```

### Compare Multiple Models

```python
from sparrta import SpaRRTaEvaluator, load_vfm
import pandas as pd

models = [
    "dinov2_vitb14",
    "dinov2_vitl14", 
    "clip_vitb16",
    "mae_vitb16",
]

evaluator = SpaRRTaEvaluator(data_path="data/sparrta")
results = {}

for model_name in models:
    model = load_vfm(model_name)
    results[model_name] = evaluator.evaluate(
        model=model,
        environments="all",
        tasks=["ego", "allo"],
    )

# Create comparison table
df = pd.DataFrame({
    name: {
        "Ego": r.mean_accuracy("ego"),
        "Allo": r.mean_accuracy("allo"),
    }
    for name, r in results.items()
}).T

print(df)
```

## Dataset Examples

### Load and Visualize Samples

```python
import matplotlib.pyplot as plt
from sparrta import SpaRRTaDataset

dataset = SpaRRTaDataset(
    data_path="data/sparrta",
    environment="forest",
    task="ego",
    split="test",
)

# Visualize samples
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for idx, ax in enumerate(axes.flat):
    image, label, metadata = dataset[idx]
    
    # Convert tensor to numpy
    img_np = image.permute(1, 2, 0).numpy()
    
    ax.imshow(img_np)
    ax.set_title(f"{metadata['source_object']['class']} → {metadata['target_object']['class']}\nLabel: {['Front', 'Back', 'Left', 'Right'][label]}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("sample_images.png", dpi=150)
plt.show()
```

### Custom Data Filtering

```python
from sparrta import SpaRRTaDataset

# Filter by specific object combinations
dataset = SpaRRTaDataset(
    data_path="data/sparrta",
    environment="desert",
    task="allo",
    object_triples=[
        ("cactus", "camel", "human"),
        ("rock", "barrel", "human"),
    ],
)

print(f"Filtered dataset size: {len(dataset)}")

# Filter by label
left_samples = [
    (img, label, meta) 
    for img, label, meta in dataset 
    if label == 2  # Left
]

print(f"Samples with 'Left' label: {len(left_samples)}")
```

## Training Examples

### Custom Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sparrta import SpaRRTaDataset, load_vfm
from sparrta.probes import EfficientProbe
from tqdm import tqdm

# Configuration
config = {
    "model": "dinov2_vitb14",
    "environment": "forest",
    "task": "ego",
    "batch_size": 256,
    "lr": 1e-3,
    "epochs": 100,
    "num_queries": 4,
}

# Setup data
train_dataset = SpaRRTaDataset("data/sparrta", config["environment"], config["task"], "train")
val_dataset = SpaRRTaDataset("data/sparrta", config["environment"], config["task"], "val")

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# Setup model and probe
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_vfm(config["model"]).to(device).eval()
probe = EfficientProbe(
    input_dim=model.feature_dim,
    num_classes=4,
    num_queries=config["num_queries"],
).to(device)

# Setup training
optimizer = AdamW(probe.parameters(), lr=config["lr"], weight_decay=1e-3)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
criterion = nn.CrossEntropyLoss()

# Training loop
best_val_acc = 0

for epoch in range(config["epochs"]):
    # Train
    probe.train()
    train_loss = 0
    
    for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            features = model.extract_features(images)
        
        logits, _ = probe(features)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    scheduler.step()
    
    # Validate
    probe.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            features = model.extract_features(images)
            logits, _ = probe(features)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    
    val_acc = 100 * correct / total
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Acc = {val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(probe.state_dict(), "best_probe.pth")

print(f"Best Val Accuracy: {best_val_acc:.2f}%")
```

### Training with Weights & Biases Logging

```python
import wandb
import torch
from sparrta import SpaRRTaEvaluator, load_vfm

# Initialize W&B
wandb.init(
    project="sparrta-evaluation",
    config={
        "model": "dinov2_vitl14",
        "probe": "efficient",
        "environments": ["forest", "desert", "city"],
    }
)

model = load_vfm(wandb.config.model)

evaluator = SpaRRTaEvaluator(
    data_path="data/sparrta",
    probe_type=wandb.config.probe,
    wandb_logging=True,  # Enable W&B logging
)

results = evaluator.evaluate(
    model=model,
    environments=wandb.config.environments,
    tasks=["ego", "allo"],
)

# Log final results
wandb.log({
    "mean_ego_accuracy": results.mean_accuracy("ego"),
    "mean_allo_accuracy": results.mean_accuracy("allo"),
})

wandb.finish()
```

## Visualization Examples

### Attention Map Visualization

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from sparrta import load_vfm, SpaRRTaDataset
from sparrta.probes import EfficientProbe

# Load model and probe
model = load_vfm("dinov2_vitb14").cuda().eval()
probe = EfficientProbe(768, 4, 4).cuda()
probe.load_state_dict(torch.load("best_probe.pth"))
probe.eval()

# Load sample image
dataset = SpaRRTaDataset("data/sparrta", "forest", "ego", "test")
image, label, metadata = dataset[0]

# Get attention maps
with torch.no_grad():
    features = model.extract_features(image.unsqueeze(0).cuda())
    logits, attention = probe(features)  # attention: [1, 4, 196]

# Visualize
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# Original image
axes[0].imshow(image.permute(1, 2, 0).numpy())
axes[0].set_title("Original Image")
axes[0].axis("off")

# Attention maps for each query
attention = attention[0].cpu().numpy()  # [4, 196]

for i in range(4):
    attn_map = attention[i].reshape(14, 14)  # Reshape to spatial grid
    attn_map = np.clip(attn_map, 0, 1)
    
    # Resize to image size
    import cv2
    attn_resized = cv2.resize(attn_map, (224, 224))
    
    axes[i+1].imshow(image.permute(1, 2, 0).numpy())
    axes[i+1].imshow(attn_resized, alpha=0.6, cmap="jet")
    axes[i+1].set_title(f"Query {i+1}")
    axes[i+1].axis("off")

plt.tight_layout()
plt.savefig("attention_visualization.png", dpi=150)
plt.show()
```

### Results Comparison Plot

```python
import matplotlib.pyplot as plt
import numpy as np

# Data from evaluation
models = ["DINO", "DINO-v2", "DINOv3", "VGGT", "MAE", "CLIP"]
ego_acc = [89.28, 91.91, 93.93, 96.18, 93.10, 56.33]
allo_acc = [64.38, 71.20, 72.05, 76.65, 70.66, 54.36]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width/2, ego_acc, width, label='Egocentric', color='#7c4dff')
bars2 = ax.bar(x + width/2, allo_acc, width, label='Allocentric', color='#536dfe')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('SpaRRTa Performance by Model (Efficient Probing)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend()
ax.set_ylim(0, 100)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
```

## Video Demo Integration

### Embedding YouTube Video

To showcase video demos on your documentation pages:

```html
<div class="video-container">
  <iframe 
    src="https://www.youtube.com/embed/VIDEO_ID" 
    title="SpaRRTa Demo"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>
```

## Advanced Examples

### Custom Model Evaluation

```python
import torch
import torch.nn as nn
from sparrta.models import register_model, VFMBase

@register_model("my_vit")
class MyViT(VFMBase):
    """Custom Vision Transformer wrapper."""
    
    def __init__(self, checkpoint_path=None):
        super().__init__()
        
        # Load your custom model
        from timm import create_model
        self.model = create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,  # Remove classification head
        )
        
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path)
            self.model.load_state_dict(state_dict)
    
    def extract_features(self, images):
        """Extract patch features."""
        # Get patch embeddings from ViT
        x = self.model.patch_embed(images)
        x = self.model._pos_embed(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        
        # Return patch tokens (excluding CLS)
        return x[:, 1:, :]  # [B, 196, 768]
    
    @property
    def feature_dim(self):
        return 768
    
    @property
    def num_patches(self):
        return 196

# Use custom model
from sparrta import SpaRRTaEvaluator, load_vfm

model = load_vfm("my_vit", checkpoint_path="my_weights.pth")
evaluator = SpaRRTaEvaluator(data_path="data/sparrta")
results = evaluator.evaluate(model)
```

### Multi-GPU Evaluation

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sparrta import SpaRRTaEvaluator, load_vfm
import os

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    local_rank = setup_distributed()
    
    model = load_vfm("dinov2_vitl14").cuda()
    
    evaluator = SpaRRTaEvaluator(
        data_path="data/sparrta",
        distributed=True,
        local_rank=local_rank,
    )
    
    results = evaluator.evaluate(
        model=model,
        environments="all",
    )
    
    if local_rank == 0:
        print(results.summary())
        results.save("distributed_results.json")

if __name__ == "__main__":
    main()
```

Run with:
```bash
torchrun --nproc_per_node=4 distributed_eval.py
```

## Jupyter Notebook

For interactive exploration, see our Jupyter notebook:

```python
# In Jupyter
%load_ext autoreload
%autoreload 2

from sparrta import SpaRRTaDataset, SpaRRTaEvaluator, load_vfm
import matplotlib.pyplot as plt

# Interactive visualization
dataset = SpaRRTaDataset("data/sparrta", "forest", "ego", "test")

# Use ipywidgets for interactive exploration
from ipywidgets import interact, IntSlider

@interact(idx=IntSlider(min=0, max=len(dataset)-1, step=1))
def show_sample(idx):
    image, label, meta = dataset[idx]
    plt.figure(figsize=(8, 8))
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title(f"Source: {meta['source_object']['class']}\n"
              f"Target: {meta['target_object']['class']}\n"
              f"Label: {['Front', 'Back', 'Left', 'Right'][label]}")
    plt.axis("off")
    plt.show()
```

---

<div style="text-align: center; margin-top: 2rem;">
  <a href="https://github.com/gmum/SpaRRTa" class="md-button md-button--primary" target="_blank">View on GitHub</a>
  <a href="../user-guide/" class="md-button">User Guide</a>
</div>

