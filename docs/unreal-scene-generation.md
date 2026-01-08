---
title: Unreal Scene Generation
description: Learn how SpaRRTa generates photorealistic synthetic scenes using Unreal Engine 5
---

# Unreal Scene Generation

SpaRRTa leverages **Unreal Engine 5** to generate photorealistic synthetic images with precise control over object placement, camera positions, and environmental conditions. This enables the creation of a rigorous benchmark with mathematically precise ground-truth labels.

## Why Synthetic Data?

<div class="feature-grid">
  <div class="feature-card">
    <div class="feature-icon">🎯</div>
    <div class="feature-title">Precise Control</div>
    <div class="feature-description">
      Full control over object positions, camera angles, and scene composition enables mathematically rigorous ground-truth labels.
    </div>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">📈</div>
    <div class="feature-title">Scalability</div>
    <div class="feature-description">
      Generate arbitrary amounts of diverse data without expensive manual annotation or data collection.
    </div>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">🎨</div>
    <div class="feature-title">Photorealism</div>
    <div class="feature-description">
      Unreal Engine 5's Lumen and Nanite technologies provide state-of-the-art visual fidelity.
    </div>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">🔄</div>
    <div class="feature-title">Reproducibility</div>
    <div class="feature-description">
      Fully deterministic generation enables exact reproduction of experimental conditions.
    </div>
  </div>
</div>

## Evaluation Environments

SpaRRTa includes **five diverse high-fidelity environments** to ensure robust evaluation across different visual domains:

<figure markdown>
  ![Environments](imgs/main/environments_table.png){ width="100%" }
  <figcaption>The five SpaRRTa evaluation environments: Forest, Desert, Winter Town, Bridge, and City.</figcaption>
</figure>

### Environment Details

=== "🌲 Forest"

    **Electric Dreams Environment**
    
    A sparse forest landscape with complex foliage, uneven terrain, and natural rock formations. This environment tests spatial reasoning in organic, unstructured settings.
    
    - **Source**: [Electric Dreams (Epic Games)](https://www.unrealengine.com/marketplace/en-US/product/electric-dreams-env)
    - **Characteristics**: Complex foliage, uneven terrain, natural lighting
    - **Objects**: Bear, Fox, Tent, Rocks, Trees

=== "🏜️ Desert"

    **Arid Landscape**
    
    A vast, arid landscape characterized by open terrain, sand dunes, and high-contrast lighting. This environment is sparse and texture-homogeneous.
    
    - **Characteristics**: Open terrain, high contrast lighting, minimal occlusion
    - **Objects**: Camel, Barrel, Cactus, Rocks

=== "🏔️ Winter Town"

    **Eastern European Village**
    
    A snow-covered setting reflecting a typical small Eastern European town with cold lighting, snow textures, and village buildings.
    
    - **Characteristics**: Cold lighting, snow textures, village architecture
    - **Objects**: Husky, Deer, Snowman

=== "🌉 Bridge"

    **Valley Infrastructure**
    
    A valley scene centered around a large bridge infrastructure with mixed natural and man-made elements.
    
    - **Characteristics**: Infrastructure elements, valley terrain, mixed complexity
    - **Objects**: Bicycle, Trash Can, Vehicle

=== "🏙️ City"

    **Modern Metropolis**
    
    A large-scale, modern American metropolis featuring high-rise architecture, paved roads, and complex urban geometry.
    
    - **Source**: [City Sample (Epic Games)](https://www.unrealengine.com/marketplace/en-US/product/city-sample)
    - **Characteristics**: Dense urban geometry, complex occlusion, varied lighting
    - **Objects**: Motorcycle, Traffic Cone, Fire Hydrant

## Asset Library

<figure markdown>
  ![Assets](imgs/main/assets.png){ width="100%" }
  <figcaption>SpaRRTa's curated asset library spanning Animals, Vehicles, Nature, and Human categories.</figcaption>
</figure>

### Asset Selection Criteria

Our asset selection follows specific criteria to ensure valid spatial reasoning evaluation:

1. **ImageNet Alignment**: Objects align with common ImageNet super-categories to ensure VFMs can recognize them
2. **Isotropic Sources**: Source objects (rocks, trees, cones) are rotationally symmetric to minimize orientation ambiguity
3. **Environmental Coherence**: Objects naturally fit their respective environments (e.g., camels in desert)
4. **Visual Distinctiveness**: Objects are clearly distinguishable from backgrounds and each other

| Category | Assets |
|----------|--------|
| **Animals** | Bear, Fox, Camel, Husky, Deer |
| **Vehicles** | Car, Taxi, Motorcycle, Bicycle |
| **Nature** | Trees, Rocks, Cactus |
| **Objects** | Tent, Barrel, Trash Can, Traffic Cone, Fire Hydrant, Snowman |
| **Humans** | Human agent (viewpoint for allocentric tasks) |

## Data Generation Pipeline

<figure markdown>
  ![Pipeline](imgs/main/pipeline.png){ width="100%" }
  <figcaption>The complete SpaRRTa data generation and evaluation pipeline.</figcaption>
</figure>

### Pipeline Steps

```mermaid
flowchart LR
    A[Set Stage] --> B[Set Camera]
    B --> C[Render View]
    C --> D[Get Ground Truth]
    D --> E[Run Model]
    E --> F[Calculate Results]
    
    style A fill:#7c4dff,color:#fff
    style B fill:#7c4dff,color:#fff
    style C fill:#536dfe,color:#fff
    style D fill:#536dfe,color:#fff
    style E fill:#3f1dcb,color:#fff
    style F fill:#3f1dcb,color:#fff
```

#### 1. Set Stage

The evaluator establishes the scene configuration:

- Select environment (Forest, Desert, Winter Town, Bridge, City)
- Choose source, target, and viewpoint objects from the asset library
- Randomly sample object positions from a Gaussian distribution
- Apply physics-aware terrain adaptation via raycasting

#### 2. Set Camera

Configure the viewpoint for image capture:

- Sample camera position within a defined area surrounding scene center
- Orient camera toward placed objects
- Validate visibility constraints (objects within field of view)
- Ensure proper scene composition (no extreme clustering or distance)

#### 3. Render View

Generate high-fidelity imagery using Unreal Engine 5:

- Ray-traced RGB image with dynamic global illumination
- Ground-truth segmentation masks for validation
- Resolution: 224×224 (standard VFM input size)

#### 4. Get Ground Truth

Extract spatial relation labels:

- Calculate angular relationship between source and target objects
- Apply viewpoint transformation (camera for ego, human for allo)
- Filter ambiguous configurations (objects near decision boundaries)
- Assign discrete label: **Front**, **Back**, **Left**, or **Right**

## Geometric Ambiguity Control

A key challenge in spatial classification is defining precise boundaries between classes. SpaRRTa implements strict **rejection sampling** to eliminate label noise:

<figure markdown>
  ![Scene Examples](imgs/scene_examples/2d_scene_winter.png){ width="60%" }
  <figcaption>Visualization of valid placement zones (green) and ambiguity exclusion zones (red/gray).</figcaption>
</figure>

### Exclusion Zones

Ambiguity zones are defined as conical regions centered along the diagonals:

- **45°, 135°, 225°, 315°** relative to the viewpoint's forward vector
- Any sample where the target falls within these zones is **automatically rejected**
- This guarantees unambiguous ground-truth labels

!!! info "Rejection Sampling"
    
    The pipeline automatically discards configurations where the target object lies within ±22.5° of a diagonal boundary, ensuring all retained samples have mathematically precise labels.

## Technical Implementation

### Rendering Stack

| Component | Details |
|-----------|---------|
| **Engine** | Unreal Engine 5.5 |
| **Lighting** | Lumen (dynamic global illumination) |
| **Geometry** | Nanite (virtualized geometry) |
| **API** | Python Editor API + UnrealCV |
| **Hardware** | 2× NVIDIA RTX 2080 Ti (11GB VRAM) |

### Camera Configuration

```python
# Standardized camera settings
SENSOR_WIDTH = 50  # mm
FOCAL_LENGTH = 50  # mm
RESOLUTION = (224, 224)  # pixels
FOV = 2 * arctan(SENSOR_WIDTH / (2 * FOCAL_LENGTH))  # ~53°
```

### Object Placement Algorithm

```python
def place_objects(environment, objects):
    """
    Place objects with physics-aware terrain adaptation.
    """
    center = sample_center_point(environment)
    
    for obj in objects:
        # Sample position around center
        position = center + sample_gaussian(mean=0, std=MAX_DISTANCE)
        
        # Raycast to find ground level
        ground_z = raycast_terrain(position.x, position.y)
        position.z = ground_z + obj.bounding_box.height / 2
        
        # Validate no collisions
        if check_aabb_overlap(obj, placed_objects):
            continue  # Reject and resample
            
        spawn_object(obj, position)
```

## Dataset Statistics

| Environment | Ego Images | Allo Images | Total |
|-------------|------------|-------------|-------|
| Forest | 5,000 | 10,000 | 15,000 |
| Desert | 5,000 | 10,000 | 15,000 |
| Winter Town | 5,000 | 10,000 | 15,000 |
| Bridge | 5,000 | 10,000 | 15,000 |
| City | 5,000 | 10,000 | 15,000 |
| **Total** | **25,000** | **50,000** | **75,000** |

!!! note "Dataset Size Rationale"
    
    - **Egocentric**: 5,000 images sufficient for generalization
    - **Allocentric**: 10,000 images needed due to increased task complexity (perspective transformation learning)

### Environment-Asset Relations

Each environment contains **3 unique object triples** used for evaluation. The table below shows the complete mapping of environments to their source objects, target objects, and viewpoint configurations:

<table class="env-table">
  <thead>
    <tr>
      <th>Triple ID</th>
      <th>Source Object</th>
      <th>Target Object</th>
      <th>Viewpoint</th>
    </tr>
  </thead>
  <tbody>
    <tr class="env-bridge">
      <td><strong>Bridge-1</strong></td>
      <td>Truck</td>
      <td>Tree</td>
      <td>Camera / Human 1</td>
    </tr>
    <tr class="env-bridge">
      <td><strong>Bridge-2</strong></td>
      <td>Bike</td>
      <td>Trash Bin</td>
      <td>Camera / Human 2</td>
    </tr>
    <tr class="env-bridge">
      <td><strong>Bridge-3</strong></td>
      <td>Vespa</td>
      <td>Trash Bin</td>
      <td>Camera / Human 3</td>
    </tr>
    <tr class="env-city">
      <td><strong>City-1</strong></td>
      <td>Vespa</td>
      <td>Cone</td>
      <td>Camera / Human 1</td>
    </tr>
    <tr class="env-city">
      <td><strong>City-2</strong></td>
      <td>Taxi</td>
      <td>Fire Hydrant</td>
      <td>Camera / Human 2</td>
    </tr>
    <tr class="env-city">
      <td><strong>City-3</strong></td>
      <td>Bike</td>
      <td>Cone</td>
      <td>Camera / Human 3</td>
    </tr>
    <tr class="env-desert">
      <td><strong>Desert-1</strong></td>
      <td>Truck</td>
      <td>Rock</td>
      <td>Camera / Human 1</td>
    </tr>
    <tr class="env-desert">
      <td><strong>Desert-2</strong></td>
      <td>Camel</td>
      <td>Cactus</td>
      <td>Camera / Human 2</td>
    </tr>
    <tr class="env-desert">
      <td><strong>Desert-3</strong></td>
      <td>Camel</td>
      <td>Barrel</td>
      <td>Camera / Human 3</td>
    </tr>
    <tr class="env-forest">
      <td><strong>Forest-1</strong></td>
      <td>Tree</td>
      <td>Rock</td>
      <td>Camera / Human 1</td>
    </tr>
    <tr class="env-forest">
      <td><strong>Forest-2</strong></td>
      <td>Bear</td>
      <td>Tent</td>
      <td>Camera / Human 2</td>
    </tr>
    <tr class="env-forest">
      <td><strong>Forest-3</strong></td>
      <td>Fox</td>
      <td>Rock</td>
      <td>Camera / Human 3</td>
    </tr>
    <tr class="env-winter">
      <td><strong>Winter-1</strong></td>
      <td>Truck</td>
      <td>Tree</td>
      <td>Camera / Human 1</td>
    </tr>
    <tr class="env-winter">
      <td><strong>Winter-2</strong></td>
      <td>Husky</td>
      <td>Snowman</td>
      <td>Camera / Human 2</td>
    </tr>
    <tr class="env-winter">
      <td><strong>Winter-3</strong></td>
      <td>Deer</td>
      <td>Tree</td>
      <td>Camera / Human 3</td>
    </tr>
  </tbody>
</table>

!!! info "Viewpoint Configuration"
    
    - **Camera**: Used for egocentric (SpaRRTa-ego) task evaluation
    - **Human 1 / 2 / 3**: Different human models used for allocentric (SpaRRTa-allo) task evaluation, each with unique poses and positions

---

<div style="text-align: center; margin-top: 2rem;">
  <a href="../evaluation-vfms/" class="md-button md-button--primary">Next: Evaluation of VFMs →</a>
</div>

