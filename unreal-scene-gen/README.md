# SpaRRTa — Unreal Engine Data Generation Pipeline

Config-driven pipeline for generating spatial reasoning scenes in Unreal Engine 5. Produces rendered images, camera/object metadata (JSON), and segmentation masks across multiple environments and object triples.

## Prerequisites

- **Unreal Engine 5.5** with the Python Editor Script Plugin enabled
- **Python 3.10+** (UE embedded or system)
- **PyYAML** — `pip install pyyaml`
- **UnrealCV** plugin (only for mask generation with `batch_generate_masks.py`)
- **NumPy** and **Pillow** (only for mask generation)

### Trobleshooting

If you encounter module not found errors and cannot pip install from Unreal Engine Python Console, watch this [video](https://dev.epicgames.com/community/learning/tutorials/lJly/python-install-modules-with-pip-unreal-engine-5-tutorial) called Python Install Modules With PIP - Unreal Engine 5.

## Project Structure

```
unreal-scene-gen/
├── config.yaml               # Master configuration (environments, triples, assets)
├── main.py                    # Scene generation entrypoint (runs inside UE editor)
├── camera.py                  # CineCameraActor wrapper
├── mesh_actor.py              # StaticMeshActor wrapper with collision/bounds
├── assets.py                  # Asset path registry
├── pose.py                    # Legacy environment poses (kept for backward compat)
├── serialize.py               # JSON serialization of camera/object parameters
├── utils.py                   # Ground detection (line trace) and PyTick scheduler
├── batch_generate_masks.py    # Standalone mask generation via UnrealCV
└── README.md
```

## Configuration

All parameters are controlled through `config.yaml`. Key sections:

### Active Selection

```yaml
active_environment: "desert"   # Which environment to generate
active_triple: "desert_1"      # Which object triple within that environment
```

### Output & Rendering

```yaml
output_dir: "output"                     # Base output directory
num_images: 500                          # Images to generate per run
screenshot_resolution: [2048, 2048]      # Width x Height in pixels
```

### Camera Intrinsics

```yaml
camera_intrinsics:
  sensor_mm: 50.0        # Filmback sensor size (square)
  focal_length: 50.0     # Focal length in mm
  aperture: 10.0         # f-stop
```

### Environments

Each environment defines a world-space anchor pose, sampling parameters, and its available triples:

```yaml
environments:
  desert:
    pose: [67170.0, 40950.0, -36400.0]
    sampling:
      obj_proximity: 500         # Gaussian std-dev for object spread
      sample_radius: 300         # Gaussian std-dev for center offset
      max_distance: 1000         # Max allowed distance between objects
      camera_range: 2500         # Camera XY sampling range
      camera_height_range: 1500  # Camera Z range above object avg
    triples:
      desert_1:
        source: "truck"
        target: "rock"
        viewpoint: "human_1"
```

### Assets

Maps string keys to Unreal Engine asset paths with optional scale overrides:

```yaml
assets:
  truck:
    path: "/Game/CitySampleVehicles/vehicle04_Truck/Mesh/SM_vehTruck_vehicle04_LOD.SM_vehTruck_vehicle04_LOD"
    scale: null              # null = default scale; or [x, y, z]
```

## Supported Environments & Triples

| Environment  | Triple    | Source | Target      | Viewpoint |
|-------------|-----------|--------|-------------|-----------|
| Bridge      | bridge_1  | Truck  | Tree        | Human 1   |
| Bridge      | bridge_2  | Bike   | Trash Bin   | Human 2   |
| Bridge      | bridge_3  | Vespa  | Trash Bin   | Human 3   |
| City        | city_1    | Vespa  | Cone        | Human 1   |
| City        | city_2    | Taxi   | Fire Hydrant| Human 2   |
| City        | city_3    | Bike   | Cone        | Human 3   |
| Desert      | desert_1  | Truck  | Rock        | Human 1   |
| Desert      | desert_2  | Camel  | Cactus      | Human 2   |
| Desert      | desert_3  | Camel  | Barrel      | Human 3   |
| Forest      | forest_1  | Tree   | Rock        | Human 1   |
| Forest      | forest_2  | Bear   | Tent        | Human 2   |
| Forest      | forest_3  | Fox    | Rock        | Human 3   |
| Winter Town | winter_1  | Truck  | Tree        | Human 1   |
| Winter Town | winter_2  | Husky  | Snowman     | Human 2   |
| Winter Town | winter_3  | Deer   | Tree        | Human 3   |

## Usage

### Scene Generation (inside Unreal Editor)

1. Edit `config.yaml` to set the desired `active_environment` and `active_triple`.
2. Open your Unreal Engine project with the target environment level loaded.
3. Run `main.py` from the UE Python console or editor scripting panel:

```
py "path/to/unreal-scene-gen/main.py"
```

Output is written to `output/<environment>/<triple_id>/`:

```
output/desert/desert_1/
├── img_0000.jpg
├── params_0000.json
├── img_0001.jpg
├── params_0001.json
└── ...
```

Each `params_XXXX.json` contains:

```json
{
  "environment": "desert",
  "triple_id": "desert_1",
  "camera": { "label": "...", "location": {...}, "rotation": {...}, "intrinsics": {...} },
  "actors": { ... }
}
```

### Mask Generation (standalone Python)

After generating scenes, run mask generation using UnrealCV:

1. Ensure the UnrealCV plugin is installed and the game is running.
2. Run:

```bash
python batch_generate_masks.py
```

This reads `config.yaml` for the active environment/triple, locates the generated params files, and produces binary segmentation masks under `images/<environment>/<triple_id>/`.

## License

See [LICENSE](LICENSE).
