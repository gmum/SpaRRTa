'''
https://github.com/unrealcv/unrealcv/issues/198
- The camera index is 1, not 0. We import Fusion Camera Actor.
- Objects should be movable
'''

from __future__ import division, absolute_import, print_function
import time; print(time.strftime("The last update of this file: %Y-%m-%d %H:%M:%S", time.gmtime()))
from unrealcv import client
import os
import sys
import json
import re
import csv
from io import BytesIO
from typing import Dict, Tuple, List

import numpy as np
import PIL.Image
from PIL import Image
import shutil


def read_png(res: bytes) -> np.ndarray:
    img = PIL.Image.open(BytesIO(res))
    return np.asarray(img)


class Color(object):
    """Parse UE color string like "(R=255,G=0,B=0,A=255)" into RGBA ints."""
    regexp = re.compile('\(R=(.*),G=(.*),B=(.*),A=(.*)\)')

    def __init__(self, color_str: str):
        self.color_str = color_str
        match = self.regexp.match(color_str)
        if match is None:
            raise ValueError('Unexpected color format: %r' % color_str)
        (self.R, self.G, self.B, self.A) = [int(match.group(i)) for i in range(1, 5)]

    def __iter__(self):
        yield self.R; yield self.G; yield self.B; yield self.A

    def rgb_tuple(self) -> Tuple[int, int, int]:
        return (self.R, self.G, self.B)

    def __repr__(self):
        return self.color_str


def match_color(object_mask: np.ndarray, target_rgb: Tuple[int, int, int], tolerance: int = 3) -> np.ndarray:
    """Return boolean mask for pixels within tolerance of target_rgb."""
    match_region = np.ones(object_mask.shape[0:2], dtype=bool)
    for c in range(3):  # r, g, b
        min_val = target_rgb[c] - tolerance
        max_val = target_rgb[c] + tolerance
        channel_region = (object_mask[:, :, c] >= min_val) & (object_mask[:, :, c] <= max_val)
        match_region &= channel_region
    return match_region


def sanitize_label(label: str) -> str:
    """Make label safe for filenames."""
    # Keep alnum, dash, underscore; replace others with underscore
    return re.sub(r'[^A-Za-z0-9_-]+', '_', label).strip('_') or 'object'


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_camera_intrinsics() -> None:
    # Fixed by requirement: 2048x2048 and FOV 53.13
    client.request('vset /camera/1/size 2048 2048')
    client.request('vset /camera/1/fov 53.13')


def set_camera_transform(location: Dict[str, float], rotation: Dict[str, float]) -> None:
    client.request('vset /camera/1/location %.6f %.6f %.6f' % (location['x'], location['y'], location['z']))
    client.request('vset /camera/1/rotation %.6f %.6f %.6f' % (rotation['pitch'], rotation['yaw'], rotation['roll']))
    res = client.request('vget /camera/1/location')
    print('camera location:', res)
    res = client.request('vget /camera/1/rotation')
    print('camera rotation:', res)


def set_actor_transform(actor_id: str, location: Dict[str, float], rotation: Dict[str, float]) -> None:
    client.request('vset /object/%s/location %.6f %.6f %.6f' % (actor_id, location['x'], location['y'], location['z']))
    client.request('vset /object/%s/rotation %.6f %.6f %.6f' % (actor_id, rotation['pitch'], rotation['yaw'], rotation['roll']))


def build_id_to_color(scene_object_ids) -> Dict[str, Color]:
    id2color: Dict[str, Color] = {}
    for obj_id in scene_object_ids:
        try:
            color = Color(client.request('vget /object/%s/color' % obj_id))
            id2color[obj_id] = color
        except Exception:
            # Skip objects that don't respond to color query
            continue
    return id2color


def save_binary_mask(mask_bool: np.ndarray, out_path: str) -> None:
    mask = (mask_bool.astype(np.uint8)) * 255
    Image.fromarray(mask).save(out_path)


def process_params_json(params_path: str, output_root: str) -> None:
    with open(params_path, 'r') as f:
        params = json.load(f)

    # Create output subdirectory for this JSON file
    params_name = os.path.splitext(os.path.basename(params_path))[0]
    out_dir = os.path.join(output_root, params_name)
    ensure_dir(out_dir)

    # Determine mapping from object id to color
    scene_objects = client.request('vget /objects').split(' ')
    print('Number of objects in the scene:', len(scene_objects))
    scene_objects = [obj for obj in scene_objects if 'StaticMeshActor' in obj]
    print('Number of objects with StaticMeshActor in the name:', len(scene_objects))
    id2color = build_id_to_color(scene_objects)


    # Move camera (using only location/rotation, ignoring path/label)
    camera_cfg = params.get('camera', {})
    cam_loc = camera_cfg.get('location') or {"x": 0, "y": 0, "z": 0}
    cam_rot = camera_cfg.get('rotation') or {"pitch": 0, "yaw": 0, "roll": 0}
    set_camera_transform(cam_loc, cam_rot)

    # Move all actors according to JSON
    actors_cfg: Dict[str, dict] = params.get('actors', {})
    # change position of last two elements of scene_objects
    scene_objects[-2], scene_objects[-1] = scene_objects[-1], scene_objects[-2]
    i = 0
    for key in actors_cfg.keys():
        actors_cfg[scene_objects[i]] = actors_cfg[key]
        del actors_cfg[key]
        i += 1
        if i == len(scene_objects):
            break
    for actor_id, cfg in actors_cfg.items():
        try:
            set_actor_transform(actor_id, cfg.get('location', {}), cfg.get('rotation', {}))
        except Exception:
            # Skip if actor_id not found in level (non-fatal, continue processing)
            print('Warning: Could not move actor %s, skipping...' % actor_id)
            continue

    # Extract number from params_name (e.g., "params_0000" -> "0000")
    number_match = re.search(r'(\d+)$', params_name)
    img_number = number_match.group(1) if number_match else '0000'
    
    # Find and copy the original image from dataset folder
    dataset_dir = os.path.dirname(params_path)
    # Try JPG first (most common), then PNG
    original_img_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        candidate = os.path.join(dataset_dir, 'img_%s%s' % (img_number, ext))
        if os.path.exists(candidate):
            original_img_path = candidate
            break
    
    if original_img_path is None:
        raise FileNotFoundError('Original image not found for params_%s in %s' % (img_number, dataset_dir))
    
    # Copy the exact original image to output (preserving original format)
    img_ext = os.path.splitext(original_img_path)[1]
    img_path = os.path.join(out_dir, 'img_%s%s' % (img_number, img_ext))
    shutil.copy2(original_img_path, img_path)
    
    # Capture object mask after transforms are applied
    object_mask_png = client.request('vget /camera/1/object_mask png')
    object_mask = read_png(object_mask_png)
    
    # Create metadata subfolder
    metadata_dir = os.path.join(out_dir, 'metadata')
    ensure_dir(metadata_dir)

    # Track label counts for duplicate handling
    label_counts: Dict[str, int] = {}
    csv_rows: List[Tuple[str, str, str]] = []

    # For each actor in the JSON, derive its mask from the color-coded mask
    for actor_id, cfg in actors_cfg.items():
        label = sanitize_label(cfg.get('label', 'object'))
        color = id2color.get(actor_id)
        if color is None:
            # Try fallback: some IDs may be stripped; attempt match by suffix
            matches = [(i, c) for i, c in id2color.items() if actor_id.endswith(i)]
            if matches:
                _, color = matches[0]  # Use matched scene object's color
            else:
                # Skip if actor not found in scene
                print('Warning: Actor %s not found in scene, skipping mask generation...' % actor_id)
                continue

        mask_bool = match_color(object_mask, color.rgb_tuple(), tolerance=3)
        
        # Handle duplicate labels with numeric suffix
        if label in label_counts:
            label_counts[label] += 1
            filename_label = '%s_%d' % (label, label_counts[label])
        else:
            label_counts[label] = 1
            filename_label = label
        
        # Save as mask_{label}.png (or mask_{label}_{n}.png for duplicates) in metadata folder
        mask_filename = 'mask_%s.png' % filename_label
        out_path = os.path.join(metadata_dir, mask_filename)
        save_binary_mask(mask_bool, out_path)
        
        # Record for CSV with relative path (using forward slashes for cross-platform compatibility)
        relative_path = 'metadata/' + mask_filename
        csv_rows.append((actor_id, cfg.get('label', 'object'), relative_path))

    # Write CSV log file in metadata folder
    csv_path = os.path.join(metadata_dir, 'masks_log.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['actor_id', 'label', 'output_mask_path'])
        writer.writerows(csv_rows)


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install it with: pip install pyyaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    env_name = cfg["active_environment"]
    triple_id = cfg["active_triple"]
    output_dir = cfg.get("output_dir", "output")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, output_dir, env_name, triple_id)
    output_root = os.path.join(base_dir, "images", env_name, triple_id)
    ensure_dir(output_root)

    client.connect()
    if not client.isconnected():
        print('UnrealCV server is not running. Run the game from http://unrealcv.github.io first.')
        sys.exit(-1)

    # Print status and set camera intrinsics
    try:
        print(client.request('vget /unrealcv/status'))
    except Exception:
        pass
    set_camera_intrinsics()

    # Iterate all JSON files directly in dataset directory (not subdirectories)
    json_files = []
    for f in os.listdir(dataset_dir):
        full_path = os.path.join(dataset_dir, f)
        if os.path.isfile(full_path) and f.lower().endswith('.json'):
            json_files.append(full_path)
    json_files.sort()
    print('Found %d params files' % len(json_files))

    # Process each JSON file (fail-fast: any exception will stop execution)
    for params_path in json_files:
        print('Processing', params_path)
        process_params_json(params_path, output_root)
        print('Completed', params_path)
        # wait for 0.5 second
        time.sleep(0.5)

    client.disconnect()
    print('All files processed successfully!')


if __name__ == '__main__':
    main()
