import os
import sys
import importlib
import time
import unreal
import random
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import assets
import utils
import camera
import mesh_actor
import serialize

importlib.invalidate_caches()
importlib.reload(camera)
importlib.reload(utils)
importlib.reload(mesh_actor)
importlib.reload(assets)
importlib.reload(serialize)


# =============================================================================
# Config Loading
# =============================================================================

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required. Install it with: pip install pyyaml"
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_config(cfg):
    """Extract the active environment, triple, sampling params, and asset paths."""
    env_name = cfg["active_environment"]
    triple_id = cfg["active_triple"]

    env = cfg["environments"][env_name]
    triple = env["triples"][triple_id]
    sampling = env["sampling"]
    pose = env["pose"]

    asset_section = cfg.get("assets", {})
    def resolve_asset(key):
        if key in asset_section and asset_section[key].get("path"):
            return asset_section[key]["path"]
        return assets.ASSET_REGISTRY[key]

    def resolve_scale(key):
        if key in asset_section:
            s = asset_section[key].get("scale")
            if s is not None:
                return unreal.Vector(s[0], s[1], s[2])
        return None

    actor_defs = [
        {"key": triple["source"], "label": triple["source"].replace("_", " ").title()},
        {"key": triple["target"], "label": triple["target"].replace("_", " ").title()},
    ]
    if triple.get("viewpoint"):
        actor_defs.append({
            "key": triple["viewpoint"],
            "label": triple["viewpoint"].replace("_", " ").title(),
        })

    for d in actor_defs:
        d["path"] = resolve_asset(d["key"])
        d["scale"] = resolve_scale(d["key"])

    return {
        "env_name": env_name,
        "triple_id": triple_id,
        "pose": pose,
        "sampling": sampling,
        "actor_defs": actor_defs,
        "num_images": cfg.get("num_images", 500),
        "screenshot_resolution": tuple(cfg.get("screenshot_resolution", [2048, 2048])),
        "camera_intrinsics": cfg.get("camera_intrinsics", {}),
    }


# =============================================================================
# Object and Camera Sampling
# =============================================================================

def sample_obj_pos(objects, ax, ay, az, obj_proximity, sample_radius, max_distance):
    ax_center = random.gauss(ax, sample_radius)
    ay_center = random.gauss(ay, sample_radius)

    for o in objects:
        try:
            rot = unreal.Rotator(0, 0, random.uniform(0.0, 360.0))

            x = random.gauss(ax_center, obj_proximity)
            y = random.gauss(ay_center, obj_proximity)

            ground_z = utils.detect_ground_at_position(x, y, az)

            object_offset = o.get_ground_offset()
            spawn_z = ground_z - object_offset + 5

            print(f"Object {o.actor.get_actor_label()}: ground_z={ground_z:.2f}, offset={object_offset:.2f}, spawn_z={spawn_z:.2f}")

            pos = unreal.Vector(x, y, spawn_z)
            o.move_to(pos, rot)

        except Exception as e:
            print(f"Error positioning object {o.actor.get_actor_label()}: {e}")
            pos = unreal.Vector(
                random.gauss(ax_center, obj_proximity),
                random.gauss(ay_center, obj_proximity),
                random.gauss(az, 0),
            )
            o.move_to(pos, rot)

    n = len(objects)
    for i in range(n):
        for j in range(i + 1, n):
            if objects[i].overlaps(objects[j]):
                return False, (ax_center, ay_center)
            if objects[i].distance_to(objects[j]) > max_distance:
                return False, (ax_center, ay_center)
    return True, (ax_center, ay_center)


def sample_camera(cam, objects, axy_center, az, camera_range, camera_height_range):
    object_positions = [o.actor.get_actor_location() for o in objects]
    avg_z = sum(pos.z for pos in object_positions) / len(object_positions) if object_positions else az
    ax_center, ay_center = axy_center
    camera_pos = unreal.Vector(
        random.uniform(ax_center - camera_range, ax_center + camera_range),
        random.uniform(ay_center - camera_range, ay_center + camera_range),
        random.uniform(avg_z, avg_z + camera_height_range),
    )
    cam.move_to(camera_pos)
    cam.look_at_many([o.actor for o in objects])

    cam_offset_angles = [cam.angle_to(o.actor) for o in objects]

    print(f"offset angles: {cam_offset_angles}")

    if max(cam_offset_angles) > min(cam.fov()) - 5:
        return False
    if max(cam_offset_angles) < 10:
        return False

    return True


# =============================================================================
# Main Generation Loop
# =============================================================================

def schedule(cam, objects, output_path, env_name, triple_id, sampling, az,
             num_images, screenshot_res, gap=1.5):
    for i in range(num_images):
        for j in range(10):
            good, (ax_center, ay_center) = sample_obj_pos(
                objects, sampling["_ax"], sampling["_ay"], az,
                sampling["obj_proximity"],
                sampling["sample_radius"],
                sampling["max_distance"],
            )
            print(f"Sampling obj pos iter={j}, good={good}")
            if good:
                break

        for j in range(15):
            good = sample_camera(
                cam, objects, (ax_center, ay_center), az,
                sampling["camera_range"],
                sampling["camera_height_range"],
            )
            print(f"Sampling camera iter={j}, good={good}")
            if good:
                break

        cam.take_screenshot(
            out_name=f"{output_path}/img_{i:04d}.jpg",
            res=screenshot_res,
            delay=0.2,
        )

        params = serialize.snapshot_params(
            [o.actor for o in objects],
            cam.actor,
            environment=env_name,
            triple_id=triple_id,
        )
        json.dump(
            params,
            open(f"{output_path}/params_{i:04d}.json", "w"),
            indent=2,
            ensure_ascii=False,
        )

        t0 = time.time()
        while time.time() - t0 < gap:
            yield None


if __name__ == "__main__":
    cfg = load_config()
    resolved = resolve_config(cfg)

    env_name = resolved["env_name"]
    triple_id = resolved["triple_id"]
    ax, ay, az = resolved["pose"]
    sampling = resolved["sampling"]
    sampling["_ax"] = ax
    sampling["_ay"] = ay

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        resolved.get("output_dir", cfg.get("output_dir", "output")),
        env_name,
        triple_id,
    )
    os.makedirs(output_path, exist_ok=True)

    utils.destroy_by_tag(tag="SCRIPT_GENERATED")

    objects = []
    for actor_def in resolved["actor_defs"]:
        kwargs = {"mesh_path": actor_def["path"], "label": actor_def["label"]}
        if actor_def["scale"] is not None:
            kwargs["scale"] = actor_def["scale"]
        objects.append(mesh_actor.MeshActor(**kwargs))

    cam_intrinsics = resolved["camera_intrinsics"]
    cam = camera.RenderCineCamera(
        label="RenderCamera",
        sensor_mm=cam_intrinsics.get("sensor_mm", 50.0),
        focal_length=cam_intrinsics.get("focal_length", 50.0),
        aperture=cam_intrinsics.get("aperture", 10.0),
    )

    pt = utils.PyTick()
    pt.schedule.append(
        schedule(
            cam, objects, output_path,
            env_name, triple_id, sampling, az,
            num_images=resolved["num_images"],
            screenshot_res=resolved["screenshot_resolution"],
        )
    )
