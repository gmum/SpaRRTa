# =============================================================================
# SpaRRTa Asset Paths
# =============================================================================
# Unreal Engine asset paths for all objects used in scene generation.
# Update these paths to match your actual Unreal Engine project structure.

# --- Vehicles ---
TRUCK_PATH = "/Game/CitySampleVehicles/vehicle04_Truck/Mesh/SM_vehTruck_vehicle04_LOD.SM_vehTruck_vehicle04_LOD"
TAXI_PATH = "/Game/CitySampleVehicles/vehicle06_Car/Mesh/SM_vehCar_vehicle06_LOD"
BIKE_PATH = "/Game/sketchfab/old_bicycle/StaticMeshes/SM_old_bicycle"
VESPA_PATH = "/Game/sketchfab/vespa/vespa"

# --- Nature ---
TREE_PATH = "/Game/sketchfab/tree/StaticMeshes/tree"
ROCK_PATH = "/Game/RealisticDesertPack/Meshes/Rocks/SM_Rock_10"
CACTUS_PATH = "/Game/sketchfab/cactus/cactus"

# --- Animals ---
BEAR_PATH = "/Game/sketchfab/bear/bear"
FOX_PATH = "/Game/sketchfab/fox/fox"
CAMEL_PATH = "/Game/sketchfab/camel/camel"
HUSKY_PATH = "/Game/sketchfab/husky/husky"
DEER_PATH = "/Game/sketchfab/deer/deer"

# --- Objects ---
TENT_PATH = "/Game/sketchfab/tent/tent"
BARREL_PATH = "/Game/sketchfab/barrel/barrel"
SNOWMAN_PATH = "/Game/sketchfab/snowman/snowman"
TRASH_BIN_PATH = "/Game/AutomotiveBridgeScene/Meshes/Environment/Bridge/SM_Trashcan"
CONE_PATH = "/Game/sketchfab/traffic_cone/traffic_cone"
FIRE_HYDRANT_PATH = "/Game/sketchfab/fire_hydrant/fire_hydrant"

# --- Humans (different models for allocentric viewpoints) ---
HUMAN_1_PATH = "/Game/Scanned3DPeoplePack/RP_Character/rp_dennis_posed_004_ue4/rp_dennis_posed_004"
HUMAN_2_PATH = "/Game/Scanned3DPeoplePack/RP_Character/rp_eric_posed_001_ue4/rp_eric_posed_001"
HUMAN_3_PATH = "/Game/Scanned3DPeoplePack/RP_Character/rp_manuel_posed_002_ue4/rp_manuel_posed_002"

ASSET_REGISTRY = {
    "truck":        TRUCK_PATH,
    "taxi":         TAXI_PATH,
    "bike":         BIKE_PATH,
    "vespa":        VESPA_PATH,
    "tree":         TREE_PATH,
    "rock":         ROCK_PATH,
    "cactus":       CACTUS_PATH,
    "bear":         BEAR_PATH,
    "fox":          FOX_PATH,
    "camel":        CAMEL_PATH,
    "husky":        HUSKY_PATH,
    "deer":         DEER_PATH,
    "tent":         TENT_PATH,
    "barrel":       BARREL_PATH,
    "snowman":      SNOWMAN_PATH,
    "trash_bin":    TRASH_BIN_PATH,
    "cone":         CONE_PATH,
    "fire_hydrant": FIRE_HYDRANT_PATH,
    "human_1":      HUMAN_1_PATH,
    "human_2":      HUMAN_2_PATH,
    "human_3":      HUMAN_3_PATH,
}
