"""Protocol constants shared by data preparation, adapters, and evaluation."""

REFERENCE_POINTS = 8_192
UPSAMPLING_INPUT_POINTS = {4: 2_048, 16: 512}
COMPLETION_INPUT_POINTS = 2_048
COMPLETION_VIEWS_PER_MODE = 4
VISIBILITY_RESOLUTION = 256
NORMALIZED_F1_TOLERANCE = 0.01
FORESTRY_VOXEL_SIZE_METRES = 0.05
VERTICAL_PROFILE_BINS = 20
FUEL_BIN_SIZE_METRES = 0.25
FUEL_LAYER_ACTIVE_FRACTION = 0.10
FUEL_LAYER_MINIMUM_BINS = 2

HELD_OUT_GENERA = (
    "Prunus",
    "Corylus",
    "Euonymus",
    "Larix",
    "Abies",
    "Populus",
    "Crataegus",
    "Ulmus",
    "Tilia",
    "Quercus",
)

POINT_METRIC_NAMES = ("cd_l2", "cd_l1", "mhd2", "dcd", "f1")

TREE_METRIC_NAMES = (
    "tree_height_abs_error",
    "tree_height_relative_error",
    "crown_width_abs_error",
    "crown_width_relative_error",
    "crown_projection_area_abs_error",
    "crown_projection_area_relative_error",
    "crown_projection_iou",
    "occupied_volume_abs_error",
    "occupied_volume_relative_error",
    "canopy_occupancy_iou",
    "vertical_profile_l1",
    "crown_length_abs_error",
    "crown_length_relative_error",
    "relative_height_25_abs_error",
    "relative_height_50_abs_error",
    "relative_height_75_abs_error",
    "relative_height_95_abs_error",
    "foliage_height_diversity_abs_error",
    "crown_base_height_abs_error",
    "crown_base_height_relative_error",
    "fuel_stratum_profile_l1",
    "stratum_percentage_cover_l1",
    "fuel_layer_count_abs_error",
    "lowest_fuel_layer_base_height_abs_error",
    "dominant_fuel_layer_depth_abs_error",
    "mean_interlayer_distance_abs_error",
    "largest_vertical_gap_abs_error",
)

ALL_METRIC_NAMES = POINT_METRIC_NAMES + TREE_METRIC_NAMES

if len(ALL_METRIC_NAMES) != 32:
    raise RuntimeError("TreeBench must expose exactly 32 metrics")

