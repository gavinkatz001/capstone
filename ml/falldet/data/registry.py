"""Dataset metadata registry. Single source of truth for all supported datasets."""

# Standardized label taxonomies
FALL_TYPES = ["forward", "backward", "lateral", "seated", "syncope", "other_fall"]
ADL_TYPES = ["walk", "sit", "stand", "lie", "transition", "other_adl"]

DATASETS = {
    "nhoyh": {
        "name": "HR_IMU Fall Detection Dataset",
        "format": "mat",
        "sampling_rate_hz": 50,
        "placement": "wrist",
        "has_gyro": True,
        "n_subjects": 21,
        "url": "https://github.com/nhoyh/HR_IMU_falldetection_dataset",
        "download_type": "git_clone",
        "notes": (
            "Contains accel (ax,ay,az) and gyro angular velocity data. "
            "19 activity types including 6 fall types and 13 ADL types."
        ),
        # Mapping from dataset-specific labels to our standard taxonomy
        "fall_label_map": {
            "Fall forward": "forward",
            "Fall forward knees": "forward",
            "Fall backward": "backward",
            "Fall sideward": "lateral",
            "Fall sitting chair": "seated",
            "Fall from bed": "seated",
        },
        "adl_label_map": {
            "Walking": "walk",
            "Jogging": "walk",
            "Going upstairs": "walk",
            "Going downstairs": "walk",
            "Sitting": "sit",
            "Sitting on chair": "sit",
            "Standing": "stand",
            "Standing from sitting": "transition",
            "Lying down from standing": "transition",
            "Lying down": "lie",
            "Picking up object": "other_adl",
            "Jumping": "other_adl",
            "Stretching": "other_adl",
        },
    },
    "microchip": {
        "name": "Microchip ML Fall Detection SAMD21 IMU",
        "format": "csv",
        "sampling_rate_hz": 100,
        "placement": "unknown",
        "has_gyro": True,
        "n_subjects": None,  # not specified in dataset
        "url": "https://github.com/MicrochipTech/ml-Fall-Detection-SAMD21-IMU",
        "download_type": "git_clone",
        "notes": (
            "Pre-segmented samples. Contains fall, idle, and normal activity classes. "
            "CSV format with 6-axis IMU data."
        ),
        "fall_label_map": {
            "fall": "other_fall",
        },
        "adl_label_map": {
            "idle": "sit",
            "normal": "other_adl",
        },
    },
    "edgefall": {
        "name": "IEEE EdgeFall IMU Dataset",
        "format": "csv",
        "sampling_rate_hz": None,  # must be measured from timestamps
        "placement": "neck",
        "has_gyro": True,
        "n_subjects": None,
        "url": "https://ieee-dataport.org/documents/fall-detection-imu-dataset-wearable-applications",
        "download_type": "manual",
        "notes": "Requires IEEE DataPort access. ~1500 trials, neck-mounted sensor.",
        "fall_label_map": {},
        "adl_label_map": {},
    },
    "ur_fall": {
        "name": "UR Fall Detection Dataset",
        "format": "csv",
        "sampling_rate_hz": 60,
        "placement": "body",
        "has_gyro": False,
        "n_subjects": None,
        "url": "https://fenix.ur.edu.pl/mkepski/ds/uf.html",
        "download_type": "manual",
        "notes": "Accelerometer only (no gyro). 30 fall + 40 ADL sequences.",
        "fall_label_map": {
            "fall": "other_fall",
        },
        "adl_label_map": {
            "adl": "other_adl",
        },
    },
}


def get_dataset_info(name: str) -> dict:
    """Get metadata for a dataset by name."""
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASETS[name]


def list_datasets() -> list[str]:
    """List all registered dataset names."""
    return list(DATASETS.keys())
