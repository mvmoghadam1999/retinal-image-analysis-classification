"""
Retinal Image Classification & Dataset Analysis
-----------------------------------------------

Edit the paths in CONFIG, then run the individual stages you need.
"""

from pathlib import Path

from src.data_pipeline import (
    augment_dataset,
    dataset_summary,
    extract_zip,
    merge_folders,
    split_classification_dataset,
)
from src.image_analysis import analyze_dataset

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
CONFIG = {
    # Leave dataset URLs/links blank here if you later add public sources.
    "dataset_source": "",

    # Colab/local input archives:
    "farabi_zip": "/content/Farabi_Dataset2_Zip.zip",
    "plus_zip": "/content/Plus_dataset_main.zip",

    "extracted_farabi": "/content/dataset3",
    "extracted_plus": "/content/dataset2",

    # Source and split directories:
    "source_zone": "/content/dataset3/Zone",
    "train_dir": "/content/train_dataset_3",
    "validation_dir": "/content/validation_dataset_3",
    "test_dir": "/content/test_dataset_3",

    # Additional real-world test set used by the original workflow:
    "real_test_dir": "/content/real_test_dataset_3",

    # Analysis datasets:
    "plus_analysis_dir": "/content/dataset3/Plus",
    "zone_analysis_dir": "/content/dataset3/Zone",

    "batch_size": 16,
    "epochs": 100,
    "backbone": "tf_efficientnetv2_m",
}


def prepare_data():
    """Run only the data-preparation operations you actually need."""
    # Uncomment when the archives are available.
    # extract_zip(CONFIG["farabi_zip"], CONFIG["extracted_farabi"])
    # extract_zip(CONFIG["plus_zip"], CONFIG["extracted_plus"])

    # Example merge from the original workflow:
    # merge_folders(
    #     "/content/dataset2/Plus_dataset_main/Normal_comp",
    #     "/content/dataset3/Plus/No Plus",
    # )

    # IMPORTANT:
    # Run augmentation on TRAINING data, not on validation/test data,
    # unless you intentionally want augmented test data for a separate experiment.
    #
    # augment_dataset("/content/train_dataset_3", augmentations_per_image=12)

    # Example split:
    # split_classification_dataset(
    #     CONFIG["source_zone"],
    #     CONFIG["train_dir"],
    #     CONFIG["validation_dir"],
    #     CONFIG["test_dir"],
    #     train_ratio=0.70,
    #     validation_ratio_of_holdout=0.50,
    # )

    dataset_summary(
        CONFIG["train_dir"],
        CONFIG["validation_dir"],
        CONFIG["test_dir"],
    )


def run_analysis():
    """
    The original notebook contained the same complete analysis three times.
    One function now handles both datasets by changing only the path.
    """
    analyze_dataset(CONFIG["plus_analysis_dir"])
    analyze_dataset(CONFIG["zone_analysis_dir"])


if __name__ == "__main__":
    # Choose the stage you want:
    # prepare_data()
    run_analysis()
