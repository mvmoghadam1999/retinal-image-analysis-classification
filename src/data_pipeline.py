from __future__ import annotations

import random
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import albumentations as A
import inspect
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def extract_zip(zip_path: str | Path, output_dir: str | Path) -> None:
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        print(f"Extracting: {zip_path}")
        archive.extractall(output_dir)


def merge_folders(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    move_files: bool = True,
) -> None:
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for src in source_dir.iterdir():
        if not src.is_file():
            continue

        dst = target_dir / src.name
        counter = 1
        while dst.exists():
            dst = target_dir / f"{src.stem}_{counter}{src.suffix}"
            counter += 1

        if move_files:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)


def _make_random_resized_crop(size: tuple[int, int], p: float = 0.5):
    params = inspect.signature(A.RandomResizedCrop).parameters
    if "size" in params:
        return A.RandomResizedCrop(
            size=size, scale=(0.9, 1.0), ratio=(0.9, 1.1), p=p
        )
    return A.RandomResizedCrop(
        height=size[0],
        width=size[1],
        scale=(0.9, 1.0),
        ratio=(0.9, 1.1),
        p=p,
    )


def _apply_vignette(
    image: np.ndarray,
    intensity_range: tuple[float, float] = (0.1, 0.3),
) -> np.ndarray:
    h, w = image.shape[:2]
    kx = cv2.getGaussianKernel(w, w / 2.0)
    ky = cv2.getGaussianKernel(h, h / 2.0)
    mask = ky @ kx.T
    mask /= mask.max()
    intensity = random.uniform(*intensity_range)
    mask = 1 - (1 - mask) * intensity
    mask3 = np.dstack([mask] * 3)
    return (image.astype(np.float32) * mask3).clip(0, 255).astype(np.uint8)


def build_fundus_augmentation(size: tuple[int, int] = (224, 224)) -> A.Compose:
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=1.0
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.7,
        ),
        A.RGBShift(
            r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5
        ),
        A.ChannelShuffle(p=0.15),
        _make_random_resized_crop(size, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 13), p=0.5),
            A.GaussianBlur(blur_limit=(3, 13), p=0.5),
            A.Sharpen(alpha=(0.25, 0.6), lightness=(0.8, 1.4), p=0.5),
        ], p=1.0),
        A.Lambda(
            image=lambda x, **kwargs: _apply_vignette(
                x, intensity_range=(0.1, 0.3)
            ),
            p=0.4,
        ),
    ])


def augment_dataset(
    dataset_dir: str | Path,
    augmentations_per_image: int = 12,
    *,
    skip_existing_augmented: bool = True,
) -> None:
    dataset_dir = Path(dataset_dir)
    transform = build_fundus_augmentation()
    target_dirs = [
        p for p in dataset_dir.iterdir()
        if p.is_dir()
    ] or [dataset_dir]

    for target_dir in target_dirs:
        for image_path in target_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if skip_existing_augmented and "_aug" in image_path.stem:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            for index in range(augmentations_per_image):
                augmented = transform(image=image)["image"]
                output = target_dir / f"{image_path.stem}_aug{index}.jpg"
                cv2.imwrite(
                    str(output),
                    augmented,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_classification_dataset(
    source_dir: str | Path,
    train_dir: str | Path,
    validation_dir: str | Path,
    test_dir: str | Path,
    *,
    train_ratio: float = 0.70,
    validation_ratio_of_holdout: float = 0.50,
    random_state: int = 42,
    move_files: bool = True,
) -> None:
    """
    Produces approximately:
        train = 70%
        validation = 15%
        test = 15%
    when using the default ratios.
    """
    source_dir = Path(source_dir)
    destinations = {
        "train": Path(train_dir),
        "validation": Path(validation_dir),
        "test": Path(test_dir),
    }
    for path in destinations.values():
        path.mkdir(parents=True, exist_ok=True)

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < validation_ratio_of_holdout < 1:
        raise ValueError("validation_ratio_of_holdout must be between 0 and 1.")

    class_dirs = sorted(p for p in source_dir.iterdir() if p.is_dir())
    for class_dir in class_dirs:
        files = _image_files(class_dir)
        if len(files) < 3:
            raise ValueError(
                f"Class '{class_dir.name}' needs at least 3 images for splitting."
            )

        train_files, holdout = train_test_split(
            files,
            test_size=1 - train_ratio,
            random_state=random_state,
            shuffle=True,
        )
        val_files, test_files = train_test_split(
            holdout,
            test_size=1 - validation_ratio_of_holdout,
            random_state=random_state,
            shuffle=True,
        )

        for split_name, split_files in {
            "train": train_files,
            "validation": val_files,
            "test": test_files,
        }.items():
            target = destinations[split_name] / class_dir.name
            target.mkdir(parents=True, exist_ok=True)

            for src in split_files:
                dst = target / src.name
                if dst.exists():
                    dst = target / f"{src.stem}_{random_state}{src.suffix}"
                if move_files:
                    shutil.move(str(src), str(dst))
                else:
                    shutil.copy2(src, dst)


def dataset_summary(*directories: str | Path) -> None:
    for directory in directories:
        directory = Path(directory)
        print(f"\nDirectory: {directory}")
        for class_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
            count = len(_image_files(class_dir))
            print(f"  {class_dir.name}: {count} images")
