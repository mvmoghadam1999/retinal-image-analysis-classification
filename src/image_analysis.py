from __future__ import annotations

import glob
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from scipy.stats import entropy, kurtosis, skew
from skimage.color import rgb2gray
from skimage.feature import canny, graycomatrix, graycoprops
from skimage.filters import laplace
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageDataset(Dataset):
    """Reusable dataset for all analysis/feature-extraction experiments."""

    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = list(image_paths)
        self.labels = labels if labels is not None else [0] * len(self.image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]


def load_image_data(folder_path: str | Path):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder_path}")

    class_dirs = sorted(p for p in folder_path.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(f"No class subfolders found in: {folder_path}")

    classes = [p.name for p in class_dirs]
    class_to_idx = {name: i for i, name in enumerate(classes)}

    image_paths, labels = [], []
    for class_dir in class_dirs:
        paths = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        image_paths.extend(str(p) for p in paths)
        labels.extend([class_to_idx[class_dir.name]] * len(paths))

    print(f"Classes: {classes}")
    print(f"Total images: {len(image_paths)}")
    return image_paths, labels, classes, class_to_idx


def class_distribution(labels, classes, title="Class Distribution"):
    counts = Counter(labels)
    values = [counts.get(i, 0) for i in range(len(classes))]

    print("\nClass distribution:")
    for class_name, count in zip(classes, values):
        print(f"  {class_name}: {count}")

    plt.figure(figsize=(8, 4))
    plt.bar(classes, values)
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Number of images")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def display_samples(image_paths, labels, classes, n=5, title="Sample Images"):
    fig, axes = plt.subplots(
        len(classes),
        n,
        figsize=(3 * n, 3 * len(classes)),
        squeeze=False,
    )

    for class_index, class_name in enumerate(classes):
        paths = [
            path for path, label in zip(image_paths, labels)
            if label == class_index
        ][:n]

        for column in range(n):
            ax = axes[class_index, column]
            ax.axis("off")
            if column < len(paths):
                ax.imshow(Image.open(paths[column]).convert("RGB"))
                if column == 0:
                    ax.set_title(class_name)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def basic_statistics(image_paths):
    sizes, means, stds = [], [], []

    for path in tqdm(image_paths, desc="Basic statistics"):
        image = np.asarray(Image.open(path).convert("RGB"))
        sizes.append(image.shape[:2])
        means.append(image.mean(axis=(0, 1)))
        stds.append(image.std(axis=(0, 1)))

    size_counts = Counter(sizes)
    print("\nImage-size distribution:")
    for size, count in size_counts.most_common():
        print(f"  {size}: {count}")

    print("Average RGB mean:", np.mean(means, axis=0))
    print("Average RGB std :", np.mean(stds, axis=0))


def advanced_pixel_statistics(image_paths):
    metrics = {
        "brightness": [],
        "contrast": [],
        "entropy": [],
        "skew": [],
        "kurtosis": [],
    }

    for path in tqdm(image_paths, desc="Advanced pixel statistics"):
        gray = np.asarray(Image.open(path).convert("L"))

        metrics["brightness"].append(float(gray.mean()))
        metrics["contrast"].append(float(gray.std()))
        metrics["entropy"].append(
            float(entropy(np.histogram(gray, bins=256, range=(0, 255))[0] + 1e-7))
        )
        metrics["skew"].append(float(skew(gray.ravel())))
        metrics["kurtosis"].append(float(kurtosis(gray.ravel())))

    print("\nAdvanced pixel statistics:")
    for name, values in metrics.items():
        print(f"  {name}: mean={np.mean(values):.3f}")

    plt.figure(figsize=(9, 4))
    sns.boxplot(data=list(metrics.values()))
    plt.xticks(range(len(metrics)), metrics.keys(), rotation=20)
    plt.title("Advanced Pixel Statistics")
    plt.tight_layout()
    plt.show()


def blur_analysis(image_paths):
    scores = []
    for path in tqdm(image_paths, desc="Blur / sharpness"):
        gray = np.asarray(Image.open(path).convert("L"))
        scores.append(float(laplace(gray).var()))

    print(f"Mean sharpness: {np.mean(scores):.4f}")
    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=50)
    plt.title("Blur / Sharpness Distribution")
    plt.xlabel("Laplacian variance")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.show()


def texture_analysis(image_paths):
    contrasts, homogeneities, energies = [], [], []

    for path in tqdm(image_paths, desc="Texture analysis"):
        gray = (rgb2gray(np.asarray(Image.open(path).convert("RGB"))) * 255).astype(np.uint8)
        # Down-quantization keeps GLCM computationally reasonable.
        gray = (gray // 16).astype(np.uint8)
        glcm = graycomatrix(
            gray,
            distances=[1],
            angles=[0],
            levels=16,
            symmetric=True,
            normed=True,
        )
        contrasts.append(graycoprops(glcm, "contrast")[0, 0])
        homogeneities.append(graycoprops(glcm, "homogeneity")[0, 0])
        energies.append(graycoprops(glcm, "energy")[0, 0])

    print("\nTexture means:")
    print("  GLCM contrast   :", np.mean(contrasts))
    print("  GLCM homogeneity:", np.mean(homogeneities))
    print("  GLCM energy     :", np.mean(energies))


def edge_analysis(image_paths):
    densities = []
    for path in tqdm(image_paths, desc="Edge analysis"):
        gray = rgb2gray(np.asarray(Image.open(path).convert("RGB")))
        densities.append(float(canny(gray).mean()))

    print(f"Mean edge density: {np.mean(densities):.6f}")
    plt.figure(figsize=(8, 4))
    plt.hist(densities, bins=50)
    plt.title("Edge Density Distribution")
    plt.xlabel("Edge density")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.show()


def extract_resnet_features(
    image_paths,
    labels,
    batch_size=32,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(device)

    transform = weights.transforms()
    dataset = ImageDataset(image_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features = []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc="ResNet feature extraction"):
            output = model(images.to(device))
            features.append(output.flatten(1).cpu())

    return torch.cat(features).numpy()


def feature_space_analysis(features, labels, max_components=20):
    n_components = min(max_components, features.shape[0] - 1, features.shape[1])
    if n_components < 1:
        return

    pca = PCA(n_components=n_components)
    pca.fit(features)

    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title("PCA Explained Variance")
    plt.xlabel("Components")
    plt.ylabel("Cumulative explained variance")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    unique_labels = np.unique(labels)
    if len(unique_labels) >= 2 and len(features) > len(unique_labels):
        from sklearn.metrics import silhouette_score
        score = silhouette_score(features, labels)
        print(f"Silhouette score: {score:.4f}")


def outlier_detection(features, contamination=0.05):
    detector = IsolationForest(
        contamination=contamination,
        random_state=42,
    )
    predictions = detector.fit_predict(features)
    outliers = np.flatnonzero(predictions == -1)
    print(f"Detected outliers: {len(outliers)}")
    return outliers


def tsne_visualization(
    features,
    labels,
    classes,
    sample_size=2000,
    random_state=42,
):
    if len(features) < 3:
        print("Not enough samples for t-SNE.")
        return

    rng = np.random.default_rng(random_state)
    sample_size = min(sample_size, len(features))
    indices = rng.choice(len(features), sample_size, replace=False)

    # Modern sklearn requires perplexity < number of samples.
    perplexity = min(30, max(2, (sample_size - 1) // 3))
    embedding = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    ).fit_transform(features[indices])

    sampled_labels = np.asarray(labels)[indices]

    plt.figure(figsize=(9, 7))
    for class_index, class_name in enumerate(classes):
        mask = sampled_labels == class_index
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=12,
            alpha=0.7,
            label=class_name,
        )

    plt.title("t-SNE Feature Visualization")
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_dataset(
    folder_path: str | Path,
    *,
    sample_size=5,
    tsne_sample_size=2000,
):
    """
    Single reusable entry point for BOTH Plus and Zone datasets.
    The original notebook duplicated this whole analysis pipeline;
    now only folder_path changes.
    """
    image_paths, labels, classes, _ = load_image_data(folder_path)
    if not image_paths:
        raise ValueError(f"No images found in {folder_path}")

    class_distribution(labels, classes)
    display_samples(image_paths, labels, classes, n=sample_size)
    basic_statistics(image_paths)
    advanced_pixel_statistics(image_paths)
    blur_analysis(image_paths)
    texture_analysis(image_paths)
    edge_analysis(image_paths)

    features = extract_resnet_features(image_paths, labels)
    feature_space_analysis(features, labels)
    outliers = outlier_detection(features)
    tsne_visualization(
        features,
        labels,
        classes,
        sample_size=tsne_sample_size,
    )

    return {
        "image_paths": image_paths,
        "labels": labels,
        "classes": classes,
        "features": features,
        "outliers": outliers,
    }
