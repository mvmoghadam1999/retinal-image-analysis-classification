# Retinal Image Analysis & Classification

A modular computer-vision pipeline for retinal/fundus image datasets, combining dataset preparation, retinal-specific augmentation, statistical image analysis, deep feature extraction, transfer learning, and classical machine-learning classifiers.

## What was cleaned up

The original notebook contained several repeated implementations of the same analysis pipeline for different folders. This repository consolidates those repeated functions into reusable modules.

### Main improvements

- One reusable `ImageDataset` instead of multiple copies.
- One reusable `load_image_data()` for different datasets.
- One implementation of class distribution, sample visualization, statistics, blur, texture, edge analysis, feature extraction, PCA, outlier detection, and t-SNE.
- Dataset-specific analysis is controlled only by the input path.
- Training metrics are calculated through a single reusable function.
- Classifier evaluation is handled through one shared pipeline.
- Configuration is separated from implementation.
- Generated datasets, checkpoints, and caches are excluded from Git.
- Modern TorchVision pretrained-weight API is used for ResNet feature extraction.

## Project structure

```text
retinal-image-analysis-classification/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── image_analysis.py
│   └── training.py
└── outputs/
```

## Pipeline

```text
Raw Dataset
    │
    ├── ZIP extraction
    ├── Folder merging
    ├── Optional augmentation
    └── Train / Validation / Test split
             │
             ▼
      Dataset Analysis
             │
             ├── Class distribution
             ├── Sample visualization
             ├── RGB statistics
             ├── Pixel statistics
             ├── Blur / sharpness
             ├── Texture / GLCM
             ├── Edge density
             ├── ResNet features
             ├── PCA
             ├── Isolation Forest
             └── t-SNE
             │
             ▼
       Deep Learning
             │
             └── Configurable timm backbone
                    │
                    ▼
       Feature Extraction
                    │
                    ├── Logistic Regression
                    ├── Decision Tree
                    ├── Random Forest
                    ├── SVM variants
                    ├── KNN
                    ├── AdaBoost
                    ├── Naive Bayes
                    ├── MLP
                    └── XGBoost (optional)
```

## Installation

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for your CUDA environment.

## Dataset paths

Dataset paths are intentionally kept as local placeholders/configuration values rather than embedded dataset URLs.

Edit the `CONFIG` dictionary in `main.py`:

```python
CONFIG = {
    "dataset_source": "",
    "farabi_zip": "/content/your_dataset.zip",
    "plus_zip": "/content/your_other_dataset.zip",
    ...
}
```

## Dataset preparation

The data pipeline supports:

- ZIP extraction
- Moving/copying images between folders
- Duplicate filename handling
- Retinal-specific augmentation
- Class-aware train/validation/test splitting
- Dataset count summaries

The default split is approximately:

- 70% training
- 15% validation
- 15% test

## Important augmentation note

For a reliable machine-learning experiment, augmentation should normally be applied to the training set only. Augmenting validation or test images can make evaluation less representative of the intended real-world distribution.

The original notebook augmented a real-test path, so this version exposes augmentation as an explicit operation instead of silently applying it.

## Analysis

The same analysis pipeline can be applied to multiple datasets:

```python
from src.image_analysis import analyze_dataset

analyze_dataset("/path/to/dataset_A")
analyze_dataset("/path/to/dataset_B")
```

This replaces the repeated blocks that previously differed only in `folder_path`.

## Deep learning

The training module uses a configurable `timm` backbone. The default is:

```python
backbone = "tf_efficientnetv2_m"
```

Other compatible `timm` models can be selected without rewriting the model class.

The model consists of:

```text
Pretrained backbone
       ↓
Global pooled features
       ↓
Dropout
       ↓
Linear classification head
```

Optional Focal Loss is available for experiments involving class imbalance.

## Evaluation

The pipeline reports:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Confusion matrix
- Training/validation curves

Extracted deep features can also be evaluated with classical classifiers.

## Visual analysis

The repository includes:

- PCA explained variance
- t-SNE feature visualization
- Isolation Forest outlier detection
- Pixel brightness/contrast/entropy
- Skewness and kurtosis
- Laplacian sharpness
- GLCM texture
- Canny edge density
- Class distribution
- Sample image grids

## Notes on pretrained models

The feature-extraction implementation uses TorchVision's current weights API rather than the deprecated `pretrained=True` argument.

The training backbone is created through `timm.create_model(..., pretrained=True)` and therefore remains configurable.

## Reproducibility

Use a fixed `random_state` when splitting datasets and running dimensionality-reduction experiments. For stronger reproducibility, also set Python, NumPy, and PyTorch seeds before training.

## Future improvements

- Add a single CLI with commands such as `prepare`, `analyze`, `train`, and `evaluate`.
- Save all metrics to CSV/JSON.
- Save plots automatically into `outputs/`.
- Add cross-validation for classical classifiers.
- Add class-weighted loss.
- Add experiment configuration through YAML.
- Add model checkpoint metadata.
- Add Grad-CAM / explainability.
- Add automated dataset-quality reports.
- Add patient-level splitting if patient identifiers are available.
- Add unit tests for the data pipeline.

## License

Add your preferred license before publishing the repository.

## Disclaimer

This repository is intended for research and educational use. It is not a medical diagnostic system.
