from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt).pow(self.gamma) * ce).mean()


class FeatureExtractor(nn.Module):
    def __init__(self, backbone="tf_efficientnetv2_m"):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

    def forward(self, x):
        return self.backbone(x)


class ClassifierModel(nn.Module):
    def __init__(self, num_classes, backbone="tf_efficientnetv2_m", dropout=0.5):
        super().__init__()
        self.feature_model = FeatureExtractor(backbone)
        in_features = self.feature_model.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.feature_model(x)
        return self.classifier(self.dropout(features))


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    predictions, labels = [], []

    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += loss.item() * images.size(0)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(targets.cpu().numpy())

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(labels)

    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    predictions, labels = [], []

    for images, targets in tqdm(loader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(labels)

    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def plot_history(history):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    metric_pairs = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
    ]

    for metric, title in metric_pairs:
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, history[f"train_{metric}"], label="Train")
        plt.plot(epochs, history[f"val_{metric}"], label="Validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        plt.tight_layout()
        plt.show()


def extract_features(model, loader, device):
    model.eval()
    features, labels = [], []

    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="Extracting features"):
            output = model(images.to(device))
            features.append(output.cpu().numpy())
            labels.append(targets.numpy())

    return np.vstack(features), np.concatenate(labels)


def build_sklearn_classifiers():
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "SVM-Linear": SVC(kernel="linear", C=1),
        "SVM-Poly": SVC(kernel="poly", C=1, degree=3),
        "SVM-RBF": SVC(kernel="rbf", C=1),
        "SVM-Sigmoid": SVC(kernel="sigmoid", C=1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=300,
            random_state=42,
        ),
    }

    if HAS_XGB:
        classifiers["XGBoost"] = XGBClassifier(
            eval_metric="mlogloss",
            random_state=42,
        )

    return classifiers


def evaluate_classifiers(X_train, y_train, X_eval, y_eval, class_names):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    results = {}
    for name, classifier in build_sklearn_classifiers().items():
        try:
            classifier.fit(X_train_scaled, y_train)
            predictions = classifier.predict(X_eval_scaled)
            metrics = classification_metrics(y_eval, predictions)
            results[name] = metrics

            print(
                f"{name}: "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}"
            )

            cm = confusion_matrix(y_eval, predictions)
            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=class_names,
            ).plot(xticks_rotation=45)
            plt.title(f"{name} - Confusion Matrix")
            plt.tight_layout()
            plt.show()

        except Exception as exc:
            print(f"Skipping {name}: {exc}")

    return results, scaler


def run_pipeline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    class_names: list[str],
    *,
    test_loader: DataLoader | None = None,
    epochs: int = 100,
    backbone: str = "tf_efficientnetv2_m",
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-4,
    patience: int = 10,
    use_focal_loss: bool = False,
    checkpoint_path: str = "best_model.pth",
    device: str | None = None,
):
    device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = ClassifierModel(
        num_classes=num_classes,
        backbone=backbone,
    ).to(device)

    criterion = FocalLoss() if use_focal_loss else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=3,
    )

    history = {
        f"{split}_{metric}": []
        for split in ("train", "val")
        for metric in ("loss", "accuracy", "precision", "recall", "f1")
    }

    best_f1 = -np.inf
    best_loss = np.inf
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = evaluate_epoch(
            model, val_loader, criterion, device
        )

        for metric in ("loss", "accuracy", "precision", "recall", "f1"):
            history[f"train_{metric}"].append(train_metrics[metric])
            history[f"val_{metric}"].append(val_metrics[metric])

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss={train_metrics['loss']:.5f}, "
            f"acc={train_metrics['accuracy']:.4f}, "
            f"F1={train_metrics['f1']:.4f} || "
            f"Val loss={val_metrics['loss']:.5f}, "
            f"acc={val_metrics['accuracy']:.4f}, "
            f"F1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best checkpoint -> {checkpoint_path}")

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            no_improvement = 0
        else:
            no_improvement += 1

        scheduler.step(val_metrics["loss"])

        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    plot_history(history)

    X_train, y_train = extract_features(model.feature_model, train_loader, device)
    X_val, y_val = extract_features(model.feature_model, val_loader, device)

    sklearn_results, scaler = evaluate_classifiers(
        X_train, y_train, X_val, y_val, class_names
    )

    test_results = None
    if test_loader is not None:
        test_metrics = evaluate_epoch(
            model, test_loader, criterion, device
        )
        print("\nDeep-learning model on test set:")
        print(test_metrics)

        X_test, y_test = extract_features(
            model.feature_model, test_loader, device
        )

        scaled_train = scaler.transform(X_train)
        scaled_test = scaler.transform(X_test)

        test_results = {}
        for name, classifier in build_sklearn_classifiers().items():
            try:
                classifier.fit(scaled_train, y_train)
                pred = classifier.predict(scaled_test)
                test_results[name] = classification_metrics(y_test, pred)
            except Exception as exc:
                print(f"Skipping test classifier {name}: {exc}")

    return {
        "model": model,
        "history": history,
        "validation_results": sklearn_results,
        "test_results": test_results,
        "device": str(device),
    }
