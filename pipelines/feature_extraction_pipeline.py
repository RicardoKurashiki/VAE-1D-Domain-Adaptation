#!/usr/bin/env python3

import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

from models import ClassificationModel
from utils import load_data

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def _count_classes(dataset_path):
    train_dir = os.path.join(dataset_path, "train")
    return len([e for e in os.scandir(train_dir) if e.is_dir()])


def load_model(weights_dir, pretrained_model, n_classes, classifier_type="default"):
    classification_model = ClassificationModel(
        num_classes=n_classes,
        backbone=pretrained_model,
        classifier_type=classifier_type,
    )
    classification_model.load_weights(weights_dir)

    model = classification_model.model.extractor
    model.to(device)
    model.eval()

    return model


def run(
    weights_dir,
    dataset_path,
    pretrained_model,
    batch_size=32,
    prefix="",
    classifier_type="default",
    features_dir=None,
):
    n_classes = _count_classes(dataset_path)
    extraction_model = load_model(weights_dir, pretrained_model, n_classes, classifier_type)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    splits = {
        "train": load_data(dataset_path, split="train", transform=transform, val_transform=transform)["train"],
        "val": load_data(dataset_path, split="val", transform=transform, val_transform=transform)["val"],
        "test": load_data(dataset_path, split="test", transform=transform, val_transform=transform)["test"],
    }

    output_path = features_dir if features_dir is not None else os.path.join(weights_dir, "features/")
    os.makedirs(output_path, exist_ok=True)

    for phase, dataset in splits.items():
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
        )

        all_features = []
        all_labels = []

        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                features = extraction_model(inputs)
            all_features.append(features.cpu().detach().numpy())
            all_labels.append(labels.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        np.save(os.path.join(output_path, f"{prefix}_{phase}_features.npy"), all_features)
        np.save(os.path.join(output_path, f"{prefix}_{phase}_labels.npy"), all_labels)

    return output_path
