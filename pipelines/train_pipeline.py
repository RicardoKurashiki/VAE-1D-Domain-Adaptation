import os
import sys
import torch
import json

from torch.utils.data import DataLoader

NUM_WORKERS = 0 if sys.platform == "darwin" else 4

from utils import BatchSampler, load_data, train_model
from utils.augmentation_builder import build_augmentation_transforms, get_validation_transforms
from utils.loss_functions import get_loss_function
from utils.scheduler_builder import get_scheduler
from utils.optimizer_builder import get_optimizer
from models import ClassificationModel

DEFAULT_DATA_AUGMENTATION = {
    'rotation_range': 5,
    'horizontal_flip': True,
    'vertical_flip': False,
    'shear_range': 0.0,
    'zoom_range': [1.0, 1.0],
    'width_shift_range': 0.0,
    'height_shift_range': 0.0,
    'brightness_range': [0.85, 1.15],
    'contrast_range': [0.85, 1.15],
    'random_noise_std_range': [0.01, 0.05],
}

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

EARLY_STOPPING_PATIENCE = 50


def run(
    dataset_path,
    backbone,
    layers,
    batch_size,
    epochs,
    output_path="./results/",
    config=None,
    data_augmentation=True,
    mini_batch=True,
    use_early_stopping=True,
    learning_rate=1e-3,
    use_reduce_lr_on_plateau=False,
    verbose=False,
):
    if config is None:
        config = {
            'use_data_augmentation': data_augmentation,
            'mini_batch': mini_batch,
            'use_early_stopping': use_early_stopping,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'learning_rate': learning_rate,
            'data_augmentation': DEFAULT_DATA_AUGMENTATION,
            'loss_function': {
                'name': 'cross_entropy',
                'params': {}
            },
            'scheduler': {
                'name': 'reduce_on_plateau' if use_reduce_lr_on_plateau else 'step_lr',
                'params': {
                    'step_size': 3,
                    'gamma': 0.5,
                    'factor': 0.5,
                    'patience': 3,
                }
            },
        }

    if verbose:
        print("Treinando modelo a partir do dataset:", dataset_path)

    transform = build_augmentation_transforms(config, is_training=True, model_name=backbone)
    val_transform = get_validation_transforms(model_name=backbone)

    train_data = load_data(
        dataset_path,
        split="train",
        transform=transform,
        val_transform=val_transform,
    )
    val_data = load_data(
        dataset_path,
        split="val",
        transform=transform,
        val_transform=val_transform,
    )

    train_dataset = train_data["train"]
    val_dataset = val_data["val"]

    n_classes = train_dataset.n_classes

    classifier_type = config.get('classifier_type', 'default')
    classification_model = ClassificationModel(
        num_classes=n_classes,
        backbone=backbone,
        trainable_layers=layers,
        classifier_type=classifier_type,
    )
    if verbose:
        classification_model.summary()

    os.makedirs(output_path, exist_ok=True)
    architecture_file = classification_model.save_architecture(output_path)
    if verbose:
        print(f"Arquitetura do modelo salva em: {architecture_file}")

    model = classification_model.model

    criterion = get_loss_function(config['loss_function'], num_classes=n_classes)

    optimizer_name = config.get('optimizer', 'adam')
    optimizer = get_optimizer(
        optimizer_name,
        filter(lambda p: p.requires_grad, model.parameters()),
        config.get('learning_rate', learning_rate)
    )

    scheduler = get_scheduler(config['scheduler'], optimizer)

    use_persistent = False

    if config.get('mini_batch', mini_batch):
        if verbose:
            print("Utilizando mini-batch")
        train_sampler = BatchSampler(train_dataset, batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            pin_memory=True,
            num_workers=NUM_WORKERS,
            persistent_workers=use_persistent,
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
        )
    else:
        if verbose:
            print("Utilizando batch size")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle para melhor generalização
            pin_memory=True,
            num_workers=NUM_WORKERS,
            persistent_workers=use_persistent,
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Não shuffle no validation
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=use_persistent,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }

    early_stopping_patience = (
        config.get('early_stopping_patience', EARLY_STOPPING_PATIENCE)
        if config.get('use_early_stopping', use_early_stopping)
        else None
    )

    model, history, metrics = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        num_epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose,
    )

    if verbose:
        print("Salvando pesos do modelo...")

    os.makedirs(output_path, exist_ok=True)

    best_metrics = {
        'best_val_loss': min(history['val_loss']) if history['val_loss'] else None,
        'best_val_acc': max(history['val_acc']) if history['val_acc'] else None,
    }

    classification_model.save_weights(
        output_path,
        config=config,
        epoch=len(history['train_loss']),
        metrics=best_metrics
    )

    if verbose:
        print("Salvando métricas...")

    metrics_file = os.path.join(output_path, "model_metrics.json")
    all_metrics = {
        "training_history": history,
        "computational_metrics": metrics,
        "training_config": config,
    }

    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

        if verbose:
            print(f"Métricas salvas em {metrics_file}")

    return output_path
