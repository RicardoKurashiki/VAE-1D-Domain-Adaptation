"""
Train the ResNet-18 classifier on one dataset (source), evaluate it cross-dataset,
extract CNN features, compute per-class centroids, then train and test the VAE 1D
domain-alignment autoencoder (CenterLoss) against the other dataset (target).

There is only one scenario: everything below is a plain constant, edit it directly.
The only axis that is swept automatically is TARGET_DATA_FRACTIONS — how much of the
target dataset the aligner is allowed to see.
"""
import os
import json
import argparse
import random

import torch
import numpy as np

from pipelines import (
    train_pipeline,
    test_pipeline,
    feature_extraction_pipeline,
    cluster_pipeline,
)
from utils import cache_paths
from datetime import datetime

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

# Source Dataset
SOURCE_DATASET = "kermany"
ALL_DATASETS = ["kermany", "rsna"]
DATA_ROOT = "data/processed"

# Classifier
MODEL = "resnet"
LAYERS = "all"
CLASSIFIER_TYPE = "default"
BATCH_SIZE = 32
EPOCHS = 500
OPTIMIZER = "adam"
LEARNING_RATE = 1e-4
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20
MINI_BATCH = True
VERBOSE = True

USE_DATA_AUGMENTATION = True
DATA_AUGMENTATION = {
    "rotation_range": 10,
    "horizontal_flip": True,
    "vertical_flip": False,
    "shear_range": 0.0,
    "zoom_range": [0.8, 1.2],
    "width_shift_range": 0.0,
    "height_shift_range": 0.0,
    "brightness_range": None,
    "contrast_range": None,
    "random_noise_std_range": None,
}

LOSS_FUNCTION = {"name": "cross_entropy", "params": {}}
SCHEDULER = {
    "name": "reduce_on_plateau",
    "params": {"mode": "min", "factor": 0.1, "patience": 5, "verbose": True},
}

# Pipeline steps to run
RUN_TRAINING = True
RUN_TEST_PIPELINE = True
RUN_FEATURE_EXTRACTION = True
RUN_CLUSTERING = True
K_CLUSTERS = 2  # KMeans centroids for the alignment CenterLoss (= number of classes)

# Domain-alignment autoencoder: VAE + CenterLoss, align_weight=0.9, kl_weight=0.1
# (mmd_weight is derived: 1 - align_weight - kl_weight = 0.0). Swept across every
# fraction of target-domain data the aligner is allowed to see during training.
RUN_AUTOENCODER_TRAINING = True
RUN_AUTOENCODER_TEST = True
AUTOENCODER_ARCHITECTURE = "vae"
ALIGN_FN = "center"
ALIGN_WEIGHT = 0.9
ALIGN_MARGIN = 5.0
KL_WEIGHT = 0.1
AE_EPOCHS = 500
AE_LEARNING_RATE = 1e-4
AE_BATCH_SIZE = 64
AE_EARLY_STOPPING_PATIENCE = 10
TARGET_DATA_FRACTIONS = [1.0, 0.8, 0.4, 0.2, 0.1, 0.05, 0.01]
# ═══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(prog="VAE 1D Domain Adaptation")
parser.add_argument(
    "--source",
    choices=ALL_DATASETS,
    default=SOURCE_DATASET,
    help=f"Dataset the classifier is trained on (default: {SOURCE_DATASET})",
)
parser.add_argument(
    "--force-retrain",
    action="store_true",
    help="Re-train even if cached weights exist (implies --force-reextract)",
)
parser.add_argument(
    "--force-reextract",
    action="store_true",
    help="Re-extract features even if cached features exist",
)
args = parser.parse_args()


def _build_classifier_config():
    return {
        "model": MODEL,
        "layers": LAYERS,
        "classifier_type": CLASSIFIER_TYPE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "use_early_stopping": USE_EARLY_STOPPING,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "mini_batch": MINI_BATCH,
        "use_data_augmentation": USE_DATA_AUGMENTATION,
        "data_augmentation": DATA_AUGMENTATION,
        "loss_function": LOSS_FUNCTION,
        "scheduler": SCHEDULER,
    }


def main():
    source_dataset = args.source
    target_datasets = [d for d in ALL_DATASETS if d != source_dataset]

    current_time = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_path = os.path.join("results", f"{MODEL}18_{source_dataset}")

    print(f"Source dataset: {source_dataset} | Target dataset(s): {target_datasets}")
    print(f"Output path: {output_path}")

    os.makedirs(output_path, exist_ok=True)

    classifier_config = _build_classifier_config()

    with open(os.path.join(output_path, "run_config.json"), "w") as f:
        json.dump({"source_dataset": source_dataset, "timestamp": current_time,
                   "classifier": classifier_config}, f, indent=2)

    # Shared cache keyed by (model, source dataset, train hash). Weights and raw
    # features live here and are reused across reruns that share the same
    # training-affecting fields.
    root = cache_paths.cache_root()
    train_hash = cache_paths.compute_train_hash(classifier_config)
    weights_dir = cache_paths.weights_cache_dir(root, MODEL, source_dataset, train_hash)
    features_dir = cache_paths.features_cache_dir(root, MODEL, source_dataset, train_hash)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    force_retrain = args.force_retrain
    force_reextract = args.force_reextract or args.force_retrain

    with open(os.path.join(output_path, "cache_info.json"), "w") as f:
        json.dump({
            "train_hash": train_hash,
            "weights_dir": weights_dir,
            "features_dir": features_dir,
        }, f, indent=2)

    print(f"Cache: train_hash={train_hash}")
    print(f"  weights_dir={weights_dir}")
    print(f"  features_dir={features_dir}")

    source_dataset_path = os.path.join(DATA_ROOT, source_dataset)

    # Training (train once per train_hash; cache sentinel governs the skip)
    if RUN_TRAINING and (force_retrain or not cache_paths.weights_ready(weights_dir)):
        train_pipeline(
            source_dataset_path,
            MODEL,
            LAYERS,
            BATCH_SIZE,
            EPOCHS,
            weights_dir,
            config=classifier_config,
            verbose=VERBOSE,
        )
        cache_paths.mark_weights_complete(weights_dir)
        cache_paths.invalidate_features(features_dir)
    elif RUN_TRAINING:
        print(f"Reusing cached weights: {weights_dir}")

    for dataset_name in [source_dataset] + target_datasets:
        dataset_path = os.path.join(DATA_ROOT, dataset_name)
        test_dir = os.path.join(dataset_path, "test")
        has_test = os.path.isdir(test_dir)

        if RUN_TEST_PIPELINE and has_test:
            test_pipeline(
                weights_dir,
                dataset_path,
                MODEL,
                BATCH_SIZE,
                prefix=dataset_name,
                verbose=VERBOSE,
                output_dir=output_path,
            )
        elif RUN_TEST_PIPELINE and VERBOSE:
            print(f"Skipping test pipeline - directory not found: {test_dir}")

        if RUN_FEATURE_EXTRACTION and has_test:
            if force_reextract or not cache_paths.features_ready(features_dir, dataset_name):
                feature_extraction_pipeline(
                    weights_dir,
                    dataset_path,
                    MODEL,
                    BATCH_SIZE,
                    prefix=dataset_name,
                    classifier_type=CLASSIFIER_TYPE,
                    features_dir=features_dir,
                )
                cache_paths.mark_features_complete(features_dir, dataset_name)
            else:
                print(f"Reusing cached features for {dataset_name}")
        elif RUN_FEATURE_EXTRACTION and VERBOSE:
            print(f"Skipping feature extraction - directory not found: {test_dir}")

        if RUN_CLUSTERING and has_test:
            cluster_pipeline(
                output_path,
                dataset_name,
                k=K_CLUSTERS,
                source_dataset_name=source_dataset,
                features_dir=features_dir,
            )
        elif RUN_CLUSTERING and VERBOSE:
            print(f"Skipping clustering - directory not found: {test_dir}")

    if RUN_AUTOENCODER_TRAINING or RUN_AUTOENCODER_TEST:
        from pipelines import train_autoencoder, test_autoencoder
        source_labels = np.load(os.path.join(features_dir, f"{source_dataset}_train_labels.npy"))
        n_classes = int(source_labels.max()) + 1

    if RUN_AUTOENCODER_TRAINING:
        test_fn = None
        test_kwargs = None
        if RUN_AUTOENCODER_TEST:
            test_fn = test_autoencoder.run
            test_kwargs = {
                "model_path": output_path,
                "weights_dir": weights_dir,
                "features_dir": features_dir,
                "pretrained_model": MODEL,
                "num_classes": n_classes,
                "classifier_type": CLASSIFIER_TYPE,
                "output_path_base": os.path.join(output_path, "autoencoder_test"),
            }

        train_autoencoder.sweep(
            latent_path=features_dir,
            centroid_path=os.path.join(output_path, "centroids"),
            source_dataset=source_dataset,
            target_datasets=target_datasets,
            architectures=[AUTOENCODER_ARCHITECTURE],
            align_fns=[ALIGN_FN],
            align_weights=[ALIGN_WEIGHT],
            align_margin=ALIGN_MARGIN,
            kl_weights=[KL_WEIGHT],
            target_data_fractions=TARGET_DATA_FRACTIONS,
            epochs=AE_EPOCHS,
            lr=AE_LEARNING_RATE,
            batch_size=AE_BATCH_SIZE,
            early_stopping_patience=AE_EARLY_STOPPING_PATIENCE,
            output_path=os.path.join(output_path, "autoencoder"),
            test_fn=test_fn,
            test_kwargs=test_kwargs,
        )
    elif RUN_AUTOENCODER_TEST:
        test_autoencoder.sweep_test(
            model_path=output_path,
            ae_sweep_path=os.path.join(output_path, "autoencoder"),
            target_datasets=target_datasets,
            pretrained_model=MODEL,
            num_classes=n_classes,
            classifier_type=CLASSIFIER_TYPE,
            output_path=os.path.join(output_path, "autoencoder_test"),
            weights_dir=weights_dir,
            features_dir=features_dir,
        )


if __name__ == "__main__":
    main()
