import os
import pandas as pd

from utils.custom_dataset import CustomDataset


def gen_dataframe(root_dir):
    map_result = {"path": [], "label": []}
    if not os.path.isdir(root_dir):
        return pd.DataFrame(map_result)

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".gif",
        ".webp",
    }

    class_names = os.listdir(root_dir)
    for class_name in class_names:
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path) or class_name.startswith("."):
            continue
        for img in os.listdir(class_path):
            if img.startswith("."):
                continue

            img_path = os.path.join(class_path, img)

            if not os.path.isfile(img_path):
                continue

            _, ext = os.path.splitext(img)
            if ext.lower() not in valid_extensions:
                continue

            map_result["path"].append(img_path)
            map_result["label"].append(class_name)
    return pd.DataFrame(map_result)


def load_data(path, split="train", transform=None, val_transform=None):
    """
    Load data from a specific split directory.

    Args:
        path: Base path containing 'train/', 'val/', 'test/' subdirectories
        split: Which split to load ('train', 'val', 'test')
        transform: Transform for training/test data
        val_transform: Transform for validation data
    """
    split_path = os.path.join(path, split)
    df = gen_dataframe(split_path)

    if df.empty:
        raise ValueError(f"No images found in {split} dataset path: {split_path}")

    if split == "train":
        dataset = CustomDataset(df, transform=transform)
        print(f"Found {len(df)} samples in the training dataset")
    elif split == "val":
        dataset = CustomDataset(df, transform=val_transform)
        print(f"Found {len(df)} samples in the validation dataset")
    elif split == "test":
        dataset = CustomDataset(df, transform=transform)
        print(f"Found {len(df)} samples in the test dataset")
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

    return {split: dataset}
