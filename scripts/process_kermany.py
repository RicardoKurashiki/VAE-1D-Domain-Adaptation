from time import sleep
import os
import shutil
import argparse
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split

def create_dirs(train_path, val_path, test_path):
    print("Making directories for dataset...")

    splits = [train_path, val_path, test_path]
    classes = ["NORMAL", "PNEUMONIA"]

    for split in splits:
        if os.path.exists(split):
            print(f"Removing {split}")
            shutil.rmtree(split)

        for cls in classes:
            class_path = os.path.join(split, cls)
            os.makedirs(class_path)
            print(f"Creating {class_path}")

    print("Directories created successfully\n")

def create_dataframe(path):
    data = []

    classes = ["NORMAL", "PNEUMONIA"]
    splits = ["train", "test"]

    for sp in splits:
        for cls in classes:
            img_paths = [img for img in os.listdir(os.path.join(path, sp, cls)) if img.endswith(".jpeg")]
            for img_path in img_paths:
                data.append({"className": cls, "path": os.path.join(path, sp, cls, img_path), "imgName": img_path})

    return pd.DataFrame(data=data)

def split_data(df, train_ratio, val_ratio, test_ratio, seed):
    train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=seed, stratify=df["className"], shuffle=True)
    val_df, test_df   = train_test_split(temp_df, train_size=test_ratio/(test_ratio+val_ratio), random_state=seed, stratify=temp_df["className"], shuffle=True)

    return train_df, val_df, test_df

def save_images(df, path, target_size=224):
    for index, row in df.iterrows():
        try:
            img = resize_image(row["path"], target_size)
            save_path = os.path.join(path, row["className"], row["imgName"])
            counter = 1
            while os.path.exists(save_path):
                save_path = save_path.replace(".jpeg", f"_{counter}.jpeg")
                counter += 1
            img.save(save_path, "JPEG")
        except Exception as e:
            print(f"Save Image error {row['path']}: {e}")

def resize_image(image, target_size=224):
    try:
        img = Image.open(image).convert('RGB')
        img = img.resize((target_size, target_size))
        return img
    except Exception as e:
        print(f"Resize Image error {image}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/original/CellData/chest_xray/",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed/kermany/",
        help="Path to save the processed data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of training data",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of validation data",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Ratio of test data",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="Target size for images",
    )

    args = parser.parse_args()

    dataset_path = args.dataset
    output_path = args.output
    seed = args.seed
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    target_size = args.target_size

    train_path = os.path.join(output_path, "train")
    val_path = os.path.join(output_path, "val")
    test_path = os.path.join(output_path, "test")

    create_dirs(train_path, val_path, test_path)
    df = create_dataframe(dataset_path)
    train_df, val_df, test_df = split_data(df, train_ratio, val_ratio, test_ratio, seed)

    print(f"\nFull Dataset ({len(df)} Samples)")
    print(f"Full Dataset Split: {df['className'].value_counts()}")

    print(f"\nTrain Dataset ({len(train_df)} Samples)")
    print(f"Train Split: {train_df['className'].value_counts()}")

    print(f"\nVal Dataset ({len(val_df)} Samples)")
    print(f"Val Split: {val_df['className'].value_counts()}")

    print(f"\nTest Dataset ({len(test_df)} Samples)")
    print(f"Test Split: {test_df['className'].value_counts()}")

    save_images(train_df, train_path, target_size)
    save_images(val_df, val_path, target_size)
    save_images(test_df, test_path, target_size)

    print(f"\nTrain set split: {len(os.listdir(os.path.join(train_path, "NORMAL"))) + len(os.listdir(os.path.join(train_path, "PNEUMONIA")))}")
    print(f"Val set split: {len(os.listdir(os.path.join(val_path, "NORMAL"))) + len(os.listdir(os.path.join(val_path, "PNEUMONIA")))}")
    print(f"Test set split: {len(os.listdir(os.path.join(test_path, "NORMAL"))) + len(os.listdir(os.path.join(test_path, "PNEUMONIA")))}")

    print("\nKermany dataset processed successfully!")
