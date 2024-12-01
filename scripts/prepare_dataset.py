"""
Prepare OCR datasets for training.

Handles downloading, converting, and splitting datasets.
Supports IAM, PubTables, and custom data.
"""

import argparse
import os
import csv
import json
import shutil
import random
from pathlib import Path


def prepare_iam(output_dir, split_ratio=(0.8, 0.1, 0.1)):
    """Prepare IAM dataset from HuggingFace."""
    from datasets import load_dataset

    print("Downloading IAM dataset from HuggingFace...")
    ds = load_dataset("Teklia/IAM-line")

    for split_name, ratio in zip(["train", "val", "test"], split_ratio):
        split_dir = os.path.join(output_dir, split_name)
        img_dir = os.path.join(split_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        if split_name == "train":
            data = ds["train"]
        elif split_name == "val":
            data = ds["validation"]
        else:
            data = ds["test"]

        labels = []
        for i, item in enumerate(data):
            img = item["image"]
            filename = f"iam_{split_name}_{i:05d}.png"
            img.save(os.path.join(img_dir, filename))
            labels.append((filename, item["text"]))

        # write labels.csv
        with open(os.path.join(split_dir, "labels.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text"])
            writer.writerows(labels)

        print(f"  {split_name}: {len(labels)} samples")


def prepare_custom(source_dir, output_dir, split_ratio=(0.8, 0.1, 0.1)):
    """
    Prepare custom dataset.
    Expects source_dir to have images/ and labels.csv.
    """
    src_labels = os.path.join(source_dir, "labels.csv")
    src_images = os.path.join(source_dir, "images")

    if not os.path.exists(src_labels):
        raise FileNotFoundError(f"No labels.csv in {source_dir}")

    # read all samples
    samples = []
    with open(src_labels, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                img_path = os.path.join(src_images, row[0])
                if os.path.exists(img_path):
                    samples.append((row[0], row[1], img_path))

    random.shuffle(samples)

    # split
    n = len(samples)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    for split_name, split_samples in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        img_dir = os.path.join(split_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        labels = []
        for filename, text, src_path in split_samples:
            dst_path = os.path.join(img_dir, filename)
            shutil.copy2(src_path, dst_path)
            labels.append((filename, text))

        with open(os.path.join(split_dir, "labels.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text"])
            writer.writerows(labels)

        print(f"  {split_name}: {len(labels)} samples")


def create_sample_dataset(output_dir, num_samples=100):
    """Create a dummy dataset for testing the pipeline."""
    from PIL import Image, ImageDraw, ImageFont
    import random
    import string

    for split_name in ["train", "val", "test"]:
        n = num_samples if split_name == "train" else num_samples // 5
        split_dir = os.path.join(output_dir, split_name)
        img_dir = os.path.join(split_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        labels = []
        for i in range(n):
            # generate random text
            text = "".join(random.choices(string.ascii_lowercase + " ", k=random.randint(10, 40)))
            text = text.strip()

            # create image with text
            img = Image.new("RGB", (400, 64), "white")
            draw = ImageDraw.Draw(img)
            draw.text((10, 20), text, fill="black")

            filename = f"sample_{i:04d}.png"
            img.save(os.path.join(img_dir, filename))
            labels.append((filename, text))

        with open(os.path.join(split_dir, "labels.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text"])
            writer.writerows(labels)

        print(f"  {split_name}: {n} samples")


def main():
    parser = argparse.ArgumentParser(description="Prepare OCR datasets")
    parser.add_argument("--dataset", type=str, choices=["iam", "custom", "sample"],
                        required=True)
    parser.add_argument("--source", type=str, default=None, help="Source directory (for custom)")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=100, help="For sample dataset")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.dataset == "iam":
        prepare_iam(args.output)
    elif args.dataset == "custom":
        if not args.source:
            raise ValueError("--source required for custom dataset")
        prepare_custom(args.source, args.output)
    elif args.dataset == "sample":
        create_sample_dataset(args.output, args.num_samples)

    print("Done!")


if __name__ == "__main__":
    main()
