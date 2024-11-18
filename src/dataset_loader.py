"""
Dataset loaders for various OCR datasets.

Supports:
- IAM (handwritten text lines)
- PubTables-1M (table detection/structure)
- Custom datasets (images + labels.csv)
"""

import os
import csv
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
import torchvision.transforms as T


class IAMLinesDataset(Dataset):
    """IAM handwritten text line dataset from HuggingFace."""

    def __init__(self, split="train", transform=None, max_length=128):
        self.dataset = load_dataset("Teklia/IAM-line", split=split)
        self.max_length = max_length
        self.transform = transform or T.Compose([
            T.Resize((32, 384)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        text = item["text"]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "text": text}


class PubTablesDataset(Dataset):
    """PubTables-1M dataset for table detection and structure recognition."""

    def __init__(self, data_dir, split="train", task="detection", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.task = task
        self.transform = transform

        # load annotations
        ann_file = os.path.join(data_dir, f"{split}_annotations.json")
        if os.path.exists(ann_file):
            with open(ann_file) as f:
                self.annotations = json.load(f)
        else:
            # fallback: scan directory
            self.annotations = self._scan_directory()

        print(f"PubTables {split}: {len(self.annotations)} samples")

    def _scan_directory(self):
        """Scan directory structure for images and labels."""
        img_dir = os.path.join(self.data_dir, self.split, "images")
        ann_dir = os.path.join(self.data_dir, self.split, "annotations")

        samples = []
        if not os.path.exists(img_dir):
            return samples

        for img_file in sorted(os.listdir(img_dir)):
            if not img_file.endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(img_dir, img_file)
            ann_file = os.path.join(ann_dir, Path(img_file).stem + ".json")

            sample = {"image_path": img_path}
            if os.path.exists(ann_file):
                with open(ann_file) as f:
                    sample["annotation"] = json.load(f)

            samples.append(sample)

        return samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        result = {"image": image}

        if "annotation" in sample:
            result["annotation"] = sample["annotation"]

        return result


class CustomOCRDataset(Dataset):
    """
    Custom OCR dataset from a directory.
    Expected structure:
        data_dir/
            images/
                img1.jpg
                img2.jpg
            labels.csv   (filename, text)
    """

    def __init__(self, data_dir, transform=None, max_length=128):
        self.data_dir = data_dir
        self.transform = transform
        self.max_length = max_length
        self.samples = []

        labels_file = os.path.join(data_dir, "labels.csv")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(
                f"labels.csv not found in {data_dir}\n"
                f"Expected format: filename,text"
            )

        with open(labels_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    img_path = os.path.join(data_dir, "images", row[0])
                    if os.path.exists(img_path):
                        self.samples.append((img_path, row[1]))

        print(f"Custom dataset: {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"image": image, "text": text, "path": img_path}


class MultiLanguageOCRDataset(Dataset):
    """
    Multi-language OCR dataset loader.
    Supports Chinese, Japanese, Korean, etc.

    Data format: same as CustomOCRDataset but with a lang field.
    """

    def __init__(self, data_dir, lang="en", transform=None):
        self.data_dir = data_dir
        self.lang = lang
        self.transform = transform
        self.samples = []

        lang_dir = os.path.join(data_dir, lang)
        if not os.path.exists(lang_dir):
            lang_dir = data_dir

        labels_file = os.path.join(lang_dir, "labels.csv")
        with open(labels_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    img_path = os.path.join(lang_dir, "images", row[0])
                    if os.path.exists(img_path):
                        self.samples.append((img_path, row[1]))

        print(f"Multi-lang dataset ({lang}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"image": image, "text": text, "lang": self.lang}


def get_dataset(name, **kwargs):
    """Factory function to get dataset by name."""
    datasets_map = {
        "iam": IAMLinesDataset,
        "pubtables": PubTablesDataset,
        "custom": CustomOCRDataset,
        "multilang": MultiLanguageOCRDataset,
    }

    if name not in datasets_map:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets_map.keys())}")

    return datasets_map[name](**kwargs)
