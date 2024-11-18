"""
TrOCR fine-tuning for handwritten text recognition.
Runs on AMD GPUs via ROCm/HIP.

Had some issues with flash attention on ROCm — had to disable it.
Also the tokenizer needed some workarounds. See notes in experiments/notes.md
"""

import argparse
import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AutoTokenizer,
    get_scheduler,
)
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IAMDataset(Dataset):
    """Load IAM handwritten text dataset for TrOCR."""

    def __init__(self, processor, dataset_split, max_target_length=128):
        self.processor = processor
        self.dataset = dataset_split
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        text = item["text"]

        # processor handles both image and text encoding
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        # replace padding token id with -100 so it's ignored in loss
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


class CustomOCRDataset(Dataset):
    """Load custom dataset from a directory of images + labels."""

    def __init__(self, processor, data_dir, max_target_length=128):
        self.processor = processor
        self.max_target_length = max_target_length
        self.samples = []

        # expect images/ dir and labels.csv
        labels_path = os.path.join(data_dir, "labels.csv")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"No labels.csv found in {data_dir}")

        import csv
        with open(labels_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                img_path = os.path.join(data_dir, "images", row[0])
                if os.path.exists(img_path):
                    self.samples.append((img_path, row[1]))

        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def compute_cer(preds, targets):
    """Compute character error rate."""
    import editdistance
    total_dist = 0
    total_len = 0
    for pred, target in zip(preds, targets):
        total_dist += editdistance.eval(pred, target)
        total_len += len(target)
    return total_dist / max(total_len, 1)


def train(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        # NOTE: on ROCm, some operations might fall back to CPU
        # if you see weird slowness check HIP compatibility warnings

    # load processor and model
    model_name = config.get("model_name", "microsoft/trocr-base-handwritten")
    logger.info(f"Loading model: {model_name}")

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # ROCm workaround: disable flash attention if it's trying to use it
    # flash_attn doesn't work properly on ROCm as of 6.1.x
    model.config.use_flash_attention = False

    model.to(device)

    # set decoder config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # load dataset
    dataset_name = config.get("dataset", "iam")
    if dataset_name == "iam":
        ds = load_dataset("Teklia/IAM-line")
        train_ds = IAMDataset(processor, ds["train"])
        val_ds = IAMDataset(processor, ds["validation"])
    else:
        data_dir = config.get("data_dir", "data/custom")
        train_ds = CustomOCRDataset(processor, os.path.join(data_dir, "train"))
        val_ds = CustomOCRDataset(processor, os.path.join(data_dir, "val"))

    batch_size = config.get("batch_size", 8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # optimizer and scheduler
    lr = config.get("learning_rate", 5e-5)
    epochs = config.get("epochs", 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    num_training_steps = epochs * len(train_loader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps,
    )

    # training loop
    output_dir = config.get("output_dir", "outputs/trocr")
    os.makedirs(output_dir, exist_ok=True)
    best_cer = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} — avg loss: {avg_loss:.4f}")

        # validation
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                generated = model.generate(pixel_values, max_length=128)
                pred_texts = processor.batch_decode(generated, skip_special_tokens=True)

                # decode targets (replace -100 with pad token)
                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = processor.tokenizer.pad_token_id
                target_texts = processor.batch_decode(labels_for_decode, skip_special_tokens=True)

                all_preds.extend(pred_texts)
                all_targets.extend(target_texts)

        cer = compute_cer(all_preds, all_targets)
        logger.info(f"Validation CER: {cer:.4f}")

        if cer < best_cer:
            best_cer = cer
            model.save_pretrained(os.path.join(output_dir, "best_model"))
            processor.save_pretrained(os.path.join(output_dir, "best_model"))
            logger.info(f"Saved best model (CER: {cer:.4f})")

    logger.info(f"Training complete. Best CER: {best_cer:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/trocr_finetune.yaml")
    args = parser.parse_args()
    train(args.config)
