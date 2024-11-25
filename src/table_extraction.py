"""
Table extraction using a TableNet-style approach.

TableNet detects table regions in documents and extracts their structure.
This is a simplified version — the full model needs more training data
than what I have, but it gets the gist for simple tables.

Currently training on PubTables-1M dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import os


class TableNetEncoder(nn.Module):
    """VGG16-based encoder for TableNet."""

    def __init__(self, pretrained=True):
        super().__init__()
        vgg = models.vgg16(pretrained=pretrained)
        # use first few conv blocks from VGG
        self.features = vgg.features

        # freeze early layers — they're generic enough
        for param in list(self.features.parameters())[:10]:
            param.requires_grad = False

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in {3, 8, 15, 22}:  # after each conv block
                features.append(x)
        return features


class TableDecoder(nn.Module):
    """Decoder for table region detection."""

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        return self.decoder(x)


class ColumnDecoder(nn.Module):
    """Decoder for column region detection."""

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        return self.decoder(x)


class TableNet(nn.Module):
    """TableNet for table and column detection."""

    def __init__(self, pretrained_encoder=True):
        super().__init__()
        self.encoder = TableNetEncoder(pretrained=pretrained_encoder)
        self.table_decoder = TableDecoder()
        self.column_decoder = ColumnDecoder()

    def forward(self, x):
        features = self.encoder(x)
        # use the last feature map
        x = features[-1]
        table_mask = self.table_decoder(x)
        column_mask = self.column_decoder(x)
        return table_mask, column_mask


class TableExtractor:
    """High-level table extraction from document images."""

    def __init__(self, model_path=None, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = TableNet(pretrained_encoder=(model_path is None))
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image_path, target_size=(1024, 1024)):
        """Preprocess image for the model."""
        img = Image.open(image_path).convert("RGB")
        original_size = img.size

        img_resized = img.resize(target_size)
        img_np = np.array(img_resized, dtype=np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = (img_np - mean) / std

        # HWC -> CHW, add batch dim
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)
        return img_tensor, original_size

    def detect_tables(self, image_path, threshold=0.5):
        """Detect table regions in an image."""
        img_tensor, original_size = self.preprocess(image_path)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            table_mask, column_mask = self.model(img_tensor)

        # upsample to original size
        table_mask = F.interpolate(
            table_mask, size=original_size[::-1], mode="bilinear", align_corners=False
        )
        column_mask = F.interpolate(
            column_mask, size=original_size[::-1], mode="bilinear", align_corners=False
        )

        table_mask_np = torch.sigmoid(table_mask).squeeze().cpu().numpy()
        column_mask_np = torch.sigmoid(column_mask).squeeze().cpu().numpy()

        # threshold to get binary masks
        table_binary = (table_mask_np > threshold).astype(np.uint8) * 255
        column_binary = (column_mask_np > threshold).astype(np.uint8) * 255

        return table_binary, column_binary

    def extract_table_regions(self, image_path, threshold=0.5):
        """Get bounding boxes for detected tables."""
        table_mask, _ = self.detect_tables(image_path, threshold)

        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > 1000:  # filter tiny regions
                regions.append({
                    "bbox": [x, y, x + w, y + h],
                    "area": area,
                })

        # sort by area descending
        regions.sort(key=lambda r: r["area"], reverse=True)
        return regions

    def extract_table_content(self, image_path, ocr_engine=None, threshold=0.5):
        """Extract actual table content using detected regions + OCR."""
        regions = self.extract_table_regions(image_path, threshold)

        if ocr_engine is None:
            # lazy import
            from paddle_ocr_runner import OCRWrapper
            ocr_engine = OCRWrapper()

        img = Image.open(image_path).convert("RGB")
        results = []

        for i, region in enumerate(regions):
            bbox = region["bbox"]
            crop = img.crop(bbox)

            # save crop temporarily
            tmp_path = f"/tmp/table_crop_{i}.jpg"
            crop.save(tmp_path)

            detections = ocr_engine.recognize(tmp_path)
            os.remove(tmp_path)

            results.append({
                "region": region,
                "content": detections,
            })

        return results


def train_table_net(dataset_dir, epochs=20, batch_size=4, lr=1e-4):
    """Train TableNet on a dataset of document images with masks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TableNet(pretrained_encoder=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # this is just a skeleton — you need to fill in dataset loading
    # for PubTables-1M or your own annotated data
    print("Training TableNet...")
    print(f"Device: {device}")
    print("Note: you need to implement the dataset loader for your data")

    for epoch in range(epochs):
        model.train()
        # TODO: actual training loop
        # for batch in dataloader:
        #     images, table_masks, column_masks = batch
        #     pred_table, pred_column = model(images.to(device))
        #     loss = criterion(pred_table, table_masks.to(device)) + \
        #            criterion(pred_column, column_masks.to(device))
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()
        print(f"Epoch {epoch + 1}/{epochs} — (fill in your training data)")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Table extraction")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    extractor = TableExtractor(model_path=args.model)
    regions = extractor.extract_table_regions(args.image, threshold=args.threshold)

    print(f"Found {len(regions)} table regions:")
    for i, region in enumerate(regions):
        print(f"  Table {i + 1}: bbox={region['bbox']}, area={region['area']}")
