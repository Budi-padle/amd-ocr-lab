"""
Document layout analysis using YOLOv8.

Detects regions like text blocks, tables, figures, headers, etc.
Uses a fine-tuned YOLOv8 model on DocLayNet or similar dataset.

ultralytics works on ROCm through the PyTorch backend — no special
modifications needed beyond what you did for PyTorch itself.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    print("Install ultralytics: pip install ultralytics")
    YOLO = None


# DocLayNet-style class labels
LAYOUT_CLASSES = {
    0: "text",
    1: "title",
    2: "list",
    3: "table",
    4: "figure",
    5: "caption",
    6: "footnote",
    7: "formula",
    8: "page_header",
    9: "page_footer",
    10: "section_header",
}


class LayoutAnalyzer:
    """Document layout analysis using YOLOv8."""

    def __init__(self, model_path=None, confidence=0.5, device=None):
        if YOLO is None:
            raise ImportError("ultralytics not installed")

        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.confidence = confidence

        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded layout model from {model_path}")
        else:
            # use pretrained yolov8 and fine-tune later
            self.model = YOLO("yolov8n.pt")
            print("Using base YOLOv8n — you'll want to fine-tune this")

    def analyze(self, image_path):
        """Analyze document layout. Returns list of detected regions."""
        results = self.model.predict(
            source=image_path,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        regions = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().tolist()

                regions.append({
                    "class_id": cls_id,
                    "class_name": LAYOUT_CLASSES.get(cls_id, "unknown"),
                    "confidence": round(conf, 3),
                    "bbox": [int(x) for x in xyxy],
                })

        # sort by vertical position
        regions.sort(key=lambda r: r["bbox"][1])
        return regions

    def analyze_batch(self, image_dir, output_file=None):
        """Analyze multiple images."""
        import json

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        results = {}

        image_paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if Path(f).suffix.lower() in image_extensions
        ]

        for path in image_paths:
            regions = self.analyze(path)
            results[os.path.basename(path)] = regions

        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

        return results

    def extract_regions(self, image_path, region_type=None):
        """Extract and crop specific region types from an image."""
        regions = self.analyze(image_path)
        img = Image.open(image_path).convert("RGB")

        if region_type:
            regions = [r for r in regions if r["class_name"] == region_type]

        crops = []
        for region in regions:
            bbox = region["bbox"]
            crop = img.crop(bbox)
            crops.append({
                "region": region,
                "crop": crop,
            })

        return crops

    def visualize(self, image_path, output_path=None):
        """Draw detected regions on the image."""
        img = cv2.imread(image_path)
        regions = self.analyze(image_path)

        # colors for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128),
        ]

        for region in regions:
            bbox = region["bbox"]
            cls_id = region["class_id"]
            color = colors[cls_id % len(colors)]

            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{region['class_name']} {region['confidence']:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Saved to {output_path}")

        return img


def fine_tune_layout(data_yaml, epochs=50, img_size=1024, batch_size=8):
    """Fine-tune YOLOv8 on custom layout dataset."""
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name="layout_model",
        patience=10,
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Document layout analysis")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--visualize", type=str, default=None, help="Output path for visualization")
    args = parser.parse_args()

    analyzer = LayoutAnalyzer(model_path=args.model, confidence=args.conf)
    regions = analyzer.analyze(args.image)

    print(f"Found {len(regions)} regions:")
    for r in regions:
        print(f"  [{r['confidence']:.2f}] {r['class_name']} at {r['bbox']}")

    if args.visualize:
        analyzer.visualize(args.image, args.visualize)
