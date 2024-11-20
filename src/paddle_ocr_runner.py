"""
PaddleOCR inference wrapper.
PaddleOCR is actually pretty solid — good multilingual support
and runs fine on ROCm through the CPU fallback (not ideal but works).

For actual GPU acceleration with Paddle on AMD, you need paddle-rocm
build which is... not straightforward. I'm using CPU for now.
"""

import argparse
import os
import json
import time
from pathlib import Path

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("Install PaddleOCR: pip install paddleocr paddlepaddle")
    raise

from PIL import Image
import cv2
import numpy as np


class OCRWrapper:
    """Wrapper around PaddleOCR with some convenience methods."""

    def __init__(
        self,
        lang="en",
        use_gpu=False,
        det_model_dir=None,
        rec_model_dir=None,
        use_angle_cls=True,
    ):
        self.lang = lang
        self.use_gpu = use_gpu

        kwargs = {
            "lang": lang,
            "use_gpu": use_gpu,
            "use_angle_cls": use_angle_cls,
            "show_log": False,
        }

        if det_model_dir:
            kwargs["det_model_dir"] = det_model_dir
        if rec_model_dir:
            kwargs["rec_model_dir"] = rec_model_dir

        self.ocr = PaddleOCR(**kwargs)
        print(f"PaddleOCR initialized (lang={lang}, gpu={use_gpu})")

    def recognize(self, image_path):
        """Run OCR on a single image. Returns list of (text, confidence) tuples."""
        result = self.ocr.ocr(image_path, cls=True)
        if not result or not result[0]:
            return []

        detections = []
        for line in result[0]:
            bbox = line[0]  # list of 4 points
            text = line[1][0]
            confidence = line[1][1]
            detections.append({
                "text": text,
                "confidence": confidence,
                "bbox": bbox,
            })

        return detections

    def recognize_region(self, image_path, bbox):
        """Run OCR on a specific region of an image."""
        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        # bbox format: [x1, y1, x2, y2] normalized or absolute
        x1, y1, x2, y2 = bbox
        if max(bbox) <= 1.0:
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        result = self.ocr.ocr(crop, cls=True)
        if not result or not result[0]:
            return []

        return [
            {"text": line[1][0], "confidence": line[1][1]}
            for line in result[0]
        ]

    def batch_recognize(self, image_dir, output_file=None):
        """Run OCR on all images in a directory."""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        results = {}

        image_paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if Path(f).suffix.lower() in image_extensions
        ]

        print(f"Processing {len(image_paths)} images...")

        for i, path in enumerate(image_paths):
            start = time.time()
            detections = self.recognize(path)
            elapsed = time.time() - start

            results[os.path.basename(path)] = {
                "detections": detections,
                "time": elapsed,
                "text": " ".join([d["text"] for d in detections]),
            }

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(image_paths)} done")

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {output_file}")

        return results

    def get_full_text(self, image_path):
        """Just get the full text, no bounding boxes."""
        detections = self.recognize(image_path)
        # sort by vertical position then horizontal
        detections.sort(key=lambda d: (
            min(p[1] for p in d["bbox"]),
            min(p[0] for p in d["bbox"]),
        ))
        return "\n".join([d["text"] for d in detections])


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR inference")
    parser.add_argument("--image", type=str, required=True, help="Image path or directory")
    parser.add_argument("--lang", type=str, default="en", help="Language")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (Paddle needs rocm build)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--region", type=float, nargs=4, default=None,
                        help="Bounding box: x1 y1 x2 y2")
    args = parser.parse_args()

    wrapper = OCRWrapper(lang=args.lang, use_gpu=args.gpu)

    if os.path.isdir(args.image):
        results = wrapper.batch_recognize(args.image, args.output)
        total_time = sum(r["time"] for r in results.values())
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Avg per image: {total_time / len(results):.3f}s")
    elif args.region:
        results = wrapper.recognize_region(args.image, args.region)
        for r in results:
            print(f"[{r['confidence']:.2f}] {r['text']}")
    else:
        text = wrapper.get_full_text(args.image)
        print(text)


if __name__ == "__main__":
    main()
