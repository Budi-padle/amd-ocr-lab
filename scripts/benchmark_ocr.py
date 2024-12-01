"""
Benchmark different OCR engines on the same dataset.
Compares speed and accuracy between TrOCR, PaddleOCR, and others.

Usage:
    python scripts/benchmark_ocr.py --dataset path/to/images --labels path/to/labels.csv
"""

import argparse
import os
import sys
import time
import csv
import json
from pathlib import Path

# add parent dir to path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
import editdistance


def load_labels(labels_path):
    """Load ground truth labels."""
    labels = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                labels[row[0]] = row[1]
    return labels


def run_paddle_ocr(image_paths, lang="en"):
    """Run PaddleOCR on images."""
    from src.paddle_ocr_runner import OCRWrapper

    wrapper = OCRWrapper(lang=lang, use_gpu=False)
    results = {}

    for path in image_paths:
        start = time.time()
        text = wrapper.get_full_text(path)
        elapsed = time.time() - start
        results[os.path.basename(path)] = {"text": text, "time": elapsed}

    return results


def run_trocr(image_paths, model_path="microsoft/trocr-base-handwritten"):
    """Run TrOCR on images."""
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    model.eval()

    results = {}

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        start = time.time()
        with torch.no_grad():
            generated = model.generate(pixel_values, max_length=128)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        elapsed = time.time() - start

        results[os.path.basename(path)] = {"text": text, "time": elapsed}

    return results


def compute_metrics(predictions, ground_truth):
    """Compute CER and word-level accuracy."""
    total_cer_dist = 0
    total_chars = 0
    total_word_correct = 0
    total_words = 0

    for filename, pred_text in predictions.items():
        if filename not in ground_truth:
            continue

        gt_text = ground_truth[filename]

        # character error rate
        total_cer_dist += editdistance.eval(pred_text, gt_text)
        total_chars += len(gt_text)

        # word accuracy
        pred_words = pred_text.split()
        gt_words = gt_text.split()
        for pw, gw in zip(pred_words, gt_words):
            if pw == gw:
                total_word_correct += 1
        total_words += max(len(gt_words), 1)

    cer = total_cer_dist / max(total_chars, 1)
    word_acc = total_word_correct / max(total_words, 1)

    return {"cer": cer, "word_accuracy": word_acc}


def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR engines")
    parser.add_argument("--dataset", type=str, required=True, help="Directory with test images")
    parser.add_argument("--labels", type=str, required=True, help="CSV with filename,text")
    parser.add_argument("--engines", nargs="+", default=["paddle", "trocr"],
                        help="Engines to benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    # load ground truth
    gt = load_labels(args.labels)
    print(f"Loaded {len(gt)} ground truth labels")

    # get image paths
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([
        os.path.join(args.dataset, f)
        for f in os.listdir(args.dataset)
        if Path(f).suffix.lower() in image_extensions
        and f in gt
    ])
    print(f"Found {len(image_paths)} images with labels")

    all_results = {}

    if "paddle" in args.engines:
        print("\n--- PaddleOCR ---")
        start = time.time()
        paddle_results = run_paddle_ocr(image_paths)
        total_time = time.time() - start

        paddle_preds = {k: v["text"] for k, v in paddle_results.items()}
        metrics = compute_metrics(paddle_preds, gt)
        print(f"CER: {metrics['cer']:.4f}")
        print(f"Word accuracy: {metrics['word_accuracy']:.4f}")
        print(f"Total time: {total_time:.2f}s ({total_time / len(image_paths):.3f}s/img)")

        all_results["paddle"] = {**metrics, "total_time": total_time}

    if "trocr" in args.engines:
        print("\n--- TrOCR ---")
        start = time.time()
        trocr_results = run_trocr(image_paths)
        total_time = time.time() - start

        trocr_preds = {k: v["text"] for k, v in trocr_results.items()}
        metrics = compute_metrics(trocr_preds, gt)
        print(f"CER: {metrics['cer']:.4f}")
        print(f"Word accuracy: {metrics['word_accuracy']:.4f}")
        print(f"Total time: {total_time:.2f}s ({total_time / len(image_paths):.3f}s/img)")

        all_results["trocr"] = {**metrics, "total_time": total_time}

    # save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
