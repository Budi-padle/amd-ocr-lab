# CPU Baseline Benchmarks — Document AI / OCR Pipeline

> **Note:** AMD GPU benchmark is pending — I currently do not have access to ROCm-capable hardware.
> These CPU numbers serve as a reference point for future GPU comparisons.

## Test Environment (CPU)

- **CPU:** AMD Ryzen 9 7950X (16C/32T)
- **RAM:** 64GB DDR5-5600
- **OS:** Ubuntu 24.04
- **PyTorch:** 2.3.0 (CPU-only)
- **Python:** 3.10

## TrOCR Inference

| Metric | Value |
|--------|-------|
| Model | `microsoft/trocr-large-handwritten` |
| Precision | fp32 |
| Single image (224×224) | 1.85s avg |
| Batch of 4 | 6.92s (1.73s/image) |
| Batch of 8 | 12.6s (1.58s/image) |
| Peak RAM | 4.2 GB |
| Batch 8 RAM | 5.8 GB |

## PaddleOCR Inference

| Metric | Value |
|--------|-------|
| Engine | PaddleOCR (CPU mode) |
| Detection + Recognition | 2.4s per page (single) |
| Detection only | 0.9s per page |
| Recognition only (cropped text) | 0.3s per line |
| Batch of 4 pages | 8.1s (2.0s/page) |
| Peak RAM | 2.1 GB |

## Table Extraction (TableNet)

| Metric | Value |
|--------|-------|
| Single table region | 3.8s |
| Full page (with table detection) | 5.2s |
| Peak RAM | 3.4 GB |

## Layout Analysis (YOLOv8)

| Metric | Value |
|--------|-------|
| Single A4 scan (300 DPI) | 1.1s |
| Batch of 4 | 3.9s (0.98s/image) |
| Peak RAM | 1.8 GB |

## Summary

On CPU, OCR inference is dominated by the recognition models (TrOCR, PaddleOCR recognition). The detection and layout stages are relatively fast. For a typical document processing pipeline (layout → detect → recognize), expect **4–7 seconds per page** on CPU.

GPU acceleration is expected to provide:
- 5–10x speedup on TrOCR inference (encoder-decoder is very parallelizable)
- 3–5x speedup on PaddleOCR recognition
- Minimal speedup on table extraction (currently CPU-bound preprocessing)
