# amd-ocr-lab

Experiments with OCR and document AI stuff running on my RX 7800 XT. ROCm is... well, it works now, but it took a while to get here lol.

## What's this

I've been messing around with different OCR models trying to get them running on AMD GPUs instead of the usual NVIDIA path. Mostly TrOCR and PaddleOCR, plus some table extraction experiments.

Things in here:
- **TrOCR fine-tuning** — the HuggingFace one, got it training on ROCm with some hacks
- **PaddleOCR wrapper** — surprisingly good out of the box, wanted to wrap it nicely
- **Table extraction** — TableNet-style approach for pulling tables out of scanned docs
- **Document layout analysis** — using YOLOv8 to detect regions (text, tables, figures, etc)
- **Handwritten text** — fine-tuned TrOCR on IAM dataset, results are... ok-ish

## Setup

You need ROCm installed (I'm on 6.1.2). Then:

```bash
pip install -r requirements.txt
```

Fair warning: getting PyTorch to work with ROCm is its own adventure. I had to build from source once when the pre-built wheels didn't work with my kernel version. Check the experiments/notes.md for my pain.

## Usage

Train TrOCR:
```bash
python src/trocr_finetune.py --config configs/trocr_finetune.yaml
```

Run PaddleOCR inference:
```bash
python src/paddle_ocr_runner.py --image path/to/scan.jpg
```

Benchmark different engines:
```bash
python scripts/benchmark_ocr.py --dataset path/to/test/images
```

## Hardware

- AMD RX 7800 XT (16GB VRAM)
- ROCm 6.1.2
- Ubuntu 24.04

## Status

TrOCR fine-tuning works, getting ~92% character accuracy on IAM after 3 epochs. PaddleOCR is faster but less customizable. Table extraction is still WIP, the model converges but results are messy on real-world scans.

Multi-language support is next — want to try Chinese and Japanese OCR.

## Why AMD / ROCm

Document AI and OCR pipelines process millions of pages daily in enterprise settings. Running these workloads on AMD GPUs using ROCm offers a viable alternative to the NVIDIA-dominated stack, especially for batch processing where cost-per-page matters more than raw latency. Target hardware:

- AMD MI300X / MI250X for datacenter OCR processing
- RX 7900 XTX / RX 7800 XT for development and prototyping
- ROCm 6.x with PyTorch ROCm builds
- OCR workloads are compute-bound on the recognition models — GPU acceleration provides the biggest gains here
- PaddleOCR has existing ROCm-compatible builds, making it a natural fit for AMD hardware

## AMD GPU Credit Use Plan

If granted AMD GPU access, I plan to:

1. Validate the full OCR pipeline (detection → recognition → table extraction) on ROCm-compatible GPUs
2. Compare CPU vs AMD GPU inference latency for TrOCR and PaddleOCR across different document types
3. Test fp16 vs fp32 accuracy on OCR output — character accuracy regression is the key metric
4. Profile VRAM usage for batch processing (pages 1–16) to find optimal batch sizes
5. Document ROCm-specific setup issues, workarounds, and PaddlePaddle ROCm integration notes
6. Publish benchmark results and ROCm compatibility matrix back to this repository

## License

MIT, do whatever you want with it.
