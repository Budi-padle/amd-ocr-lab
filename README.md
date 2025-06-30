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

Primary test target: AMD RX 7800 XT (ROCm 6.1.2). Some tests also run on CPU for baseline comparison.

## Status

TrOCR fine-tuning works, getting ~92% character accuracy on IAM after 3 epochs. PaddleOCR is faster but less customizable. Table extraction is still WIP, the model converges but results are messy on real-world scans.

Multi-language support is next — want to try Chinese and Japanese OCR.

## License

MIT, do whatever you want with it.
