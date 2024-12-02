# Experiment Notes

## 2024-11-15 — Initial setup

Finally got ROCm working on the 7800 XT. Key things:
- Had to set `HSA_OVERRIDE_GFX_VERSION=11.0.0` (RDNA3 gfx1101 isn't in the default supported list, but 11.0.0 works)
- MIOpen takes forever on first run because it's tuning kernels. Be patient.
- PyTorch 2.1 with ROCm 6.0 wheels worked. Didn't need to build from source.

## 2024-11-18 — TrOCR first attempt

TrOCR base model loads fine. Inference works. Started fine-tuning on IAM.

First issue: the HuggingFace `datasets` library was trying to use some CUDA-specific feature and crashed. Had to downgrade datasets from 2.17 to 2.16. Annoying.

Training at batch_size=8, using about 10GB VRAM. Plenty of headroom on the 16GB card.

## 2024-11-20 — PaddleOCR

PaddleOCR is really easy to set up. Just `pip install paddleocr` and it works. The default models are surprisingly good for printed text.

Problem: no ROCm build for PaddlePaddle. Using CPU mode for now. It's still fast enough for inference — about 0.3s per image on my Ryzen 7600X.

For training custom Paddle models you'd need GPU support though. Might try building paddle-rocm from source later but that sounds painful.

## 2024-11-22 — TrOCR results

After 3 epochs on IAM:
- CER: 0.081 (91.9% character accuracy)
- Not bad for a base model with minimal tuning

Some observations:
- The model struggles with really messy handwriting (expected)
- Capital letters are sometimes confused
- Running with mixed precision didn't work out of the box — got NaN losses. Disabling fp16 for now.

## 2024-11-25 — Table extraction

Started working on TableNet. The architecture is straightforward (VGG encoder + dual decoders), but training data is the issue.

PubTables-1M is huge (~500GB). Downloaded a subset. Training on the small subset for now, results are mediocre. The model finds tables but the boundaries are sloppy.

Tried post-processing with morphological operations — helped a bit.

## 2024-11-28 — Layout analysis

YOLOv8 is dead simple to use. Fine-tuned on DocLayNet format. The ultralytics package works on ROCm without issues since it's just using PyTorch underneath.

Current mAP@0.5: 0.78 on my validation set. Not state of the art but usable.

Classes I'm detecting: text, title, table, figure, caption, header, footer, section header.

## 2024-12-01 — Benchmarks

Quick benchmark on 50 test images:
- PaddleOCR: 0.12 CER, 0.28s/image (CPU)
- TrOCR (fine-tuned): 0.08 CER, 0.05s/image (GPU)
- TrOCR (base): 0.15 CER, 0.04s/image (GPU)

TrOCR is faster on GPU and more accurate when fine-tuned. PaddleOCR is better out of the box though.

## TODO

- [ ] Try Chinese OCR with PaddleOCR
- [ ] Build PaddlePaddle with ROCm support
- [ ] Improve table extraction (need more training data?)
- [ ] Add handwriting synthesis for data augmentation
- [ ] Try Donut model for end-to-end document understanding
