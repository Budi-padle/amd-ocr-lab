# ROCm Notes — Document AI / OCR Pipeline

## Target Environment

- **ROCm version:** 6.1.x (primary), 6.0 (fallback)
- **PyTorch:** torch 2.3+ with ROCm backend (`--index-url https://download.pytorch.org/whl/rocm6.0`)
- **GPU target:** Single GPU first (RX 7800 XT), multi-GPU later
- **OS:** Ubuntu 24.04, kernel 6.5+

## Current Status

### What Works
- PaddleOCR inference via PaddlePaddle's ROCm builds — runs out of the box on gfx1100
- TrOCR fine-tuning with HuggingFace Transformers + PyTorch ROCm
- Table extraction model loads and runs (correctness still WIP)

### Known Blockers
- Some custom CUDA ops in older PaddleOCR versions fall back to CPU — need to profile which layers hit the fallback path
- `torchvision::nms` used by the layout analyzer (YOLOv8) has intermittent issues on ROCm 6.0; works better on 6.1.2
- Mixed-precision training with `amp` has occasional NaN losses on first epoch — likely a ROCm-specific matmul precision issue
- No ROCm-compatible version of EasyOCR — TrOCR or PaddleOCR are the viable alternatives

### Workarounds Applied
- Set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` for better VRAM management
- Force `torch.backends.cudnn.enabled = False` — avoids miopen conv issues on some kernels
- Use `torch.compile(dynamic=True)` sparingly — it works but compilation time is 3-4x slower than CUDA inductor

## Planned Benchmarks

| Test | Target Metric | Notes |
|------|--------------|-------|
| TrOCR fp32 inference | latency/page | Baseline on CPU vs GPU |
| TrOCR fp16 inference | latency/page | Expected ~1.8x speedup |
| PaddleOCR batch inference | throughput | 1, 4, 8, 16 pages |
| Table extraction | latency/table | Single-table and multi-table |
| Layout analysis (YOLOv8) | FPS | Image preprocessing only |
| VRAM usage across models | peak GB | fp32 vs fp16 comparison |

## ROCm-Specific Technical Notes

- PaddleOCR's `paddlepaddle-rocm` package maps to ROCm 5.x out of the box. For ROCm 6.x, you may need to set `FLAGS_selected_gpus=0` and verify `hipDeviceGetAttribute` calls don't crash.
- TrOCR's encoder-decoder attention benefits from ROCm's flash-attention variant (`flash_attn` built for ROCm). Install via `pip install flash-attn --no-build-isolation` with `ROCm` target.
- The table extraction model uses deformable convolutions which aren't natively supported in miopen — expect CPU fallback for that layer until a custom HIP kernel is available.

## Validation Checklist

- [ ] Run TrOCR inference on 100 sample images, compare output to CPU baseline
- [ ] Verify PaddleOCR detection + recognition pipeline end-to-end on GPU
- [ ] Measure peak VRAM on a 16GB card with batch sizes 1–16
- [ ] Test fp16 via `torch.autocast("cuda")` (maps to hip on ROCm)
- [ ] Check that no operations silently fall back to CPU without warning
- [ ] Run for 1+ hour to check for memory leaks or thermal throttling
