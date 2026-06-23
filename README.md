# amd-ocr-lab

This repo is a small OCR playground focused on receipts, forms, and table-like documents. I'm especially interested in Indonesian-style receipts and scanned forms where spacing, low resolution, and mixed language text make OCR annoying.

## The problem I'm solving

Indonesian receipts are messy. They have:
- Thermal print that fades quickly
- Mixed Indonesian/English product names
- Tables with no grid lines (just aligned text)
- Low resolution from phone camera scans

Standard OCR tools give you raw text, but lose the structure. You get a wall of text instead of line items with prices.

## What I'm working on

1. **Receipt parsing**: Extract store name, date, items, total
2. **Table extraction**: Detect and extract table rows from scanned documents
3. **Indonesian text handling**: Mixed language, abbreviations (Indomie = Indofood Mi Instan)

## Current state

- Basic OCR with Tesseract and TrOCR
- Receipt template matching (works for common Indonesian receipt formats)
- Table detection: simple column alignment heuristic

## What I'm NOT building

- Enterprise document processing
- Multi-language OCR for 50 languages
- Cloud API integration

This is for my own receipts and forms. If it works for mine, it probably works for similar Indonesian documents.

## Quick start

```bash
pip install -r requirements.txt
python ocr_lab.py receipt scan receipt.jpg --output parsed.json
```

## Examples

- `examples/indonesian_receipt_sample.md` — parsing an Indomaret receipt
- `examples/form_table_output.md` — extracting a table from a scanned form


## Troubleshooting
**Q: Getting OOM errors?**
A: Reduce batch size or enable gradient checkpointing.

## Hardware Tested
- AMD RX 7800 XT (RDNA3)
- AMD RX 7900 XTX (RDNA3)