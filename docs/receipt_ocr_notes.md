# Receipt OCR Notes

## Indonesian receipt characteristics

### Common formats
1. **Minimarket** (Indomaret, Alfamart): Thermal print, table format, store header
2. **Traditional market**: Handwritten or printed, often no table lines
3. **Restaurant**: Printed, sometimes with item codes
4. **Online**: Digital receipt, PDF or image, cleaner format

### Challenges
- **Thermal print fading**: Older receipts have low contrast
- **Thousands separator**: Indonesia uses dots (Rp 28.000), not commas
- **Mixed language**: Product names can be Indonesian, English, or brand names
- **Abbreviations**: "Indomie", "Aqua", "Rinso" are brand names, not dictionary words
- **No grid lines**: Tables are just aligned text, no borders

## OCR engine comparison

| Engine | Accuracy | Speed | Notes |
|--------|----------|-------|-------|
| Tesseract | Good | Fast | Best for clean prints |
| TrOCR | Better | Slower | Better for faded text |
| PaddleOCR | Good | Medium | Best for Chinese/Japanese (not needed here) |

## Pre-processing tricks

1. **Contrast enhancement**: `ImageEnhance.Contrast(img).enhance(2.0)` for faded receipts
2. **Grayscale**: Always convert to grayscale first
3. **Deskew**: Receipts are often scanned at an angle
4. **Binarization**: Otsu's threshold for clean separation

## Post-processing

1. **Price extraction**: Regex for `Rp [0-9.]+` patterns
2. **Date extraction**: Common formats: DD/MM/YYYY, DD-MM-YYYY, DD Month YYYY
3. **Item-price alignment**: Match items with prices by vertical position
