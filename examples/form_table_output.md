# Example: Table Extraction from Scanned Form

## Input

A scanned government form with a table:

```
No  Nama                Alamat              No. Telp
1   Budi Santoso        Jl. Merdeka 5       08123456789
2   Siti Rahayu         Jl. Sudirman 12     08567890123
3   Ahmad Fauzi         Jl. Gatot Subroto 8 08789012345
```

## Command

```bash
python ocr_lab.py table scan form_table.jpg --output table.json
```

## Output

```json
{
  "headers": ["No", "Nama", "Alamat", "No. Telp"],
  "rows": [
    ["1", "Budi Santoso", "Jl. Merdeka 5", "08123456789"],
    ["2", "Siti Rahayu", "Jl. Sudirman 12", "08567890123"],
    ["3", "Ahmad Fauzi", "Jl. Gatot Subroto 8", "08789012345"]
  ]
}
```

## How table detection works

1. Run OCR with bounding boxes (Tesseract `--psm 6`)
2. Group text by vertical position (rows)
3. Within each row, detect column boundaries by whitespace gaps
4. Match columns across rows to build table structure

## Known issues

- Forms with merged cells: detected as separate columns
- Handwritten forms: OCR quality too low for reliable extraction
- Rotated forms: need deskewing first (not implemented yet)
