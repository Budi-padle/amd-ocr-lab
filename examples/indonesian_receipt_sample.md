# Example: Parsing an Indonesian Receipt

## Input

Indomaret receipt (thermal print, slightly faded, phone scan):

```
INDOMARET
Jl. Sudirman No. 123
Jakarta Selatan

Telur Ayam 1kg      Rp 28.000
Indomie Goreng 5pcs Rp 13.500
Aqua 600ml          Rp  3.500
Rinso 770g          Rp 18.900
                        -------
Sub Total           Rp 63.900
PPN 10%             Rp  6.390
TOTAL               Rp 70.290

Tunai               Rp100.000
Kembali             Rp 29.710

15/06/2026 14:32
Kasir: Andi
```

## Command

```bash
python ocr_lab.py receipt scan indomaret_receipt.jpg --output receipt_parsed.json
```

## Output

```json
{
  "store": "INDOMARET",
  "address": "Jl. Sudirman No. 123, Jakarta Selatan",
  "date": "2026-06-15",
  "time": "14:32",
  "cashier": "Andi",
  "items": [
    {"name": "Telur Ayam 1kg", "price": 28000},
    {"name": "Indomie Goreng 5pcs", "price": 13500},
    {"name": "Aqua 600ml", "price": 3500},
    {"name": "Rinso 770g", "price": 18900}
  ],
  "subtotal": 63900,
  "tax": 6390,
  "total": 70290,
  "payment": 100000,
  "change": 29710
}
```

## What worked

- Item names extracted correctly (including brand names)
- Prices parsed with proper thousands separator handling (Indonesian uses dots, not commas)
- Date/time extracted from footer

## What didn't work

- Faded thermal print: some characters misread (e.g., "1" read as "l")
- Receipt from different store (Alfamart): different layout, parser failed
- Very long receipts (> 20 items): table alignment drifted
