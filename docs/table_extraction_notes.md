# Table Extraction Notes

## Approach

Tables in scanned documents don't have explicit structure (no HTML tables, no CSV). You have to infer structure from visual layout.

## Method 1: Whitespace column detection

1. Run OCR with bounding boxes
2. For each row, find gaps between text blocks
3. Align gaps across rows to detect column boundaries

Pros: Simple, works for well-aligned tables
Cons: Fails on messy alignment, merged cells

## Method 2: Line detection

1. Detect horizontal and vertical lines using Hough transform
2. Find intersections to identify cell boundaries
3. Extract text within each cell

Pros: Works for tables with visible grid lines
Cons: Fails on tables without lines (most Indonesian forms)

## Method 3: ML-based

Use a table detection model (e.g., TableTransformer):
1. Detect table region in the image
2. Detect rows and columns
3. Extract text from each cell

Pros: Most robust
Cons: Needs GPU, more complex setup

## Current implementation

Using Method 1 (whitespace column detection). It works for:
- Clean printed tables with consistent alignment
- Forms with clear column spacing

It fails on:
- Handwritten tables
- Tables with merged cells
- Tables with inconsistent spacing

## Next steps

- Try Method 3 (ML-based) for more robust detection
- Add support for merged cells (complex but important for Indonesian forms)
- Test on more varied receipt formats
