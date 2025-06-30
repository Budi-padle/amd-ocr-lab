# OCR Pipeline — Sample Input & Expected Output

## Example 1: Printed Document Scan

**Input:** A scanned business letter (300 DPI, A4 size, JPEG)

```
Image description: A printed business letter on company letterhead.
Resolution: 2480 × 3508 pixels (A4 at 300 DPI)
Content: A formal letter with sender address, date, recipient, body paragraph, and signature.
```

**Expected OCR output (PaddleOCR):**
```
ACME CORPORATION
123 Business Avenue, Suite 400
San Francisco, CA 94105

June 15, 2025

Mr. John Smith
456 Oak Street
Portland, OR 97201

Dear Mr. Smith,

We are pleased to inform you that your application for the Senior Software
Engineer position has been approved. Your start date is scheduled for
July 1, 2025. Please find enclosed the employment contract for your review.

Sincerely,
Jane Williams
Director of Engineering
```

## Example 2: Handwritten Note (TrOCR)

**Input:** A photo of handwritten text on lined paper

```
Image description: A smartphone photo of handwritten notes on college-ruled paper.
Resolution: 1920 × 1080 pixels
Content: Meeting notes with bullet points and a to-do list.
```

**Expected TrOCR output:**
```
Meeting Notes - Tuesday
- Review Q2 targets with team
- Update project timeline
- Schedule 1:1 with Sarah
TODO: Send follow-up email by Friday
```

> Note: TrOCR is trained on IAM handwriting dataset. Accuracy drops with non-Latin scripts, heavy skew, or unusual handwriting styles.

## Example 3: Table Extraction

**Input:** A scanned invoice with a line-item table

```
Image description: A printed invoice with header info and a 5-row table.
Resolution: 2480 × 3508 pixels
Table columns: Item | Qty | Unit Price | Total
```

**Expected table extraction output:**

| Item | Qty | Unit Price | Total |
|------|-----|------------|-------|
| Widget A | 10 | $5.00 | $50.00 |
| Widget B | 5 | $12.00 | $60.00 |
| Gadget C | 2 | $25.00 | $50.00 |
| Service D | 1 | $150.00 | $150.00 |
| **Total** | | | **$310.00** |

## Example 4: Layout Analysis

**Input:** A multi-column academic paper page

```
Image description: A two-column academic paper with figures and equations.
```

**Expected layout detection output:**
```
Region 1: [text] (0, 0, 1240, 280) — Title and authors
Region 2: [text] (0, 290, 600, 1400) — Left column
Region 3: [text] (640, 290, 1240, 1400) — Right column
Region 4: [figure] (200, 1420, 1040, 1800) — Figure 1
Region 5: [caption] (200, 1810, 1040, 1900) — Figure 1 caption
```
