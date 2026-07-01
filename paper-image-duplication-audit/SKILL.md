---
name: paper-image-duplication-audit
description: Audit scientific paper PDFs or figure images for suspicious duplicated, reused, spliced, or relabeled images. Use when Codex needs to inspect manuscript figures, split figures into subpanels, compare same-category panels, detect Western blot/gel band reuse, microscopy reuse, or other image duplication concerns, and produce a human-reviewable report with highlighted candidate regions.
---

# Paper Image Duplication Audit

## Overview

Use this skill to triage paper figures for suspicious image reuse. Treat outputs as review candidates, not final conclusions; cite the figure, panel, page, score, and highlighted region so a human can verify the evidence.

## Quick Start

Check or install dependencies before running OCR-enhanced audits.

macOS/Linux:

```bash
scripts/install_dependencies.sh --check
scripts/install_dependencies.sh --install
```

Windows PowerShell:

```powershell
.\scripts\install_dependencies.ps1 --check
.\scripts\install_dependencies.ps1 --install
```

Run the bundled audit script on a manuscript PDF:

```bash
python3 scripts/audit_paper_images.py /path/to/manuscript.pdf --out /path/to/audit-output
```

For a targeted check, limit to a specific figure:

```bash
python3 scripts/audit_paper_images.py /path/to/manuscript.pdf --out /path/to/audit-output --figure 5
```

On macOS, force the legacy PDFKit path only when comparing backend differences:

```bash
python3 scripts/audit_paper_images.py /path/to/manuscript.pdf --out /path/to/audit-output --pdf-backend swift
```

Open `report.html` for visual review and `results.json` for structured evidence.
OCR preprocessed images, TSV, parsed word boxes, and diagnostic overlays are saved under `ocr/` in the output directory.

## Workflow

1. Check Python/PDF/OCR dependencies with `scripts/install_dependencies.sh --check` on macOS/Linux or `scripts/install_dependencies.ps1 --check` on Windows.
2. Extract PDF text/layout with PyMuPDF on Windows, macOS, and Linux; use Swift/PDFKit only as a macOS fallback.
3. Render figure pages with PyMuPDF on Windows, macOS, and Linux; use Swift/PDFKit only as a macOS fallback.
4. Crop figure regions using figure-title and caption coordinates.
5. Segment figures into panels and run Tesseract OCR on each cropped figure image.
6. Prefer OCR-detected rasterized panel labels such as `A`, `B`, `C`; fall back to row-major labels when OCR is unavailable or uncertain.
7. Classify blot-like panels by detecting a large grayscale blot ROI.
8. Extract both strip-level and local band-patch candidates from blot panels, require minimum evidence-patch size, and use high-confidence OCR text boxes to suppress text-like false positives.
9. Compare candidates only within the same figure/category by normalized cross-correlation, while filtering undersized or mismatched patch pairs that are prone to false positives.
10. Generate `report.html`, `report.md`, and `results.json` with side-by-side highlighted evidence, evidence area, area ratio, and optional context score.

## Installation Dependencies

Use `scripts/install_dependencies.sh` or `scripts/install_dependencies.ps1` to make the skill environment reproducible across Windows, macOS, and Linux:

- `pymupdf`: cross-platform PDF text/layout extraction and page rendering.
- `pillow` and `numpy`: image processing.
- `tesseract`: OCR engine.
- `tesseract-lang`: additional OCR language data, including `chi_sim`, `chi_tra`, `chi_sim_vert`, and `chi_tra_vert`.
- macOS automatic Tesseract install uses Homebrew; Linux uses common package managers when available; Windows uses `winget` or Chocolatey when available.

## Script Notes

- Use PyMuPDF as the default PDF backend on Windows, macOS, and Linux. Set `CLANG_MODULE_CACHE_PATH` only when using the macOS Swift/PDFKit fallback and Swift cannot write its module cache.
- Run the platform installer when `pymupdf`, `tesseract`, `chi_sim`, or `chi_tra` is missing.
- Use the bundled Python runtime when the system Python lacks Pillow/NumPy.
- Use `--min-score` to adjust sensitivity. Start with `0.82` for routine triage; lower to `0.70` when exploring faint or heavily compressed bands.
- Keep the default `--min-patch-area 450`, `--min-patch-width 18`, `--min-patch-height 12`, and `--min-area-ratio 0.55` for routine WB/gel audits. Raise `--min-patch-area` to suppress tiny-band false positives; lower it only for exploratory checks of very small bands.
- Use `--min-context-score` only as a stricter second-pass filter. The report always records context score, but the default is `0.0` because cropped WB bands can have valid local reuse even when surrounding lanes differ.
- Use the default `--dpi 180` for cross-platform PyMuPDF audits. Lower values such as `--dpi 150` reduce runtime but can lower small band similarity scores.
- Use `--keep-existing` to reuse PDF layout and rendered pages while tuning segmentation/comparison.
- Inspect `ocr/*_overlay.png` when panel labels look wrong or when OCR text-filtered strip counts are unexpectedly high.

## Review Rules

Read `references/review-rules.md` before making a written assessment. Always describe findings as suspicious candidates unless the user explicitly asks for a stronger forensic conclusion and the evidence supports it.

## Known Limits

- Panel labels fall back to row-major order when rasterized labels are missing, too stylized, or not confidently recognized by OCR.
- OCR text filtering is conservative and only removes strip candidates that are mostly covered by high-confidence text boxes.
- Very small local band patches are filtered by default because resizing tiny blobs can create artificially high correlation scores.
- Figure discovery still depends on the PDF text layer. For scanned PDFs on any platform, add page-level OCR title/caption detection before relying on this workflow.
- The first version focuses on Western blot/gel reuse. Microscopy and flow cytometry support should be extended with category-specific extractors before relying on those classes.
- Similar band-shaped biological signals can look alike. High scores require visual review of the highlighted context, lane identity, and caption claims.
