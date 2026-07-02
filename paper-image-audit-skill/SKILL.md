---
name: paper-image-audit-skill
description: Audit scientific paper PDFs or figure images for common PubPeer-style image-integrity concerns, including duplicated, transformed, reused, spliced, cloned, selectively enhanced, or relabeled images. Use when Codex needs to inspect manuscript figures, split figures into subpanels, detect Western blot/gel band reuse, screen non-WB panels for whole-panel reuse or transforms, triage microscopy/TEM/photo/flow/chart concerns, and produce a human-reviewable report with highlighted candidate regions and cautious evidence wording.
---

# Paper Image Audit

## Overview

Use this skill to triage paper figures for suspicious image reuse and related image-integrity concerns. Treat outputs as review candidates, not final conclusions; cite the figure, panel, page, score, highlighted region, and relevant figure context so a human can verify the evidence.

The bundled script is strongest for:

- WB/gel local band or lane reuse within extracted figures.
- Non-WB/gel whole-panel reuse, mirror, flip, or rotation candidates when `--compare-other-panels` is enabled.

Use the manual review rules for concerns that are not reliably automated here: local cloning, undeclared splicing, background patching, contrast over-adjustment, relabeling without pixel-level reuse, chart/plot reuse, cross-paper reuse, and possible synthetic or AI-generated image artifacts.

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

Run it on one or more already-extracted figure images:

```bash
python3 scripts/audit_paper_images.py /path/to/figure1.png /path/to/figure2.jpg --out /path/to/audit-output --compare-other-panels --panel-scope all-figures
```

For a targeted check, limit to a specific figure:

```bash
python3 scripts/audit_paper_images.py /path/to/manuscript.pdf --out /path/to/audit-output --figure 5
```

For a broader image-integrity pass, include non-WB/gel whole-panel comparison:

```bash
python3 scripts/audit_paper_images.py /path/to/manuscript.pdf --out /path/to/audit-output --compare-other-panels
```

Use `--panel-scope all-figures` only when deliberately looking for reuse across figures in the same manuscript; expect more false positives from charts, axes, and common layouts.

On macOS, force the legacy PDFKit path only when comparing backend differences:

```bash
python3 scripts/audit_paper_images.py /path/to/manuscript.pdf --out /path/to/audit-output --pdf-backend swift
```

Open `report.html` for visual review, `results.json` for structured evidence, and `manual_review_checklist.md` for the non-automated checks to perform before writing a finding.
OCR preprocessed images, TSV, parsed word boxes, and diagnostic overlays are saved under `ocr/` in the output directory.

## Workflow

1. Check Python/PDF/OCR dependencies with `scripts/install_dependencies.sh --check` on macOS/Linux or `scripts/install_dependencies.ps1 --check` on Windows.
2. For PDF input, extract PDF text/layout with PyMuPDF on Windows, macOS, and Linux; use Swift/PDFKit only as a macOS fallback.
3. For PDF input, render figure pages with PyMuPDF on Windows, macOS, and Linux; use Swift/PDFKit only as a macOS fallback.
4. For PDF input, crop figure regions using figure-title and caption coordinates; for image input, treat each input image as one figure after trimming blank border.
5. Segment figures into panels and run Tesseract OCR on each cropped figure image.
6. Prefer OCR-detected rasterized panel labels such as `A`, `B`, `C`; fall back to row-major labels when OCR is unavailable or uncertain.
7. Classify blot-like panels by detecting a large grayscale blot ROI.
8. Detect WB/gel protein rows from right-side OCR labels such as `p-TBK1`, `TBK1`, `cGAS`, `STING`, and loading controls; propagate consistent row labels across matched blot panels when OCR is incomplete.
9. Extract both strip-level and lane-local band-patch candidates from blot panels, require minimum evidence-patch size, and use high-confidence OCR text boxes to suppress text-like false positives.
10. Compare WB/gel candidates only within the same figure/category and same protein row by normalized cross-correlation, while filtering undersized or mismatched patch pairs that are prone to false positives.
11. Aggregate same-row local evidence into row-level clusters when multiple independent local matches share consistent lane offset, orientation, and surrounding-context support.
12. Write portable multimodal review tasks under `multimodal/` so Codex, OpenClaw, or any vision-capable agent/model can inspect aggregate evidence images without requiring the audit script itself to call a model API.
13. Optionally merge external multimodal review JSON back into the report with `--multimodal-review-json`.
14. When `--compare-other-panels` is enabled, compare non-WB/gel panels for whole-panel reuse after contrast normalization and transform search (`none`, mirror, vertical flip, 180/90/270 rotations).
15. Review `report.html`, `report.md`, `results.json`, `manual_review_checklist.md`, OCR overlays, panels, strips, aggregate images, and source figures against `references/review-rules.md` before writing a finding.
16. Generate cautious report language that distinguishes WB/gel local reuse candidates, aggregate support, non-WB whole-panel candidates, multimodal review status, and manually observed cloning/splicing/enhancement/relabeling concerns.

## Installation Dependencies

Use `scripts/install_dependencies.sh` or `scripts/install_dependencies.ps1` to make the skill environment reproducible across Windows, macOS, and Linux:

- `pymupdf`: cross-platform PDF text/layout extraction and page rendering.
- `pillow` and `numpy`: image processing.
- `tesseract`: OCR engine.
- `tesseract-lang`: additional OCR language data, including `chi_sim`, `chi_tra`, `chi_sim_vert`, and `chi_tra_vert`.
- macOS automatic Tesseract install uses Homebrew; Linux uses common package managers when available; Windows uses `winget` or Chocolatey when available.

## Script Notes

- Use PyMuPDF as the default PDF backend on Windows, macOS, and Linux. Set `CLANG_MODULE_CACHE_PATH` only when using the macOS Swift/PDFKit fallback and Swift cannot write its module cache.
- The script accepts either one PDF or one or more image files (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`). Do not mix PDF and image inputs in one run.
- Run the platform installer when `pymupdf`, `tesseract`, `chi_sim`, or `chi_tra` is missing.
- Use the bundled Python runtime when the system Python lacks Pillow/NumPy.
- Use `--min-score` to adjust sensitivity. Start with `0.82` for routine triage; lower to `0.70` when exploring faint or heavily compressed bands.
- Keep the default `--min-patch-area 450`, `--min-patch-width 18`, `--min-patch-height 12`, and `--min-area-ratio 0.55` for routine WB/gel audits. Raise `--min-patch-area` to suppress tiny-band false positives; lower it only for exploratory checks of very small bands.
- Keep the default protein-row matching for WB/gel audits. Use `--allow-row-mismatch` only for exploratory debugging because cross-protein comparisons can surface high-scoring false positives.
- Use `--min-context-score` only as a stricter second-pass filter. The report always records context score, but the default is `0.0` because cropped WB bands can have valid local reuse even when surrounding lanes differ.
- Keep evidence aggregation enabled with the default `--min-aggregate-matches 2`, `--min-aggregate-context-score 0.55`, and `--min-aggregate-orientation-fraction 0.80`. These defaults require at least two one-to-one local matches with consistent orientation and enough surrounding context before promoting pairwise candidates into a row-level evidence cluster.
- Use the generated `multimodal/multimodal_review.md` or `multimodal/multimodal_review.json` for second-stage visual review with Codex, OpenClaw, or another vision-capable agent. The script does not need to call the model API itself.
- If an external agent returns review JSON, rerun or post-process with `--multimodal-review-json /path/to/review.json` to merge the status, confidence, and rationale into `report.html`, `report.md`, and `results.json`.
- Keep multimodal review local to the current agent/session for confidential manuscripts unless the user explicitly approves sending images to an external or hosted model.
- Use `--compare-other-panels` for microscopy, TEM, histology, colony plates, animal/gross photos, and other raster panels where whole-panel duplication or transformed reuse is plausible. Keep the default `--min-panel-score 0.92` for routine triage; lower it only for exploratory review.
- Keep `--panel-scope same-figure` unless the task explicitly asks for cross-figure reuse within the manuscript. Use `--panel-scope all-figures` as a second pass and inspect charts/axes especially carefully.
- Use the default `--dpi 180` for cross-platform PyMuPDF audits. Lower values such as `--dpi 150` reduce runtime but can lower small band similarity scores.
- Use `--keep-existing` to reuse PDF layout and rendered pages while tuning segmentation/comparison.
- Inspect `ocr/*_overlay.png` when panel labels look wrong or when OCR text-filtered strip counts are unexpectedly high.

## Review Rules

Read `references/review-rules.md` before making a written assessment. It contains the anomaly taxonomy, category-specific checks, false-positive rules, and report templates. Always describe findings as suspicious candidates unless the user explicitly asks for a stronger forensic conclusion and the evidence supports it.

## Known Limits

- Panel labels fall back to row-major order when rasterized labels are missing, too stylized, or not confidently recognized by OCR.
- OCR text filtering is conservative and only removes strip candidates that are mostly covered by high-confidence text boxes.
- Protein-row labels are OCR-assisted and may be propagated across panels when the row index is consistent; inspect row labels and highlighted context before making a finding.
- Very small local band patches are filtered by default because resizing tiny blobs can create artificially high correlation scores.
- Figure discovery still depends on the PDF text layer. For scanned PDFs on any platform, add page-level OCR title/caption detection before relying on this workflow.
- Multiple image inputs are treated as separate figures. WB/gel strip comparison remains within each figure; use whole-panel `--panel-scope all-figures` for cross-image/cross-figure raster-panel reuse screening.
- Automation is strongest for WB/gel local reuse and non-WB whole-panel duplication/transform triage. Local cloning, splice boundaries, background patching, contrast manipulation, flow-gate relabeling, and chart-data reuse require manual or vision-assisted review.
- Whole-panel comparison can flag charts, axes, legends, or common layouts. Treat those as false-positive-prone until the underlying raster content also matches.
- Similar band-shaped biological signals can look alike. High scores require visual review of the highlighted context, lane identity, and caption claims.
- Evidence aggregates are row-level support summaries, not claims that a whole WB row is pixel-identical. The report records `full_row_score` as a diagnostic so local band reuse can be distinguished from full-row duplication.
