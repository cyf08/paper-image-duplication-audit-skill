# Paper Image Audit Skill

This repository packages the `paper-image-audit-skill` Codex skill.

## Install

Copy the skill folder into a Codex skill root:

```bash
cp -R paper-image-audit-skill ~/.codex/skills/
```

Then check dependencies.

macOS/Linux:

```bash
~/.codex/skills/paper-image-audit-skill/scripts/install_dependencies.sh --check
```

Windows PowerShell:

```powershell
~\.codex\skills\paper-image-audit-skill\scripts\install_dependencies.ps1 --check
```

Install dependencies when needed.

macOS/Linux:

```bash
~/.codex/skills/paper-image-audit-skill/scripts/install_dependencies.sh --install
```

Windows PowerShell:

```powershell
~\.codex\skills\paper-image-audit-skill\scripts\install_dependencies.ps1 --install
```

## Platform Support

- Windows, macOS, and Linux use PyMuPDF as the default PDF backend.
- macOS can fall back to the bundled Swift/PDFKit helpers if PyMuPDF is unavailable, or when `--pdf-backend swift` is set.
- Tesseract OCR is required for OCR-assisted panel labels and text filtering.

## WB/Gel False-Positive Controls

The audit script filters tiny WB/gel patch candidates by default because very small dark blobs can become artificially similar after resizing. Routine audits use:

```bash
python3 paper-image-audit-skill/scripts/audit_paper_images.py manuscript.pdf \
  --out audit-output \
  --min-patch-area 450 \
  --min-patch-width 18 \
  --min-patch-height 12 \
  --min-area-ratio 0.55
```

Raise `--min-patch-area` for a stricter second pass, or lower it only for exploratory checks of very small bands. `report.html` and `results.json` include evidence area, area ratio, context score, and skipped-pair counts for manual review.

## Package

The same skill is also available as a tarball under `dist/`.

## Contents

- `paper-image-audit-skill/SKILL.md`
- `paper-image-audit-skill/agents/openai.yaml`
- `paper-image-audit-skill/references/review-rules.md`
- `paper-image-audit-skill/scripts/`
- `paper-image-audit-skill/requirements.txt`
