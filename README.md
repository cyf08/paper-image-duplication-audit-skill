# Paper Image Duplication Audit Skill

This repository packages the `paper-image-duplication-audit` Codex skill.

## Install

Copy the skill folder into a Codex skill root:

```bash
cp -R paper-image-duplication-audit ~/.codex/skills/
```

Then check dependencies.

macOS/Linux:

```bash
~/.codex/skills/paper-image-duplication-audit/scripts/install_dependencies.sh --check
```

Windows PowerShell:

```powershell
~\.codex\skills\paper-image-duplication-audit\scripts\install_dependencies.ps1 --check
```

Install dependencies when needed.

macOS/Linux:

```bash
~/.codex/skills/paper-image-duplication-audit/scripts/install_dependencies.sh --install
```

Windows PowerShell:

```powershell
~\.codex\skills\paper-image-duplication-audit\scripts\install_dependencies.ps1 --install
```

## Platform Support

- Windows, macOS, and Linux use PyMuPDF as the default PDF backend.
- macOS can fall back to the bundled Swift/PDFKit helpers if PyMuPDF is unavailable, or when `--pdf-backend swift` is set.
- Tesseract OCR is required for OCR-assisted panel labels and text filtering.

## Package

The same skill is also available as a tarball under `dist/`.

## Contents

- `paper-image-duplication-audit/SKILL.md`
- `paper-image-duplication-audit/agents/openai.yaml`
- `paper-image-duplication-audit/references/review-rules.md`
- `paper-image-duplication-audit/scripts/`
- `paper-image-duplication-audit/requirements.txt`
