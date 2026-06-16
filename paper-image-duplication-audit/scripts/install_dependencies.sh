#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/install_dependencies.sh --check
  scripts/install_dependencies.sh --install

Checks or installs dependencies for paper-image-duplication-audit on macOS and Linux:
  - Python packages: pymupdf, pillow, numpy
  - Tesseract OCR
  - Tesseract English and Chinese language data: eng, chi_sim, chi_tra

For Windows, use scripts/install_dependencies.ps1.
USAGE
}

mode="${1:---check}"
case "$mode" in
  --check|--install|-h|--help) ;;
  *) usage; exit 2 ;;
esac

if [[ "$mode" == "-h" || "$mode" == "--help" ]]; then
  usage
  exit 0
fi

python_bin="${PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python_bin=python3
  elif command -v python >/dev/null 2>&1; then
    python_bin=python
  else
    echo "Missing dependency: python" >&2
    exit 1
  fi
fi

check_python_packages() {
  "$python_bin" - <<'PY'
import importlib.util
import sys

missing = []
for module, package in (("fitz", "pymupdf"), ("PIL", "pillow"), ("numpy", "numpy")):
    if importlib.util.find_spec(module) is None:
        missing.append(package)

if missing:
    print("Missing Python packages: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PY
}

install_python_packages() {
  "$python_bin" -m pip install --upgrade pymupdf pillow numpy
}

install_tesseract_macos() {
  local brew_bin
  brew_bin="$(command -v brew || true)"
  if [[ -z "$brew_bin" && -x /opt/homebrew/bin/brew ]]; then
    brew_bin=/opt/homebrew/bin/brew
  elif [[ -z "$brew_bin" && -x /usr/local/bin/brew ]]; then
    brew_bin=/usr/local/bin/brew
  fi
  if [[ -z "$brew_bin" ]]; then
    echo "Homebrew is required for automatic Tesseract installation on macOS." >&2
    echo "Install Homebrew or install tesseract and tesseract-lang manually." >&2
    return 1
  fi
  "$brew_bin" install tesseract
  "$brew_bin" install tesseract-lang
}

install_tesseract_linux() {
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim tesseract-ocr-chi-tra
    return
  fi
  if command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y tesseract tesseract-langpack-eng tesseract-langpack-chi_sim tesseract-langpack-chi_tra
    return
  fi
  if command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --needed tesseract tesseract-data-eng tesseract-data-chi_sim tesseract-data-chi_tra
    return
  fi
  echo "Unsupported Linux package manager. Install Tesseract OCR and eng/chi_sim/chi_tra data manually." >&2
  return 1
}

install_tesseract() {
  case "$(uname -s)" in
    Darwin) install_tesseract_macos ;;
    Linux) install_tesseract_linux ;;
    *)
      echo "Unsupported OS for this Bash installer. On Windows, use scripts/install_dependencies.ps1." >&2
      return 1
      ;;
  esac
}

find_tesseract() {
  if command -v tesseract >/dev/null 2>&1; then
    command -v tesseract
    return
  fi
  for candidate in /opt/homebrew/bin/tesseract /usr/local/bin/tesseract; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done
}

if [[ "$mode" == "--install" ]]; then
  install_python_packages
  install_tesseract
fi

if ! check_python_packages; then
  echo "Run: PYTHON=$python_bin scripts/install_dependencies.sh --install" >&2
  exit 1
fi

tesseract_bin="$(find_tesseract || true)"
if [[ -z "$tesseract_bin" ]]; then
  echo "Missing dependency: tesseract" >&2
  echo "Run: scripts/install_dependencies.sh --install" >&2
  exit 1
fi

langs="$("$tesseract_bin" --list-langs 2>/dev/null || true)"
missing=0
for lang in eng osd chi_sim chi_tra; do
  if ! grep -qx "$lang" <<<"$langs"; then
    echo "Missing Tesseract language data: $lang" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "Run: scripts/install_dependencies.sh --install or install missing Tesseract language data manually." >&2
  exit 1
fi

echo "Dependency check passed."
"$python_bin" - <<'PY'
import fitz
import numpy
import PIL
print(f"PyMuPDF {fitz.version[0]}")
print(f"NumPy {numpy.__version__}")
print(f"Pillow {PIL.__version__}")
PY
"$tesseract_bin" --version | head -n 1
echo "OCR languages available: eng, chi_sim, chi_tra"
