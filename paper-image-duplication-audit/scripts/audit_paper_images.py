#!/usr/bin/env python3
"""Audit scientific paper figures for suspicious image reuse.

This script uses Python, Pillow, NumPy, PyMuPDF, and Tesseract OCR. PyMuPDF is
the cross-platform PDF backend for Windows, macOS, and Linux; the bundled Swift
PDFKit helpers remain as a macOS fallback. This is a first-pass triage tool:
report candidates for human review rather than definitive misconduct.
"""

from __future__ import annotations

import argparse
import html
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
PANEL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class FigureCandidate:
    figure: str
    figure_number: int
    page: int
    bbox: tuple[int, int, int, int]
    image_path: str


@dataclass
class ProteinRow:
    index: int
    label: str | None
    center_y: float
    source: str
    confidence: float | None


@dataclass
class PanelCandidate:
    figure: str
    page: int
    label: str
    label_source: str
    label_confidence: float | None
    bbox: tuple[int, int, int, int]
    image_path: str
    category: str
    colorfulness: float
    horizontal_score: float
    ocr_word_count: int
    strip_count: int
    text_filtered_strip_count: int
    small_filtered_strip_count: int
    protein_rows: list[ProteinRow]


@dataclass
class StripCandidate:
    figure: str
    page: int
    panel_label: str
    strip_label: str
    bbox: tuple[int, int, int, int]
    image_path: str
    variance: float
    row_label: str | None
    row_index: int | None
    row_distance: float | None
    row_source: str | None


@dataclass
class MatchCandidate:
    figure: str
    page: int
    panel_a: str
    panel_b: str
    strip_a: str
    strip_b: str
    row_label_a: str | None
    row_label_b: str | None
    row_index_a: int | None
    row_index_b: int | None
    row_match: str
    score: float
    context_score: float
    orientation: str
    level: str
    evidence_area_a: int
    evidence_area_b: int
    area_ratio: float
    panel_a_image: str
    panel_b_image: str
    strip_a_image: str
    strip_b_image: str
    review_image: str
    note: str


@dataclass
class OcrWord:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass
class StripExtractionResult:
    strips: list[tuple[tuple[int, int, int, int], float]]
    text_filtered_count: int
    small_filtered_count: int


@dataclass
class ComparisonStats:
    pairs_considered: int = 0
    pairs_skipped_row_mismatch: int = 0
    pairs_skipped_row_unknown: int = 0
    pairs_skipped_small: int = 0
    pairs_skipped_size_mismatch: int = 0
    pairs_skipped_context: int = 0
    pairs_below_score: int = 0


def log(message: str) -> None:
    print(message, flush=True)


def run_command(cmd: Sequence[str], env: dict[str, str] | None = None) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def swift_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CLANG_MODULE_CACHE_PATH", "/private/tmp/codex-swift-module-cache")
    return env


def require_swift() -> None:
    if shutil.which("swift") is None:
        raise RuntimeError(
            "No PDF backend is available. Install PyMuPDF with "
            "`python -m pip install pymupdf`; macOS can also use the Swift/PDFKit fallback."
        )


def load_pymupdf():
    try:
        return importlib.import_module("fitz")
    except ImportError:
        return None


def parse_page_spec(page_spec: str, page_count: int) -> list[int]:
    if not page_spec.strip():
        return list(range(1, page_count + 1))

    pages: set[int] = set()
    for part in page_spec.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece:
            start_text, end_text = piece.split("-", 1)
            try:
                start = max(1, int(start_text))
                end = min(page_count, int(end_text))
            except ValueError:
                continue
            if start <= end:
                pages.update(range(start, end + 1))
            continue
        try:
            page = int(piece)
        except ValueError:
            continue
        if 1 <= page <= page_count:
            pages.add(page)
    return sorted(pages)


def pdf_text_matches_from_pymupdf_page(page, page_number: int) -> dict:
    page_rect = page.rect
    page_width = float(page_rect.width)
    page_height = float(page_rect.height)
    regex_specs = [
        ("figure_title", re.compile(r"\b(?:Supplementary\s+)?Figure\s+\d+\.?", re.I)),
        ("panel_caption", re.compile(r"\b[A-Z](?:-[A-Z])?\."))
    ]

    lines: list[tuple[str, tuple[float, float, float, float], int]] = []
    text_parts: list[str] = []
    cursor = 0
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [span.get("text", "") for span in line.get("spans", [])]
            line_text = "".join(spans).strip()
            if not line_text:
                continue
            bbox = tuple(float(v) for v in line["bbox"])
            lines.append((line_text, bbox, cursor))
            text_parts.append(line_text)
            cursor += len(line_text) + 1

    matches: list[dict] = []
    for line_text, bbox, line_index in lines:
        x0, y0, x1, y1 = bbox
        for kind, regex in regex_specs:
            for match in regex.finditer(line_text):
                matches.append(
                    {
                        "kind": kind,
                        "text": match.group(0),
                        "index": line_index + match.start(),
                        "x": x0,
                        "y": page_height - y1,
                        "w": max(0.0, x1 - x0),
                        "h": max(0.0, y1 - y0),
                    }
                )

    matches.sort(key=lambda item: item["index"])
    return {
        "page": page_number,
        "width": page_width,
        "height": page_height,
        "text": "\n".join(text_parts),
        "matches": matches,
    }


def extract_layout_pymupdf(pdf_path: Path, output_path: Path) -> dict:
    fitz = load_pymupdf()
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Install with: python -m pip install pymupdf")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open(str(pdf_path))
    try:
        pages = [
            pdf_text_matches_from_pymupdf_page(document[page_index], page_index + 1)
            for page_index in range(document.page_count)
        ]
        layout = {"page_count": document.page_count, "pages": pages}
    finally:
        document.close()

    output_path.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"layout\t{output_path}")
    return layout


def render_pages_pymupdf(pdf_path: Path, output_dir: Path, dpi: int, page_spec: str) -> None:
    fitz = load_pymupdf()
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Install with: python -m pip install pymupdf")

    output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(str(pdf_path))
    try:
        pages = parse_page_spec(page_spec, document.page_count)
        log(f"page_count\t{document.page_count}")
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        for page_number in pages:
            page = document[page_number - 1]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            output_path = output_dir / f"page-{page_number:03d}.png"
            pixmap.save(str(output_path))
            log(f"rendered\t{page_number}\t{output_path}")
    finally:
        document.close()


def find_tesseract() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"):
        if Path(candidate).exists():
            return candidate
    return None


def tesseract_languages(tesseract: str) -> set[str]:
    proc = subprocess.run(
        [tesseract, "--list-langs"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return set()
    return set(proc.stdout.split())


def preferred_ocr_language(langs: set[str]) -> str:
    selected = [lang for lang in ("eng", "chi_sim", "chi_tra") if lang in langs]
    return "+".join(selected) if selected else "eng"


def check_ocr_dependencies() -> None:
    if load_pymupdf() is None:
        log("PDF dependency note: PyMuPDF not found. Install with: python -m pip install pymupdf")

    tesseract = find_tesseract()
    if tesseract is None:
        log("OCR dependency note: tesseract not found. Run scripts/install_dependencies.sh --install for OCR-assisted labeling.")
        return

    try:
        langs = tesseract_languages(tesseract)
    except OSError as exc:
        log(f"OCR dependency note: cannot inspect tesseract languages ({exc}).")
        return

    missing = [lang for lang in ("chi_sim", "chi_tra") if lang not in langs]
    if missing:
        joined = ", ".join(missing)
        log(f"OCR dependency note: missing Chinese Tesseract language data: {joined}. Run scripts/install_dependencies.sh --install.")


def parse_tesseract_tsv(tsv_text: str, scale: float) -> list[OcrWord]:
    lines = [line for line in tsv_text.splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    required = {"text", "conf", "left", "top", "width", "height"}
    if not required.issubset(header):
        return []
    idx = {name: header.index(name) for name in required}
    words: list[OcrWord] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < len(header):
            continue
        text = parts[idx["text"]].strip()
        if not text:
            continue
        try:
            confidence = float(parts[idx["conf"]])
            left = int(float(parts[idx["left"]]) / scale)
            top = int(float(parts[idx["top"]]) / scale)
            width = int(float(parts[idx["width"]]) / scale)
            height = int(float(parts[idx["height"]]) / scale)
        except ValueError:
            continue
        if confidence < 0:
            continue
        words.append(
            OcrWord(
                text=text,
                confidence=confidence,
                bbox=(left, top, left + width, top + height),
            )
        )
    return words


def run_ocr_words(
    image_path: Path,
    ocr_dir: Path,
    stem: str,
    psm: int = 11,
    scale: float = 2.0,
) -> list[OcrWord]:
    tesseract = find_tesseract()
    if tesseract is None:
        return []

    langs = tesseract_languages(tesseract)
    language = preferred_ocr_language(langs)
    ocr_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image, cutoff=1)
    image = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.BICUBIC,
    )
    preprocessed_path = ocr_dir / f"{stem}_ocr.png"
    image.save(preprocessed_path)

    proc = subprocess.run(
        [tesseract, str(preprocessed_path), "stdout", "--psm", str(psm), "-l", language, "tsv"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        log(f"OCR note: tesseract failed for {image_path.name}: {proc.stderr.strip()}")
        return []

    tsv_path = ocr_dir / f"{stem}.tsv"
    json_path = ocr_dir / f"{stem}.json"
    tsv_path.write_text(proc.stdout, encoding="utf-8")
    words = parse_tesseract_tsv(proc.stdout, scale)
    json_path.write_text(
        json.dumps([asdict(word) for word in words], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return words


def extract_layout(pdf_path: Path, output_path: Path, backend: str = "auto") -> dict:
    if backend in {"auto", "pymupdf"} and load_pymupdf() is not None:
        return extract_layout_pymupdf(pdf_path, output_path)
    if backend == "pymupdf":
        raise RuntimeError("PyMuPDF backend was requested but PyMuPDF is not installed.")

    require_swift()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "swift",
        str(SCRIPT_DIR / "extract_pdf_layout.swift"),
        str(pdf_path),
        str(output_path),
    ]
    run_command(cmd, env=swift_env())
    return json.loads(output_path.read_text(encoding="utf-8"))


def render_pages(pdf_path: Path, output_dir: Path, dpi: int, page_spec: str, backend: str = "auto") -> None:
    if backend in {"auto", "pymupdf"} and load_pymupdf() is not None:
        render_pages_pymupdf(pdf_path, output_dir, dpi, page_spec)
        return
    if backend == "pymupdf":
        raise RuntimeError("PyMuPDF backend was requested but PyMuPDF is not installed.")

    require_swift()
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "swift",
        str(SCRIPT_DIR / "render_pdf_pages.swift"),
        str(pdf_path),
        str(output_dir),
        str(dpi),
        page_spec,
    ]
    run_command(cmd, env=swift_env())


def figure_number_from_text(text: str) -> int | None:
    match = re.search(r"figure\s+(\d+)", text, re.I)
    return int(match.group(1)) if match else None


def pdf_rect_to_pixels(page: dict, rect: dict, dpi: int) -> tuple[int, int, int, int]:
    scale = dpi / 72.0
    x = int(round(rect["x"] * scale))
    y = int(round((page["height"] - (rect["y"] + rect["h"])) * scale))
    w = int(round(rect["w"] * scale))
    h = int(round(rect["h"] * scale))
    return x, y, w, h


def match_as_rect(match: dict) -> dict:
    return {
        "x": float(match["x"]),
        "y": float(match["y"]),
        "w": float(match["w"]),
        "h": float(match["h"]),
    }


def discover_figures(layout: dict, dpi: int, target_figure: int | None) -> list[dict]:
    figures: list[dict] = []
    for page in layout["pages"]:
        title_matches = [m for m in page["matches"] if m["kind"] == "figure_title"]
        caption_matches = [m for m in page["matches"] if m["kind"] == "panel_caption"]

        for title in title_matches:
            if title["text"].lower().startswith("supplementary"):
                continue
            number = figure_number_from_text(title["text"])
            if number is None:
                continue
            if target_figure is not None and number != target_figure:
                continue

            title_px = pdf_rect_to_pixels(page, match_as_rect(title), dpi)
            crop_top = title_px[1] + title_px[3] + max(12, int(dpi * 0.08))
            candidate_captions: list[tuple[int, dict]] = []
            for cap in caption_matches:
                if cap["index"] <= title["index"]:
                    continue
                cap_px = pdf_rect_to_pixels(page, match_as_rect(cap), dpi)
                if cap_px[1] <= crop_top:
                    continue
                if cap["x"] > page["width"] * 0.35:
                    continue
                candidate_captions.append((cap_px[1], cap))

            caption_px = None
            if candidate_captions:
                _, caption = sorted(candidate_captions, key=lambda item: item[0])[0]
                caption_px = pdf_rect_to_pixels(page, match_as_rect(caption), dpi)

            page_h_px = int(round(page["height"] * dpi / 72.0))
            crop_bottom = caption_px[1] - max(8, int(dpi * 0.04)) if caption_px else page_h_px - 80
            if crop_bottom - crop_top < 100:
                continue

            figures.append(
                {
                    "figure": f"Figure {number}",
                    "figure_number": number,
                    "page": page["page"],
                    "crop_top": crop_top,
                    "crop_bottom": crop_bottom,
                }
            )
    return figures


def content_bbox(image: Image.Image, threshold: int = 248) -> tuple[int, int, int, int] | None:
    gray = np.array(image.convert("L"))
    mask = gray < threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def save_figures(
    figures: list[dict],
    pages_dir: Path,
    figures_dir: Path,
    dpi: int,
) -> list[FigureCandidate]:
    saved: list[FigureCandidate] = []
    figures_dir.mkdir(parents=True, exist_ok=True)
    for fig in figures:
        page_path = pages_dir / f"page-{fig['page']:03d}.png"
        if not page_path.exists():
            continue
        page_img = Image.open(page_path).convert("RGB")
        y0 = max(0, fig["crop_top"])
        y1 = min(page_img.height, fig["crop_bottom"])
        rough = page_img.crop((0, y0, page_img.width, y1))
        bbox = content_bbox(rough)
        if bbox is None:
            continue
        x0, inner_y0, x1, inner_y1 = bbox
        pad = max(8, int(dpi * 0.05))
        x0 = max(0, x0 - pad)
        x1 = min(page_img.width, x1 + pad)
        final_y0 = max(0, y0 + inner_y0 - pad)
        final_y1 = min(page_img.height, y0 + inner_y1 + pad)
        crop = page_img.crop((x0, final_y0, x1, final_y1))
        out_path = figures_dir / f"figure-{fig['figure_number']}_page-{fig['page']:03d}.png"
        crop.save(out_path)
        saved.append(
            FigureCandidate(
                figure=fig["figure"],
                figure_number=fig["figure_number"],
                page=fig["page"],
                bbox=(x0, final_y0, x1, final_y1),
                image_path=str(out_path),
            )
        )
    return saved


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values.astype(float), kernel, mode="same")


def intervals_from_projection(
    projection: np.ndarray,
    threshold: float,
    merge_gap: int,
    min_len: int,
) -> list[tuple[int, int]]:
    active = projection > threshold
    intervals: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(active):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            intervals.append((start, idx))
            start = None
    if start is not None:
        intervals.append((start, len(active)))

    merged: list[tuple[int, int]] = []
    for begin, end in intervals:
        if not merged or begin - merged[-1][1] > merge_gap:
            merged.append((begin, end))
        else:
            merged[-1] = (merged[-1][0], end)
    return [(begin, end) for begin, end in merged if end - begin >= min_len]


def mask_from_image(image: Image.Image, threshold: int = 247) -> np.ndarray:
    gray = np.array(image.convert("L"))
    return gray < threshold


def projection_panels(image: Image.Image) -> list[tuple[int, int, int, int]]:
    mask = mask_from_image(image)
    height, width = mask.shape
    row_projection = smooth(mask.sum(axis=1), max(9, height // 80))
    row_threshold = max(20, width * 0.028)
    row_intervals = intervals_from_projection(
        row_projection,
        row_threshold,
        merge_gap=max(8, height // 80),
        min_len=max(40, height // 22),
    )

    boxes: list[tuple[int, int, int, int]] = []
    for y0, y1 in row_intervals:
        row_mask = mask[y0:y1, :]
        col_projection = smooth(row_mask.sum(axis=0), max(9, width // 120))
        col_threshold = max(10, (y1 - y0) * 0.12)
        col_intervals = intervals_from_projection(
            col_projection,
            col_threshold,
            merge_gap=max(10, width // 90),
            min_len=max(65, width // 16),
        )
        for x0, x1 in col_intervals:
            pad = 6
            box = (
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(width, x1 + pad),
                min(height, y1 + pad),
            )
            w = box[2] - box[0]
            h = box[3] - box[1]
            if w >= 80 and h >= 70 and w * h >= 9000:
                boxes.append(box)
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def component_panels(image: Image.Image) -> list[tuple[int, int, int, int]]:
    mask_img = Image.fromarray((mask_from_image(image).astype(np.uint8) * 255), mode="L")
    dilated = mask_img.filter(ImageFilter.MaxFilter(31))
    mask = np.array(dilated) > 0
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            count = 0
            while stack:
                sx, sy = stack.pop()
                count += 1
                min_x = min(min_x, sx)
                max_x = max(max_x, sx)
                min_y = min(min_y, sy)
                max_y = max(max_y, sy)
                for nx, ny in ((sx + 1, sy), (sx - 1, sy), (sx, sy + 1), (sx, sy - 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))
            if count < 500:
                continue
            box = (
                max(0, min_x - 10),
                max(0, min_y - 10),
                min(width, max_x + 11),
                min(height, max_y + 11),
            )
            w = box[2] - box[0]
            h = box[3] - box[1]
            if w >= 80 and h >= 70 and w * h >= 9000:
                boxes.append(box)
    return merge_overlapping_boxes(boxes)


def connected_component_boxes(mask: np.ndarray, min_count: int = 1) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    boxes: list[tuple[int, int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            count = 0
            while stack:
                sx, sy = stack.pop()
                count += 1
                min_x = min(min_x, sx)
                max_x = max(max_x, sx)
                min_y = min(min_y, sy)
                max_y = max(max_y, sy)
                for nx, ny in ((sx + 1, sy), (sx - 1, sy), (sx, sy + 1), (sx, sy - 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))
            if count >= min_count:
                boxes.append((min_x, min_y, max_x + 1, max_y + 1, count))
    return boxes


def boxes_overlap_or_touch(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad_x: int,
    pad_y: int,
) -> bool:
    return not (
        a[2] + pad_x < b[0]
        or b[2] + pad_x < a[0]
        or a[3] + pad_y < b[1]
        or b[3] + pad_y < a[1]
    )


def merge_overlapping_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    changed = True
    merged = boxes[:]
    while changed:
        changed = False
        result: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)
        for i, box in enumerate(merged):
            if used[i]:
                continue
            current = box
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if boxes_overlap_or_touch(current, merged[j], pad_x=8, pad_y=8):
                    current = (
                        min(current[0], merged[j][0]),
                        min(current[1], merged[j][1]),
                        max(current[2], merged[j][2]),
                        max(current[3], merged[j][3]),
                    )
                    used[j] = True
                    changed = True
            result.append(current)
        merged = result
    return sorted(merged, key=lambda b: (b[1], b[0]))


def row_major_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    rows: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):
        center_y = (box[1] + box[3]) / 2
        placed = False
        for row in rows:
            row_center = sum((b[1] + b[3]) / 2 for b in row) / len(row)
            row_height = max(b[3] - b[1] for b in row)
            if abs(center_y - row_center) < max(35, row_height * 0.45):
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    ordered: list[tuple[int, int, int, int]] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b[0]))
    return ordered


def overlap_fraction(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def expand_box(
    box: tuple[int, int, int, int],
    pad_x: int,
    pad_y: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, box[0] - pad_x),
        max(0, box[1] - pad_y),
        min(width, box[2] + pad_x),
        min(height, box[3] + pad_y),
    )


def box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def box_size(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return max(0, box[2] - box[0]), max(0, box[3] - box[1])


def box_area_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    area_a = box_area(a)
    area_b = box_area(b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return min(area_a, area_b) / max(area_a, area_b)


def has_minimum_evidence(
    box: tuple[int, int, int, int],
    min_area: int,
    min_width: int,
    min_height: int,
) -> bool:
    width, height = box_size(box)
    return width >= min_width and height >= min_height and width * height >= min_area


def normalize_panel_label(text: str) -> str | None:
    clean = re.sub(r"[^A-Za-z]", "", text).upper()
    if len(clean) == 1 and clean in PANEL_LETTERS:
        return clean
    return None


def normalize_protein_label(text: str, *, allow_short_cgas: bool = False) -> str | None:
    compact = re.sub(r"[^A-Za-z0-9αΑ-]", "", text).replace("α", "a").replace("Α", "A")
    upper = compact.upper().replace("-", "")
    if not upper:
        return None

    if "PTBK1" in upper or upper.endswith("PTBK") or "PPTBK1" in upper:
        return "p-TBK1"
    if "TBK1" in upper:
        return "TBK1"
    if "CGAS" in upper or upper.endswith("GAS") or (allow_short_cgas and upper == "AS"):
        return "cGAS"
    if "STING" in upper:
        return "STING"
    if "ANT2" in upper:
        return "ANT2"
    if "SPG7" in upper:
        return "SPG7"
    if "CYPD" in upper:
        return "CypD"
    if "TUBULIN" in upper:
        return "tubulin"
    if "GAPDH" in upper:
        return "GAPDH"
    if "ACTIN" in upper:
        return "actin"
    return None


def has_text_signal(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 2:
        return True
    return bool(re.search(r"[\u4e00-\u9fff]", compact))


def word_center(word: OcrWord) -> tuple[float, float]:
    x0, y0, x1, y1 = word.bbox
    return (x0 + x1) / 2, (y0 + y1) / 2


def box_center_y(box: tuple[int, int, int, int]) -> float:
    return (box[1] + box[3]) / 2


def assign_panel_labels(
    boxes: list[tuple[int, int, int, int]],
    ocr_words: list[OcrWord],
) -> list[tuple[str, str, float | None]]:
    candidates: list[tuple[float, int, str, float]] = []
    for word in ocr_words:
        label = normalize_panel_label(word.text)
        if label is None or word.confidence < 35:
            continue
        wx0, wy0, wx1, wy1 = word.bbox
        ww = wx1 - wx0
        wh = wy1 - wy0
        wcx, wcy = word_center(word)
        for idx, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            bw = x1 - x0
            bh = y1 - y0
            if ww > max(50, bw * 0.30) or wh > max(45, bh * 0.30):
                continue
            if not (x0 - 45 <= wcx <= x0 + bw * 0.34 and y0 - 30 <= wcy <= y0 + bh * 0.34):
                continue
            dx = max(0.0, wcx - x0)
            dy = max(0.0, wcy - y0)
            score = dx + dy * 1.8 - word.confidence * 0.2
            candidates.append((score, idx, label, word.confidence))

    labels: list[str | None] = [None] * len(boxes)
    confidences: list[float | None] = [None] * len(boxes)
    used_labels: set[str] = set()
    used_boxes: set[int] = set()
    for _, idx, label, confidence in sorted(candidates, key=lambda item: item[0]):
        if idx in used_boxes or label in used_labels:
            continue
        labels[idx] = label
        confidences[idx] = confidence
        used_boxes.add(idx)
        used_labels.add(label)

    assigned: list[tuple[str, str, float | None]] = []
    fallback_cursor = 0
    for idx in range(len(boxes)):
        if labels[idx] is not None:
            assigned.append((labels[idx] or PANEL_LETTERS[idx], "ocr", confidences[idx]))
            continue
        while fallback_cursor < len(PANEL_LETTERS) and PANEL_LETTERS[fallback_cursor] in used_labels:
            fallback_cursor += 1
        label = PANEL_LETTERS[fallback_cursor] if fallback_cursor < len(PANEL_LETTERS) else f"P{idx + 1}"
        fallback_cursor += 1
        used_labels.add(label)
        assigned.append((label, "row_major", None))
    return assigned


def words_for_box(
    words: list[OcrWord],
    box: tuple[int, int, int, int],
) -> list[OcrWord]:
    x0, y0, _, _ = box
    selected: list[OcrWord] = []
    for word in words:
        if overlap_fraction(word.bbox, box) <= 0:
            continue
        wx0, wy0, wx1, wy1 = word.bbox
        selected.append(
            OcrWord(
                text=word.text,
                confidence=word.confidence,
                bbox=(wx0 - x0, wy0 - y0, wx1 - x0, wy1 - y0),
            )
        )
    return selected


def detect_blot_row_centers(panel: Image.Image) -> list[float]:
    roi = detect_blot_roi(panel)
    if roi is None:
        return []
    gray = np.array(panel.convert("L"))
    x0, y0, x1, y1 = roi
    roi_gray = gray[y0:y1, x0:x1]
    edge_mask = roi_gray < 235
    row_projection = edge_mask.sum(axis=1)
    threshold = max(12, (x1 - x0) * 0.62)
    intervals = intervals_from_projection(
        row_projection,
        threshold=threshold,
        merge_gap=1,
        min_len=2,
    )

    centers: list[float] = []
    for top, bottom in intervals:
        center = y0 + (top + bottom) / 2
        if not centers or abs(center - centers[-1]) >= 12:
            centers.append(center)
    return centers


def detect_protein_rows(panel: Image.Image, words: list[OcrWord]) -> list[ProteinRow]:
    roi = detect_blot_roi(panel)
    image_width, _ = panel.size
    row_centers = detect_blot_row_centers(panel)
    label_candidates: list[tuple[float, str, str, float]] = []
    if roi is not None:
        x0, _, x1, _ = roi
        right_label_min = x0 + (x1 - x0) * 0.72
        right_label_max = min(image_width + 40, x1 + max(80, int(image_width * 0.35)))
    else:
        right_label_min = image_width * 0.55
        right_label_max = image_width + 40

    for word in words:
        if word.confidence < 20:
            continue
        wcx, wcy = word_center(word)
        if not (right_label_min <= wcx <= right_label_max):
            continue
        label = normalize_protein_label(word.text, allow_short_cgas=True)
        if label is None:
            continue
        label_candidates.append((wcy, label, "ocr", word.confidence))

    rows: list[ProteinRow] = []
    used_candidates: set[int] = set()
    label_for_center: dict[int, tuple[str, str, float]] = {}
    possible_matches: list[tuple[float, float, int, int]] = []
    for center_idx, center in enumerate(row_centers):
        for candidate_idx, candidate in enumerate(label_candidates):
            distance = abs(candidate[0] - center)
            if distance <= 12:
                possible_matches.append((distance, -candidate[3], center_idx, candidate_idx))
    used_centers: set[int] = set()
    for _, _, center_idx, candidate_idx in sorted(possible_matches):
        if center_idx in used_centers or candidate_idx in used_candidates:
            continue
        _, label, source, confidence = label_candidates[candidate_idx]
        label_for_center[center_idx] = (label, source, confidence)
        used_centers.add(center_idx)
        used_candidates.add(candidate_idx)

    for center_idx, center in enumerate(row_centers):
        label = None
        source = "image"
        confidence = None
        if center_idx in label_for_center:
            label, source, confidence = label_for_center[center_idx]
        rows.append(
            ProteinRow(
                index=len(rows),
                label=label,
                center_y=round(center, 2),
                source=source,
                confidence=round(confidence, 2) if confidence is not None else None,
            )
        )

    for idx, (center, label, source, confidence) in enumerate(label_candidates):
        if idx in used_candidates:
            continue
        if any(abs(center - row.center_y) <= 14 for row in rows):
            continue
        rows.append(
            ProteinRow(
                index=len(rows),
                label=label,
                center_y=round(center, 2),
                source=source,
                confidence=round(confidence, 2),
            )
        )

    rows.sort(key=lambda row: row.center_y)
    for index, row in enumerate(rows):
        row.index = index
    return rows


def assign_strip_row(
    strip_box: tuple[int, int, int, int],
    rows: list[ProteinRow],
) -> tuple[str | None, int | None, float | None, str | None]:
    if not rows:
        return None, None, None, None
    center = box_center_y(strip_box)
    best = min(rows, key=lambda row: abs(center - row.center_y))
    distance = abs(center - best.center_y)
    if distance > 22:
        return None, None, round(distance, 2), None
    return best.label, best.index, round(distance, 2), best.source


def propagate_protein_row_labels(panels: list[PanelCandidate], strips: list[StripCandidate]) -> None:
    panels_by_key = {(panel.figure, panel.page, panel.label): panel for panel in panels}
    panels_by_figure: dict[tuple[str, int], list[PanelCandidate]] = {}
    for panel in panels:
        if panel.category == "blot":
            panels_by_figure.setdefault((panel.figure, panel.page), []).append(panel)

    for figure_panels in panels_by_figure.values():
        labels_by_index: dict[int, list[str]] = {}
        for panel in figure_panels:
            for row in panel.protein_rows:
                if row.label:
                    labels_by_index.setdefault(row.index, []).append(row.label)

        propagated: dict[int, str] = {}
        for row_index, labels in labels_by_index.items():
            unique = set(labels)
            if len(unique) == 1:
                propagated[row_index] = labels[0]

        for panel in figure_panels:
            existing_labels = {
                row.label
                for row in panel.protein_rows
                if row.label is not None
            }
            for row in panel.protein_rows:
                if row.label is None and row.index in propagated:
                    label = propagated[row.index]
                    if label in existing_labels:
                        continue
                    row.label = label
                    row.source = "propagated"
                    row.confidence = None

    for strip in strips:
        if strip.row_label is not None or strip.row_index is None:
            continue
        panel = panels_by_key.get((strip.figure, strip.page, strip.panel_label))
        if panel is None or strip.row_index >= len(panel.protein_rows):
            continue
        row = panel.protein_rows[strip.row_index]
        if row.label:
            strip.row_label = row.label
            strip.row_source = row.source


def is_ocr_text_region(
    box: tuple[int, int, int, int],
    words: list[OcrWord],
    image_width: int,
    image_height: int,
) -> bool:
    for word in words:
        if word.confidence < 55 or not has_text_signal(word.text):
            continue
        word_box = expand_box(word.bbox, pad_x=3, pad_y=2, width=image_width, height=image_height)
        box_covered = overlap_fraction(box, word_box)
        word_covered = overlap_fraction(word_box, box)
        if word_covered > 0.60 and box_covered > 0.30:
            return True
    return False


def draw_ocr_overlay(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    assignments: list[tuple[str, str, float | None]],
    words: list[OcrWord],
    output_path: Path,
) -> None:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for word in words:
        if word.confidence < 35:
            continue
        draw.rectangle(word.bbox, outline=(232, 140, 0), width=2)
    for box, (label, label_source, confidence) in zip(boxes, assignments):
        color = (0, 114, 178) if label_source == "ocr" else (80, 80, 80)
        draw.rectangle(box, outline=color, width=3)
        suffix = f" {confidence:.0f}" if confidence is not None else ""
        draw.text((box[0] + 4, max(0, box[1] - 16)), f"{label} {label_source}{suffix}", fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def detect_blot_roi(image: Image.Image) -> tuple[int, int, int, int] | None:
    gray = np.array(image.convert("L"))
    height, width = gray.shape
    midtone = (gray >= 170) & (gray <= 248)
    row_projection = smooth(midtone.sum(axis=1), 5)
    row_intervals = intervals_from_projection(
        row_projection,
        threshold=max(30, width * 0.35),
        merge_gap=10,
        min_len=max(35, int(height * 0.25)),
    )
    if not row_intervals:
        return None

    y0, y1 = max(row_intervals, key=lambda interval: interval[1] - interval[0])
    roi_mask = midtone[y0:y1, :]
    col_projection = smooth(roi_mask.sum(axis=0), 5)
    col_intervals = intervals_from_projection(
        col_projection,
        threshold=max(25, (y1 - y0) * 0.45),
        merge_gap=10,
        min_len=max(55, int(width * 0.35)),
    )
    if not col_intervals:
        return None

    x0, x1 = max(col_intervals, key=lambda interval: interval[1] - interval[0])
    if x1 - x0 < width * 0.35 or y1 - y0 < height * 0.25:
        return None
    return x0, y0, x1, y1


def panel_features(image: Image.Image) -> tuple[str, float, float]:
    rgb = np.array(image.convert("RGB")).astype(float)
    gray = np.array(image.convert("L"))
    height, width = gray.shape
    colorfulness = float(np.mean(np.std(rgb, axis=2)))
    dark = gray < 230
    row_counts = dark.sum(axis=1)
    wide_rows = row_counts > max(18, width * 0.22)
    horizontal_score = float(wide_rows.mean())
    blot_roi = detect_blot_roi(image)
    category = "blot" if blot_roi is not None and width > 90 and height > 90 else "other"
    return category, colorfulness, horizontal_score


def extract_strips(
    panel: Image.Image,
    text_words: list[OcrWord] | None = None,
    min_patch_area: int = 450,
    min_patch_width: int = 18,
    min_patch_height: int = 12,
) -> StripExtractionResult:
    gray = np.array(panel.convert("L"))
    roi = detect_blot_roi(panel)
    if roi is None:
        return StripExtractionResult(strips=[], text_filtered_count=0, small_filtered_count=0)

    candidates: list[tuple[tuple[int, int, int, int], float]] = []
    text_filtered_count = 0
    small_filtered_count = 0
    text_words = text_words or []
    x0, y0, x1, y1 = roi
    roi_h = y1 - y0
    win_h = max(14, min(30, int(roi_h / 7)))
    step = max(5, win_h // 3)

    for win_y0 in range(y0, max(y0 + 1, y1 - win_h + 1), step):
        win_y1 = min(y1, win_y0 + win_h)
        box = (
            max(0, x0 - 2),
            max(0, win_y0 - 1),
            min(gray.shape[1], x1 + 2),
            min(gray.shape[0], win_y1 + 1),
        )
        patch = gray[box[1] : box[3], box[0] : box[2]]
        variance = float(np.std(patch))
        dark_fraction = float((patch < 190).mean())
        if variance < 5.0 or dark_fraction < 0.015:
            continue
        if is_ocr_text_region(box, text_words, gray.shape[1], gray.shape[0]):
            text_filtered_count += 1
            continue
        candidates.append((box, variance))

    roi_gray = gray[y0:y1, x0:x1]
    band_mask = roi_gray < 205
    band_mask_img = Image.fromarray((band_mask.astype(np.uint8) * 255), mode="L")
    band_mask = np.array(band_mask_img.filter(ImageFilter.MaxFilter(3))) > 0
    for bx0, by0, bx1, by1, count in connected_component_boxes(band_mask, min_count=5):
        bw = bx1 - bx0
        bh = by1 - by0
        if bw < 4 or bh < 2:
            continue
        if bw > (x1 - x0) * 0.38 or bh > 22:
            continue
        if count > 500:
            continue
        box = (
            max(0, x0 + bx0 - 7),
            max(0, y0 + by0 - 5),
            min(gray.shape[1], x0 + bx1 + 7),
            min(gray.shape[0], y0 + by1 + 5),
        )
        if not has_minimum_evidence(box, min_patch_area, min_patch_width, min_patch_height):
            small_filtered_count += 1
            continue
        patch = gray[box[1] : box[3], box[0] : box[2]]
        variance = float(np.std(patch))
        dark_fraction = float((patch < 190).mean())
        if variance < 6.0 or dark_fraction < 0.035:
            continue
        if is_ocr_text_region(box, text_words, gray.shape[1], gray.shape[0]):
            text_filtered_count += 1
            continue
        candidates.append((box, variance + dark_fraction * 20.0))

    for row_center in detect_blot_row_centers(panel):
        row_half_height = max(9, min(15, int(roi_h / 16)))
        row_y0 = max(y0, int(round(row_center - row_half_height)))
        row_y1 = min(y1, int(round(row_center + row_half_height)))
        if row_y1 - row_y0 < min_patch_height:
            continue
        row_gray = gray[row_y0:row_y1, x0:x1]
        row_mask = row_gray < 205
        row_mask[row_mask.sum(axis=1) > max(12, row_mask.shape[1] * 0.58), :] = False
        if row_mask.size == 0:
            continue
        col_projection = smooth(row_mask.sum(axis=0), 5)
        lane_threshold = max(2.0, row_mask.shape[0] * 0.08)
        lane_intervals = intervals_from_projection(
            col_projection,
            threshold=lane_threshold,
            merge_gap=2,
            min_len=3,
        )
        for lane_x0, lane_x1 in lane_intervals:
            if lane_x0 <= 5 or lane_x1 >= row_mask.shape[1] - 5:
                continue
            width = max(min_patch_width, min(36, lane_x1 - lane_x0 + 14))
            center_x = x0 + (lane_x0 + lane_x1) // 2
            center_y = int(round(row_center))
            box = (
                max(0, center_x - width // 2),
                max(0, center_y - min_patch_height // 2 - 3),
                min(gray.shape[1], center_x + (width + 1) // 2),
                min(gray.shape[0], center_y + min_patch_height // 2 + 4),
            )
            if not has_minimum_evidence(box, min_patch_area, min_patch_width, min_patch_height):
                small_filtered_count += 1
                continue
            patch = gray[box[1] : box[3], box[0] : box[2]]
            variance = float(np.std(patch))
            dark_fraction = float((patch < 190).mean())
            if variance < 5.0 or dark_fraction < 0.025:
                continue
            if is_ocr_text_region(box, text_words, gray.shape[1], gray.shape[0]):
                text_filtered_count += 1
                continue
            candidates.append((box, variance + dark_fraction * 22.0))

    deduped: list[tuple[tuple[int, int, int, int], float]] = []
    for box, variance in sorted(candidates, key=lambda item: item[1], reverse=True):
        duplicate = False
        for prior, _ in deduped:
            ix0 = max(box[0], prior[0])
            iy0 = max(box[1], prior[1])
            ix1 = min(box[2], prior[2])
            iy1 = min(box[3], prior[3])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            area = min((box[2] - box[0]) * (box[3] - box[1]), (prior[2] - prior[0]) * (prior[3] - prior[1]))
            if inter / max(1, area) > 0.65 and box_area_ratio(box, prior) > 0.35:
                duplicate = True
                break
        if not duplicate:
            deduped.append((box, variance))
        if len(deduped) >= 48:
            break

    return StripExtractionResult(
        strips=sorted(deduped, key=lambda item: (item[0][1], item[0][0])),
        text_filtered_count=text_filtered_count,
        small_filtered_count=small_filtered_count,
    )


def segment_and_save_panels(
    figures: list[FigureCandidate],
    panels_dir: Path,
    strips_dir: Path,
    ocr_dir: Path | None = None,
    min_patch_area: int = 450,
    min_patch_width: int = 18,
    min_patch_height: int = 12,
) -> tuple[list[PanelCandidate], list[StripCandidate]]:
    panels_dir.mkdir(parents=True, exist_ok=True)
    strips_dir.mkdir(parents=True, exist_ok=True)
    if ocr_dir is not None:
        ocr_dir.mkdir(parents=True, exist_ok=True)
    panels: list[PanelCandidate] = []
    strips: list[StripCandidate] = []

    for figure in figures:
        fig_img = Image.open(figure.image_path).convert("RGB")
        boxes = projection_panels(fig_img)
        if len(boxes) < 2:
            boxes = component_panels(fig_img)
        boxes = row_major_boxes(boxes)
        ocr_words: list[OcrWord] = []
        if ocr_dir is not None:
            ocr_stem = f"figure-{figure.figure_number}_page-{figure.page:03d}"
            ocr_words = run_ocr_words(Path(figure.image_path), ocr_dir, ocr_stem)
        label_assignments = assign_panel_labels(boxes, ocr_words)
        if ocr_dir is not None:
            draw_ocr_overlay(
                fig_img,
                boxes[: len(PANEL_LETTERS)],
                label_assignments[: len(PANEL_LETTERS)],
                ocr_words,
                ocr_dir / f"figure-{figure.figure_number}_page-{figure.page:03d}_overlay.png",
            )

        for box, (label, label_source, label_confidence) in zip(boxes[: len(PANEL_LETTERS)], label_assignments):
            crop = fig_img.crop(box)
            category, colorfulness, horizontal_score = panel_features(crop)
            panel_words = words_for_box(ocr_words, box)
            protein_rows = detect_protein_rows(crop, panel_words) if category == "blot" else []
            if category == "blot":
                strip_result = extract_strips(
                    crop,
                    panel_words,
                    min_patch_area=min_patch_area,
                    min_patch_width=min_patch_width,
                    min_patch_height=min_patch_height,
                )
            else:
                strip_result = StripExtractionResult(strips=[], text_filtered_count=0, small_filtered_count=0)
            panel_name = f"figure-{figure.figure_number}_page-{figure.page:03d}_panel-{label}"
            panel_path = panels_dir / f"{panel_name}.png"
            crop.save(panel_path)

            panel = PanelCandidate(
                figure=figure.figure,
                page=figure.page,
                label=label,
                label_source=label_source,
                label_confidence=round(label_confidence, 2) if label_confidence is not None else None,
                bbox=box,
                image_path=str(panel_path),
                category=category,
                colorfulness=round(colorfulness, 4),
                horizontal_score=round(horizontal_score, 4),
                ocr_word_count=len(panel_words),
                strip_count=len(strip_result.strips),
                text_filtered_strip_count=strip_result.text_filtered_count,
                small_filtered_strip_count=strip_result.small_filtered_count,
                protein_rows=protein_rows,
            )
            panels.append(panel)

            for strip_idx, (strip_box, variance) in enumerate(strip_result.strips, start=1):
                strip_label = f"{label}-strip-{strip_idx:02d}"
                strip_path = strips_dir / f"{panel_name}_strip-{strip_idx:02d}.png"
                crop.crop(strip_box).save(strip_path)
                row_label, row_index, row_distance, row_source = assign_strip_row(strip_box, protein_rows)
                strips.append(
                    StripCandidate(
                        figure=figure.figure,
                        page=figure.page,
                        panel_label=label,
                        strip_label=strip_label,
                        bbox=strip_box,
                        image_path=str(strip_path),
                        variance=round(variance, 4),
                        row_label=row_label,
                        row_index=row_index,
                        row_distance=row_distance,
                        row_source=row_source,
                    )
                )
    propagate_protein_row_labels(panels, strips)
    return panels, strips


def normalized_patch(path: str, flip: str = "none", size: tuple[int, int] = (180, 36)) -> np.ndarray:
    image = Image.open(path).convert("L")
    bbox = content_bbox(image, threshold=248)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        pad = 2
        image = image.crop((
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(image.width, x1 + pad),
            min(image.height, y1 + pad),
        ))
    if flip == "h":
        image = ImageOps.mirror(image)
    elif flip == "v":
        image = ImageOps.flip(image)
    image = ImageOps.autocontrast(image, cutoff=1)
    image = image.resize(size, Image.Resampling.BICUBIC)
    arr = np.array(image).astype(float)
    arr = arr - arr.mean()
    std = arr.std()
    if std < 1e-6:
        return arr
    return arr / std


def normalized_image(image: Image.Image, flip: str = "none", size: tuple[int, int] = (220, 80)) -> np.ndarray:
    image = image.convert("L")
    if flip == "h":
        image = ImageOps.mirror(image)
    elif flip == "v":
        image = ImageOps.flip(image)
    image = ImageOps.autocontrast(image, cutoff=1)
    image = image.resize(size, Image.Resampling.BICUBIC)
    arr = np.array(image).astype(float)
    arr = arr - arr.mean()
    std = arr.std()
    if std < 1e-6:
        return arr
    return arr / std


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("NCC arrays must have the same shape")
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if denom < 1e-9:
        return 0.0
    return float((a * b).sum() / denom)


def strip_similarity(path_a: str, path_b: str) -> tuple[float, str]:
    base_a = normalized_patch(path_a)
    best_score = -1.0
    best_orientation = "none"
    for orientation in ("none", "h", "v"):
        patch_b = normalized_patch(path_b, flip=orientation)
        score = ncc(base_a, patch_b)
        if score > best_score:
            best_score = score
            best_orientation = orientation
    return best_score, best_orientation


def context_similarity(
    panel_a: PanelCandidate,
    panel_b: PanelCandidate,
    strip_a: StripCandidate,
    strip_b: StripCandidate,
    orientation: str,
    margin: int,
) -> float:
    if margin <= 0:
        return 0.0
    image_a = Image.open(panel_a.image_path).convert("L")
    image_b = Image.open(panel_b.image_path).convert("L")
    context_a = expand_box(strip_a.bbox, margin, margin, image_a.width, image_a.height)
    context_b = expand_box(strip_b.bbox, margin, margin, image_b.width, image_b.height)
    patch_a = normalized_image(image_a.crop(context_a))
    patch_b = normalized_image(image_b.crop(context_b), flip=orientation)
    return ncc(patch_a, patch_b)


def candidate_level(score: float) -> str:
    if score >= 0.90:
        return "high"
    if score >= 0.82:
        return "medium"
    return "low"


def row_match_status(
    strip_a: StripCandidate,
    strip_b: StripCandidate,
    require_row_match: bool,
) -> tuple[bool, str]:
    if not require_row_match:
        if strip_a.row_label and strip_a.row_label == strip_b.row_label:
            return True, "label"
        if strip_a.row_index is not None and strip_a.row_index == strip_b.row_index:
            return True, "row-index"
        return True, "not-required"

    if strip_a.row_label and strip_b.row_label:
        return (strip_a.row_label == strip_b.row_label), "label"
    if strip_a.row_label or strip_b.row_label:
        return False, "partial-label"
    if strip_a.row_index is not None and strip_b.row_index is not None:
        return (strip_a.row_index == strip_b.row_index), "row-index"
    return False, "unknown"


def draw_review_image(
    panel_a: PanelCandidate,
    panel_b: PanelCandidate,
    strip_a: StripCandidate,
    strip_b: StripCandidate,
    output_path: Path,
) -> None:
    img_a = Image.open(panel_a.image_path).convert("RGB")
    img_b = Image.open(panel_b.image_path).convert("RGB")
    draw_a = ImageDraw.Draw(img_a)
    draw_b = ImageDraw.Draw(img_b)
    draw_a.rectangle(strip_a.bbox, outline=(220, 0, 0), width=4)
    draw_b.rectangle(strip_b.bbox, outline=(220, 0, 0), width=4)

    strip_img_a = Image.open(strip_a.image_path).convert("RGB")
    strip_img_b = Image.open(strip_b.image_path).convert("RGB")
    max_panel_h = max(img_a.height, img_b.height)
    strip_h = max(strip_img_a.height, strip_img_b.height)
    canvas_w = img_a.width + img_b.width + 30
    canvas_h = max_panel_h + strip_h + 70
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(img_a, (0, 35))
    canvas.paste(img_b, (img_a.width + 30, 35))
    strip_y = max_panel_h + 55
    canvas.paste(strip_img_a, (0, strip_y))
    canvas.paste(strip_img_b, (img_a.width + 30, strip_y))
    draw = ImageDraw.Draw(canvas)
    row_a = f" / {strip_a.row_label}" if strip_a.row_label else ""
    row_b = f" / {strip_b.row_label}" if strip_b.row_label else ""
    draw.text((0, 8), f"{panel_a.figure}{panel_a.label} / {strip_a.strip_label}{row_a}", fill=(0, 0, 0))
    draw.text(
        (img_a.width + 30, 8),
        f"{panel_b.figure}{panel_b.label} / {strip_b.strip_label}{row_b}",
        fill=(0, 0, 0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def compare_strips(
    panels: list[PanelCandidate],
    strips: list[StripCandidate],
    matches_dir: Path,
    min_score: float,
    top_n: int,
    min_patch_area: int,
    min_patch_width: int,
    min_patch_height: int,
    min_area_ratio: float,
    min_context_score: float,
    context_margin: int,
    require_row_match: bool,
) -> tuple[list[MatchCandidate], ComparisonStats]:
    panels_by_key = {(p.figure, p.page, p.label): p for p in panels}
    strips_by_figure: dict[tuple[str, int], list[StripCandidate]] = {}
    for strip in strips:
        strips_by_figure.setdefault((strip.figure, strip.page), []).append(strip)

    candidates: list[tuple[MatchCandidate, PanelCandidate, PanelCandidate, StripCandidate, StripCandidate]] = []
    stats = ComparisonStats()
    for (figure, page), figure_strips in strips_by_figure.items():
        for i, strip_a in enumerate(figure_strips):
            for strip_b in figure_strips[i + 1 :]:
                if strip_a.panel_label == strip_b.panel_label:
                    continue
                stats.pairs_considered += 1
                if not has_minimum_evidence(strip_a.bbox, min_patch_area, min_patch_width, min_patch_height):
                    stats.pairs_skipped_small += 1
                    continue
                if not has_minimum_evidence(strip_b.bbox, min_patch_area, min_patch_width, min_patch_height):
                    stats.pairs_skipped_small += 1
                    continue
                area_ratio = box_area_ratio(strip_a.bbox, strip_b.bbox)
                if area_ratio < min_area_ratio:
                    stats.pairs_skipped_size_mismatch += 1
                    continue
                panel_a = panels_by_key[(figure, page, strip_a.panel_label)]
                panel_b = panels_by_key[(figure, page, strip_b.panel_label)]
                if panel_a.category != "blot" or panel_b.category != "blot":
                    continue
                rows_match, row_match = row_match_status(strip_a, strip_b, require_row_match)
                if not rows_match:
                    if row_match in {"unknown", "partial-label"}:
                        stats.pairs_skipped_row_unknown += 1
                    else:
                        stats.pairs_skipped_row_mismatch += 1
                    continue
                score, orientation = strip_similarity(strip_a.image_path, strip_b.image_path)
                if score < min_score:
                    stats.pairs_below_score += 1
                    continue
                context_score = context_similarity(panel_a, panel_b, strip_a, strip_b, orientation, context_margin)
                if min_context_score > 0 and context_score < min_context_score:
                    stats.pairs_skipped_context += 1
                    continue
                review_name = (
                    f"{figure.lower().replace(' ', '-')}_"
                    f"{strip_a.row_label or 'row'}_"
                    f"{strip_a.strip_label}_vs_{strip_b.strip_label}_{score:.3f}.png"
                )
                review_path = matches_dir / review_name
                note = (
                    "Same-protein-row WB/gel candidates pass minimum evidence-size filters and show high "
                    "normalized cross-correlation after contrast normalization."
                )
                match = MatchCandidate(
                    figure=figure,
                    page=page,
                    panel_a=strip_a.panel_label,
                    panel_b=strip_b.panel_label,
                    strip_a=strip_a.strip_label,
                    strip_b=strip_b.strip_label,
                    row_label_a=strip_a.row_label,
                    row_label_b=strip_b.row_label,
                    row_index_a=strip_a.row_index,
                    row_index_b=strip_b.row_index,
                    row_match=row_match,
                    score=round(score, 4),
                    context_score=round(context_score, 4),
                    orientation=orientation,
                    level=candidate_level(score),
                    evidence_area_a=box_area(strip_a.bbox),
                    evidence_area_b=box_area(strip_b.bbox),
                    area_ratio=round(area_ratio, 4),
                    panel_a_image=panel_a.image_path,
                    panel_b_image=panel_b.image_path,
                    strip_a_image=strip_a.image_path,
                    strip_b_image=strip_b.image_path,
                    review_image=str(review_path),
                    note=note,
                )
                candidates.append((match, panel_a, panel_b, strip_a, strip_b))

    row_match_priority = {"label": 2, "row-index": 1, "not-required": 0}
    candidates.sort(
        key=lambda c: (
            row_match_priority.get(c[0].row_match, 0),
            c[0].score,
            c[0].context_score,
        ),
        reverse=True,
    )
    top_candidates = candidates[:top_n]
    for match, panel_a, panel_b, strip_a, strip_b in top_candidates:
        draw_review_image(panel_a, panel_b, strip_a, strip_b, Path(match.review_image))
    return [match for match, _, _, _, _ in top_candidates], stats


def relative_to(path: str | Path, root: Path) -> str:
    return os.path.relpath(str(path), str(root))


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_report(
    output_dir: Path,
    figures: list[FigureCandidate],
    panels: list[PanelCandidate],
    strips: list[StripCandidate],
    matches: list[MatchCandidate],
    comparison_stats: ComparisonStats,
    settings: dict[str, float | int | str],
) -> None:
    ocr_label_count = sum(1 for panel in panels if panel.label_source == "ocr")
    text_filtered_strip_count = sum(panel.text_filtered_strip_count for panel in panels)
    small_filtered_strip_count = sum(panel.small_filtered_strip_count for panel in panels)
    results = {
        "settings": settings,
        "comparison_stats": asdict(comparison_stats),
        "figures": [asdict(f) for f in figures],
        "panels": [asdict(p) for p in panels],
        "strips": [asdict(s) for s in strips],
        "matches": [asdict(m) for m in matches],
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# Paper Image Duplication Audit",
        "",
        f"- Figures processed: {len(figures)}",
        f"- Panels detected: {len(panels)}",
        f"- OCR panel labels: {ocr_label_count}",
        f"- WB/gel strips extracted: {len(strips)}",
        f"- OCR text-filtered strip candidates: {text_filtered_strip_count}",
        f"- Small WB/gel patch candidates filtered: {small_filtered_strip_count}",
        f"- Suspicious candidates: {len(matches)}",
        f"- Pairwise comparisons considered: {comparison_stats.pairs_considered}",
        f"- Pairs skipped for protein-row mismatch: {comparison_stats.pairs_skipped_row_mismatch}",
        f"- Pairs skipped for missing/partial protein-row labels: {comparison_stats.pairs_skipped_row_unknown}",
        f"- Pairs skipped for small evidence patches: {comparison_stats.pairs_skipped_small}",
        f"- Pairs skipped for size mismatch: {comparison_stats.pairs_skipped_size_mismatch}",
        f"- Pairs skipped for context threshold: {comparison_stats.pairs_skipped_context}",
        "",
        "## Suspicious Candidates",
        "",
    ]
    if not matches:
        md_lines.append("No candidates passed the configured threshold.")
    for match in matches:
        rel_review = relative_to(match.review_image, output_dir)
        md_lines.extend(
            [
                f"### {match.level.upper()} {match.figure}{match.panel_a} vs {match.figure}{match.panel_b}",
                "",
                f"- Page: {match.page}",
                f"- Strips: {match.strip_a} vs {match.strip_b}",
                f"- Protein rows: {match.row_label_a or match.row_index_a} vs {match.row_label_b or match.row_index_b} ({match.row_match})",
                f"- Score: {match.score:.4f}",
                f"- Context score: {match.context_score:.4f}",
                f"- Evidence area: {match.evidence_area_a} px vs {match.evidence_area_b} px",
                f"- Area ratio: {match.area_ratio:.4f}",
                f"- Orientation: {match.orientation}",
                f"- Note: {match.note}",
                "",
                f"![review]({rel_review})",
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(md_lines), encoding="utf-8")

    rows = []
    for match in matches:
        rel_review = html.escape(relative_to(match.review_image, output_dir))
        rows.append(
            "<tr>"
            f"<td>{html.escape(match.level)}</td>"
            f"<td>{html.escape(match.figure)}{html.escape(match.panel_a)} vs "
            f"{html.escape(match.figure)}{html.escape(match.panel_b)}</td>"
            f"<td>{match.page}</td>"
            f"<td>{html.escape(match.strip_a)} vs {html.escape(match.strip_b)}</td>"
            f"<td>{html.escape(str(match.row_label_a or match.row_index_a))} vs "
            f"{html.escape(str(match.row_label_b or match.row_index_b))}<br>{html.escape(match.row_match)}</td>"
            f"<td>{match.score:.4f}</td>"
            f"<td>{match.context_score:.4f}</td>"
            f"<td>{match.evidence_area_a} / {match.evidence_area_b}<br>ratio {match.area_ratio:.3f}</td>"
            f"<td>{html.escape(match.orientation)}</td>"
            f"<td><img src=\"{rel_review}\" /></td>"
            "</tr>"
        )
    table_body = "\n".join(rows) if rows else "<tr><td colspan=\"10\">No candidates passed the threshold.</td></tr>"
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Paper Image Duplication Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #1f2933; }}
    h1 {{ font-size: 28px; margin-bottom: 8px; }}
    .summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0 24px; }}
    .metric {{ border: 1px solid #d5dae1; border-radius: 6px; padding: 10px 14px; min-width: 150px; }}
    .metric b {{ display: block; font-size: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-top: 1px solid #d5dae1; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f6f8; }}
    img {{ max-width: 560px; height: auto; border: 1px solid #d5dae1; }}
    .note {{ color: #52606d; max-width: 920px; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>Paper Image Duplication Audit</h1>
  <p class="note">Candidates are computational triage results for manual review, not final conclusions.</p>
  <div class="summary">
    <div class="metric"><b>{len(figures)}</b>Figures processed</div>
    <div class="metric"><b>{len(panels)}</b>Panels detected</div>
    <div class="metric"><b>{ocr_label_count}</b>OCR panel labels</div>
    <div class="metric"><b>{len(strips)}</b>WB/gel strips</div>
    <div class="metric"><b>{text_filtered_strip_count}</b>OCR text-filtered strips</div>
    <div class="metric"><b>{small_filtered_strip_count}</b>Small patches filtered</div>
    <div class="metric"><b>{len(matches)}</b>Suspicious candidates</div>
    <div class="metric"><b>{comparison_stats.pairs_skipped_row_mismatch}</b>Protein-row mismatches skipped</div>
    <div class="metric"><b>{comparison_stats.pairs_skipped_row_unknown}</b>Unknown-row pairs skipped</div>
    <div class="metric"><b>{comparison_stats.pairs_skipped_small}</b>Small-patch pairs skipped</div>
    <div class="metric"><b>{comparison_stats.pairs_skipped_size_mismatch}</b>Size-mismatch pairs skipped</div>
  </div>
  <table>
    <thead>
      <tr><th>Level</th><th>Panels</th><th>Page</th><th>Strips</th><th>Protein Row</th><th>Score</th><th>Context</th><th>Evidence Area</th><th>Orientation</th><th>Review</th></tr>
    </thead>
    <tbody>
      {table_body}
    </tbody>
  </table>
</body>
</html>
"""
    (output_dir / "report.html").write_text(html_doc, encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Input manuscript PDF")
    parser.add_argument("--out", type=Path, required=True, help="Output audit directory")
    parser.add_argument("--dpi", type=int, default=180, help="Render DPI")
    parser.add_argument("--figure", type=int, help="Limit audit to a single figure number")
    parser.add_argument("--min-score", type=float, default=0.82, help="Minimum strip similarity to report")
    parser.add_argument("--top-n", type=int, default=40, help="Maximum candidate matches to report")
    parser.add_argument(
        "--min-patch-area",
        type=int,
        default=450,
        help="Minimum WB/gel evidence patch area in pixels; raise to suppress small-patch false positives",
    )
    parser.add_argument(
        "--min-patch-width",
        type=int,
        default=18,
        help="Minimum WB/gel evidence patch width in pixels",
    )
    parser.add_argument(
        "--min-patch-height",
        type=int,
        default=12,
        help="Minimum WB/gel evidence patch height in pixels",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.55,
        help="Minimum smaller/larger evidence-patch area ratio for pairwise comparison",
    )
    parser.add_argument(
        "--min-context-score",
        type=float,
        default=0.0,
        help="Optional minimum NCC for expanded local context; 0 disables context filtering",
    )
    parser.add_argument(
        "--context-margin",
        type=int,
        default=10,
        help="Pixel margin around each evidence patch used to compute context score",
    )
    parser.add_argument(
        "--allow-row-mismatch",
        action="store_true",
        help="Exploratory mode: compare WB/gel candidates even when protein-row labels or row indices differ",
    )
    parser.add_argument("--keep-existing", action="store_true", help="Reuse existing rendered pages/layout when present")
    parser.add_argument(
        "--pdf-backend",
        choices=("auto", "pymupdf", "swift"),
        default="auto",
        help="PDF backend: auto prefers PyMuPDF and falls back to macOS Swift/PDFKit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    pdf_path = args.pdf.expanduser().resolve()
    output_dir = args.out.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    check_ocr_dependencies()

    layout_path = output_dir / "layout.json"
    if args.keep_existing and layout_path.exists():
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
    else:
        log("Extracting PDF text/layout...")
        layout = extract_layout(pdf_path, layout_path, args.pdf_backend)

    figure_specs = discover_figures(layout, args.dpi, args.figure)
    if not figure_specs:
        raise RuntimeError("No matching figure pages were found in the PDF text layer.")

    page_numbers = sorted({spec["page"] for spec in figure_specs})
    page_spec = ",".join(str(page) for page in page_numbers)
    pages_dir = output_dir / "pages"
    if not args.keep_existing or not all((pages_dir / f"page-{page:03d}.png").exists() for page in page_numbers):
        log(f"Rendering page(s): {page_spec}")
        render_pages(pdf_path, pages_dir, args.dpi, page_spec, args.pdf_backend)

    for derived_dir in ("figures", "panels", "strips", "matches", "ocr"):
        reset_dir(output_dir / derived_dir)

    log("Cropping figure regions...")
    figures = save_figures(figure_specs, pages_dir, output_dir / "figures", args.dpi)
    log("Segmenting panels and extracting WB/gel strips...")
    panels, strips = segment_and_save_panels(
        figures,
        output_dir / "panels",
        output_dir / "strips",
        output_dir / "ocr",
        min_patch_area=args.min_patch_area,
        min_patch_width=args.min_patch_width,
        min_patch_height=args.min_patch_height,
    )
    log("Comparing same-category strips...")
    matches, comparison_stats = compare_strips(
        panels,
        strips,
        output_dir / "matches",
        args.min_score,
        args.top_n,
        min_patch_area=args.min_patch_area,
        min_patch_width=args.min_patch_width,
        min_patch_height=args.min_patch_height,
        min_area_ratio=args.min_area_ratio,
        min_context_score=args.min_context_score,
        context_margin=args.context_margin,
        require_row_match=not args.allow_row_mismatch,
    )
    log("Writing report...")
    write_report(
        output_dir,
        figures,
        panels,
        strips,
        matches,
        comparison_stats,
        {
            "dpi": args.dpi,
            "min_score": args.min_score,
            "min_patch_area": args.min_patch_area,
            "min_patch_width": args.min_patch_width,
            "min_patch_height": args.min_patch_height,
            "min_area_ratio": args.min_area_ratio,
            "min_context_score": args.min_context_score,
            "context_margin": args.context_margin,
            "require_row_match": not args.allow_row_mismatch,
            "pdf_backend": args.pdf_backend,
        },
    )

    log(f"Report: {output_dir / 'report.html'}")
    log(f"JSON: {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
