from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - only relevant on minimal Python builds.
    tk = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "input" / "main" / "images"
AOI_DIR = PROJECT_ROOT / "input" / "main" / "aois"
AOI_STORE_PATH = AOI_DIR / "aoi_shapes.json"
LEGACY_SEEDS_PATH = AOI_DIR / "seeds.json"
AOI_CSV_PATH = AOI_DIR / "aois.csv"
AOI_METHOD_SUMMARY_PATH = AOI_DIR / "method_summary.json"
PREVIEW_DIR = AOI_DIR / "previews"
LABEL_MAP_DIR = AOI_DIR / "label_maps"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".tif", ".tiff"}
PALETTE = [
    (230, 57, 70),
    (42, 157, 143),
    (69, 123, 157),
    (244, 162, 97),
    (131, 56, 236),
    (255, 190, 11),
    (58, 134, 255),
    (6, 214, 160),
    (239, 71, 111),
    (255, 255, 255),
]
DEFAULT_TOTAL_AOIS = 4
MAX_TOTAL_AOIS = 10
AUTO_MIN_TOTAL_AOIS = 4
AUTO_MAX_TOTAL_AOIS = 5
LIMITED_RADIUS_MARGIN_NORM = 0.035
LIMITED_RADIUS_MIN_NORM = 0.09
LIMITED_RADIUS_MAX_NORM = 0.24
REGION_MARGIN_NORM = 0.04
REGION_MIN_SIZE_NORM = 0.12
REGION_MAX_SIZE_NORM = 0.62
REGION_MAX_IOU = 0.18
LIMITED_RADIUS_METHOD = "predefined_region_with_margin"


@dataclass
class AOI:
    aoi_id: str
    label: str
    kind: str
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    locked: bool = False
    points: list[tuple[float, float]] | None = None
    method: str = "manual"
    source: str = ""

    @property
    def is_background(self) -> bool:
        return self.kind == "background"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def list_images(image_dir: Path = IMAGE_DIR) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)


def load_store(path: Path = AOI_STORE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 2, "images": {}, "archived_images": []}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("schema_version", 2)
    data.setdefault("images", {})
    data.setdefault("archived_images", [])
    return data


def save_store(store: dict[str, Any], path: Path = AOI_STORE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, sort_keys=True)
        f.write("\n")


def aoi_from_record(item: dict[str, Any], index: int) -> AOI:
    kind = str(item.get("kind") or item.get("type") or "ellipse")
    raw_points = item.get("points") or []
    points = [
        (clamp01(float(point[0])), clamp01(float(point[1])))
        for point in raw_points
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    return AOI(
        aoi_id=str(item.get("id") or item.get("aoi_id") or f"AOI{index + 1}"),
        label=str(item.get("label") or f"AOI {index + 1}"),
        kind=kind,
        x=clamp01(float(item.get("x", 0.0))),
        y=clamp01(float(item.get("y", 0.0))),
        w=clamp01(float(item.get("w", 1.0))),
        h=clamp01(float(item.get("h", 1.0))),
        locked=bool(item.get("locked", kind == "background")),
        points=points or None,
        method=str(item.get("method") or ("non_aoi" if kind == "background" else "manual")),
        source=str(item.get("source") or ""),
    )


def aoi_to_record(aoi: AOI) -> dict[str, Any]:
    record = {
        "id": aoi.aoi_id,
        "label": aoi.label,
        "kind": aoi.kind,
        "locked": aoi.locked,
        "method": aoi.method,
        "source": aoi.source,
    }
    if not aoi.is_background:
        record.update(
            {
                "x": round(aoi.x, 6),
                "y": round(aoi.y, 6),
                "w": round(aoi.w, 6),
                "h": round(aoi.h, 6),
            }
        )
        if aoi.kind == "polygon" and aoi.points:
            record["points"] = [[round(x, 6), round(y, 6)] for x, y in aoi.points]
    return record


def aois_from_record(record: dict[str, Any]) -> list[AOI]:
    aois = [aoi_from_record(item, i) for i, item in enumerate(record.get("aois", []))]
    return normalize_aois(aois)


def aois_to_record(aois: list[AOI]) -> dict[str, Any]:
    return {"aois": [aoi_to_record(aoi) for aoi in normalize_aois(aois)]}


def image_fingerprint(image_path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with image_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = image_path.stat()
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def record_fingerprint(record: dict[str, Any]) -> dict[str, Any] | None:
    fingerprint = record.get("image_fingerprint")
    return fingerprint if isinstance(fingerprint, dict) else None


def fingerprint_matches(record: dict[str, Any], image_path: Path) -> bool | None:
    fingerprint = record_fingerprint(record)
    if not fingerprint:
        return None
    current = image_fingerprint(image_path)
    return fingerprint.get("sha256") == current["sha256"] and fingerprint.get("size_bytes") == current["size_bytes"]


def with_image_metadata(record: dict[str, Any], image_path: Path) -> dict[str, Any]:
    record = dict(record)
    record["image_fingerprint"] = image_fingerprint(image_path)
    record["image_file"] = image_path.name
    return record


def archive_image_record(store: dict[str, Any], image_name: str, record: dict[str, Any], reason: str) -> None:
    store.setdefault("archived_images", []).append(
        {
            "image_file": image_name,
            "archived_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": reason,
            "record": record,
        }
    )


def normalize_aois(aois: list[AOI]) -> list[AOI]:
    background = [aoi for aoi in aois if aoi.is_background]
    foreground = [aoi for aoi in aois if not aoi.is_background]
    result = background[:1] or [AOI("BG", "background / rest", "background", locked=True)]
    result[0].aoi_id = "BG"
    result[0].label = result[0].label or "background / rest"
    result[0].locked = True
    result[0].method = result[0].method or "non_aoi"
    result[0].source = result[0].source or "default non-AOI category"
    for i, aoi in enumerate(foreground[: MAX_TOTAL_AOIS - 1], start=1):
        aoi.aoi_id = f"AOI{i}"
        aoi.label = aoi.label or f"AOI {i}"
        if aoi.kind == "polygon" and aoi.points:
            aoi.points = [(clamp01(x), clamp01(y)) for x, y in aoi.points]
            update_polygon_bbox(aoi)
        elif aoi.kind not in {"ellipse", "rect"}:
            aoi.kind = "ellipse"
        aoi.x = clamp01(aoi.x)
        aoi.y = clamp01(aoi.y)
        aoi.w = max(0.01, clamp01(aoi.w))
        aoi.h = max(0.01, clamp01(aoi.h))
        aoi.locked = False
        aoi.method = aoi.method or "manual"
        result.append(aoi)
    return result


def update_polygon_bbox(aoi: AOI) -> None:
    if not aoi.points:
        return
    xs = [x for x, _ in aoi.points]
    ys = [y for _, y in aoi.points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    aoi.x = clamp01(x0)
    aoi.y = clamp01(y0)
    aoi.w = max(0.01, clamp01(x1 - x0))
    aoi.h = max(0.01, clamp01(y1 - y0))


def simplify_points(points: list[tuple[float, float]], max_points: int = 140) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    step = max(1, math.ceil(len(points) / max_points))
    simplified = points[::step]
    if simplified[-1] != points[-1]:
        simplified.append(points[-1])
    return simplified


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-6)


def connected_components(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[np.ndarray, np.ndarray]] = []
    for start_y, start_x in zip(*np.where(mask & ~visited)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        xs: list[int] = []
        ys: list[int] = []
        while stack:
            y, x = stack.pop()
            xs.append(x)
            ys.append(y)
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        components.append((np.asarray(ys), np.asarray(xs)))
    return components


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def clamp_box(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    w = min(REGION_MAX_SIZE_NORM, max(REGION_MIN_SIZE_NORM, w))
    h = min(REGION_MAX_SIZE_NORM, max(REGION_MIN_SIZE_NORM, h))
    x = min(max(0.0, x), 1.0 - w)
    y = min(max(0.0, y), 1.0 - h)
    return x, y, w, h


def estimate_limited_radius_regions(image_path: Path, k: int | None) -> list[AOI]:
    image = Image.open(image_path).convert("RGB")
    small = image.copy()
    small.thumbnail((180, 180), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.float32)
    height, width = arr.shape[:2]

    border = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
    border_color = np.median(border, axis=0)
    color_delta = np.linalg.norm(arr - border_color, axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = gx + gy
    yy, xx = np.indices((height, width), dtype=np.float32)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    centrality = 1.0 - np.minimum(1.0, np.hypot((xx - cx) / width, (yy - cy) / height) * 1.35)

    saliency = 0.9 * zscore(color_delta) + 0.75 * zscore(edges) + 0.35 * zscore(saturation) + 0.25 * centrality

    # Work on coarse tiles rather than individual pixels. This produces
    # operational regions/fields, not point-like circular blobs.
    grid_cols = 10
    grid_rows = 10
    tile_scores = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for gy_idx in range(grid_rows):
        y0 = round(gy_idx * height / grid_rows)
        y1 = round((gy_idx + 1) * height / grid_rows)
        for gx_idx in range(grid_cols):
            x0 = round(gx_idx * width / grid_cols)
            x1 = round((gx_idx + 1) * width / grid_cols)
            tile_scores[gy_idx, gx_idx] = float(saliency[y0:y1, x0:x1].mean())

    threshold = np.percentile(tile_scores, 78)
    tile_mask = tile_scores >= threshold
    components = []
    for tile_ys, tile_xs in connected_components(tile_mask):
        if len(tile_xs) == 0:
            continue
        x0_tile = int(tile_xs.min())
        x1_tile = int(tile_xs.max()) + 1
        y0_tile = int(tile_ys.min())
        y1_tile = int(tile_ys.max()) + 1
        x = x0_tile / grid_cols - REGION_MARGIN_NORM
        y = y0_tile / grid_rows - REGION_MARGIN_NORM
        w = (x1_tile - x0_tile) / grid_cols + 2 * REGION_MARGIN_NORM
        h = (y1_tile - y0_tile) / grid_rows + 2 * REGION_MARGIN_NORM
        x, y, w, h = clamp_box(x, y, w, h)
        component_scores = tile_scores[tile_ys, tile_xs]
        score = float(component_scores.mean() * math.sqrt(len(tile_xs)))
        aspect = w / max(h, 1e-6)
        kind = "rect" if aspect > 1.55 or aspect < 0.65 or (w * h) > 0.18 else "ellipse"
        components.append(
            {
                "box": (x, y, w, h),
                "kind": kind,
                "score": score,
                "level": "coarse_region",
            }
        )

    high_threshold = np.percentile(saliency, 93)
    high_mask = saliency >= high_threshold
    min_component_px = max(12, int(width * height * 0.0025))
    for ys, xs in connected_components(high_mask):
        if len(xs) < min_component_px:
            continue
        x0 = float(xs.min()) / max(1, width - 1) - REGION_MARGIN_NORM
        x1 = float(xs.max()) / max(1, width - 1) + REGION_MARGIN_NORM
        y0 = float(ys.min()) / max(1, height - 1) - REGION_MARGIN_NORM
        y1 = float(ys.max()) / max(1, height - 1) + REGION_MARGIN_NORM
        x, y, w, h = clamp_box(x0, y0, x1 - x0, y1 - y0)
        area = w * h
        if area > 0.28:
            continue
        component_saliency = saliency[ys, xs]
        score = float(component_saliency.mean() * math.sqrt(len(xs)) * 1.15)
        aspect = w / max(h, 1e-6)
        kind = "rect" if aspect > 1.85 or aspect < 0.55 else "ellipse"
        components.append(
            {
                "box": (x, y, w, h),
                "kind": kind,
                "score": score,
                "level": "detail_region",
            }
        )

    components.sort(key=lambda item: item["score"], reverse=True)
    target_total = k if k is not None else min(AUTO_MAX_TOTAL_AOIS, max(AUTO_MIN_TOTAL_AOIS, len(components) + 1))
    target_foreground = max(1, min(MAX_TOTAL_AOIS - 1, target_total - 1))
    selected: list[dict[str, object]] = []
    for component in components:
        too_close = False
        for other in selected:
            overlap = box_iou(component["box"], other["box"])
            component_area = component["box"][2] * component["box"][3]
            other_area = other["box"][2] * other["box"][3]
            nested_detail = (
                component.get("level") == "detail_region"
                and component_area < other_area * 0.42
                and overlap < 0.55
            )
            if overlap > REGION_MAX_IOU and not nested_detail:
                too_close = True
                break
        if not too_close:
            selected.append(component)
        if len(selected) >= target_foreground:
            break

    fallback_boxes = [
        (0.34, 0.30, 0.32, 0.32),
        (0.12, 0.34, 0.26, 0.30),
        (0.62, 0.34, 0.26, 0.30),
        (0.34, 0.08, 0.32, 0.24),
    ]
    while len(selected) < target_foreground:
        box = fallback_boxes[len(selected) % len(fallback_boxes)]
        if all(box_iou(box, other["box"]) <= REGION_MAX_IOU for other in selected):
            selected.append({"box": box, "kind": "ellipse", "score": 0.0})
        else:
            break

    aois = [
        AOI(
            "BG",
            "non-AOI / background",
            "background",
            locked=True,
            method="non_aoi",
            source="predefined-region initialization leaves unassigned space outside AOIs",
        )
    ]
    for idx, component in enumerate(selected, start=1):
        x_norm, y_norm, w_norm, h_norm = component["box"]
        aois.append(
            AOI(
                f"AOI{idx}",
                f"operational region {idx}",
                str(component["kind"]),
                x_norm,
                y_norm,
                w_norm,
                h_norm,
                method=LIMITED_RADIUS_METHOD,
                source=(
                    "pre-gaze coarse saliency region; region bounds expanded by margin "
                    f"{REGION_MARGIN_NORM:.3f} of image extent"
                ),
            )
        )
    return normalize_aois(aois)


def image_oriented_defaults(image_path: Path, k: int | None) -> list[AOI]:
    return estimate_limited_radius_regions(image_path, k)


def legacy_seed_defaults(image_name: str) -> list[AOI] | None:
    if not LEGACY_SEEDS_PATH.exists():
        return None
    with LEGACY_SEEDS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    record = data.get("images", {}).get(image_name)
    if not record:
        return None
    aois = [AOI("BG", "background / rest", "background", locked=True)]
    for i, seed in enumerate(record.get("seeds", [])[:4], start=1):
        x = clamp01(float(seed.get("x", 0.5)) - 0.14)
        y = clamp01(float(seed.get("y", 0.5)) - 0.14)
        aois.append(AOI(f"AOI{i}", str(seed.get("label") or f"AOI {i}"), "ellipse", x, y, 0.28, 0.28))
    return normalize_aois(aois)


def get_image_aois(store: dict[str, Any], image_path: Path, k: int | None) -> list[AOI]:
    record = store["images"].get(image_path.name)
    if record:
        aois = aois_from_record(record)
        if aois:
            return aois
    return image_oriented_defaults(image_path, k or DEFAULT_TOTAL_AOIS)


def ensure_records(images: list[Path], k: int | None, overwrite: bool = False, from_legacy_seeds: bool = False) -> dict[str, Any]:
    store = load_store()
    for image_path in images:
        existing = store["images"].get(image_path.name)
        if existing and not overwrite:
            matches = fingerprint_matches(existing, image_path)
            if matches is False:
                archive_image_record(
                    store,
                    image_path.name,
                    existing,
                    "image file changed while keeping the same filename",
                )
            else:
                if matches is None:
                    store["images"][image_path.name] = with_image_metadata(existing, image_path)
                continue
        elif existing and overwrite:
            archive_image_record(store, image_path.name, existing, "overwritten during AOI initialization")
        aois = legacy_seed_defaults(image_path.name) if from_legacy_seeds else None
        store["images"][image_path.name] = with_image_metadata(
            aois_to_record(aois or image_oriented_defaults(image_path, k or DEFAULT_TOTAL_AOIS)),
            image_path,
        )
    save_store(store)
    return store


def is_limited_radius_aoi(aoi: AOI) -> bool:
    return aoi.kind in {"ellipse", "rect"} and aoi.method in {
        LIMITED_RADIUS_METHOD,
        "manual_adjusted_limited_radius",
    }


def rasterize_limited_radius_aois(width: int, height: int, aois: list[AOI]) -> np.ndarray:
    labels = np.zeros((height, width), dtype=np.uint8)
    yy, xx = np.indices((height, width), dtype=np.float32)
    candidates: list[tuple[np.ndarray, np.ndarray, int]] = []
    for idx, aoi in enumerate(aois[1:], start=1):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        x0 = round(aoi.x * width)
        y0 = round(aoi.y * height)
        x1 = round((aoi.x + aoi.w) * width)
        y1 = round((aoi.y + aoi.h) * height)
        if aoi.kind == "rect":
            draw.rectangle((x0, y0, x1, y1), fill=255)
        else:
            draw.ellipse((x0, y0, x1, y1), fill=255)
        cx = (aoi.x + aoi.w / 2.0) * width
        cy = (aoi.y + aoi.h / 2.0) * height
        center_distance = (xx - cx) ** 2 + (yy - cy) ** 2
        candidates.append((np.asarray(mask) > 0, center_distance, idx))
    if not candidates:
        return labels

    best_distance = np.full((height, width), np.inf, dtype=np.float32)
    for inside, center_distance, idx in candidates:
        update = inside & (center_distance < best_distance)
        labels[update] = idx
        best_distance[update] = center_distance[update]
    return labels


def rasterize_aois(width: int, height: int, aois: list[AOI]) -> np.ndarray:
    aois = normalize_aois(aois)
    foreground = aois[1:]
    if foreground and all(is_limited_radius_aoi(aoi) for aoi in foreground):
        return rasterize_limited_radius_aois(width, height, aois)

    labels = np.zeros((height, width), dtype=np.uint8)
    for idx, aoi in enumerate(aois[1:], start=1):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        x0 = round(aoi.x * width)
        y0 = round(aoi.y * height)
        x1 = round((aoi.x + aoi.w) * width)
        y1 = round((aoi.y + aoi.h) * height)
        if aoi.kind == "polygon" and aoi.points:
            points = [(round(x * width), round(y * height)) for x, y in aoi.points]
            draw.polygon(points, fill=255)
        elif aoi.kind == "rect":
            draw.rectangle((x0, y0, x1, y1), fill=255)
        else:
            draw.ellipse((x0, y0, x1, y1), fill=255)
        labels[np.asarray(mask) > 0] = idx
    return labels


def label_stats(labels: np.ndarray, aois: list[AOI]) -> list[dict[str, Any]]:
    height, width = labels.shape
    total = float(width * height)
    rows = []
    for idx, aoi in enumerate(normalize_aois(aois)):
        ys, xs = np.where(labels == idx)
        if len(xs) == 0:
            bbox = (0, 0, 0, 0)
        else:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        rows.append(
            {
                "aoi_id": aoi.aoi_id,
                "label": aoi.label,
                "kind": aoi.kind,
                "method": aoi.method,
                "source": aoi.source,
                "z_order": idx,
                "x_norm": "" if aoi.is_background else aoi.x,
                "y_norm": "" if aoi.is_background else aoi.y,
                "w_norm": "" if aoi.is_background else aoi.w,
                "h_norm": "" if aoi.is_background else aoi.h,
                "points_norm": "" if not aoi.points else json.dumps([[round(x, 6), round(y, 6)] for x, y in aoi.points]),
                "pixel_count": int(len(xs)),
                "area_fraction": len(xs) / total,
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_w": bbox[2],
                "bbox_h": bbox[3],
            }
        )
    return rows


def draw_preview(image: Image.Image, aois: list[AOI], labels: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    color_array = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=np.uint8)
    for idx in np.unique(labels):
        color = PALETTE[int(idx) % len(PALETTE)]
        color_array[labels == idx] = (*color, 55 if idx == 0 else 90)
    preview = Image.alpha_composite(base, Image.fromarray(color_array))

    edges = np.zeros_like(labels, dtype=bool)
    edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edges[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge_array = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=np.uint8)
    edge_array[edges] = (255, 255, 255, 235)
    preview = Image.alpha_composite(preview, Image.fromarray(edge_array))

    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    for idx, aoi in enumerate(normalize_aois(aois)):
        color = PALETTE[idx % len(PALETTE)]
        if aoi.is_background:
            draw.text((10, 10), f"{aoi.aoi_id}: {aoi.label}", fill=(255, 255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0, 255))
            continue
        x0 = aoi.x * image.width
        y0 = aoi.y * image.height
        x1 = (aoi.x + aoi.w) * image.width
        y1 = (aoi.y + aoi.h) * image.height
        if aoi.kind == "polygon" and aoi.points:
            points = [(x * image.width, y * image.height) for x, y in aoi.points]
            draw.line(points + [points[0]], fill=(*color, 255), width=4)
        elif aoi.kind == "rect":
            draw.rectangle((x0, y0, x1, y1), outline=(*color, 255), width=4)
        else:
            draw.ellipse((x0, y0, x1, y1), outline=(*color, 255), width=4)
        center_x = x0 + (x1 - x0) / 2.0
        center_y = y0 + (y1 - y0) / 2.0
        draw.ellipse((center_x - 4, center_y - 4, center_x + 4, center_y + 4), fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
        draw.text((x0 + 4, y0 + 4), f"{aoi.aoi_id}: {aoi.label}", fill=(255, 255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return preview.convert("RGB")


def write_outputs(images: list[Path], store: dict[str, Any], k: int | None) -> None:
    AOI_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_MAP_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_file",
        "image_width",
        "image_height",
        "aoi_id",
        "label",
        "kind",
        "method",
        "source",
        "z_order",
        "x_norm",
        "y_norm",
        "w_norm",
        "h_norm",
        "points_norm",
        "pixel_count",
        "area_fraction",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]
    summary_counts: Counter[int] = Counter()
    summary_methods: Counter[str] = Counter()
    with AOI_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for image_path in images:
            image = Image.open(image_path).convert("RGB")
            aois = get_image_aois(store, image_path, k)
            summary_counts[len(aois)] += 1
            summary_methods.update(aoi.method for aoi in aois)
            labels = rasterize_aois(image.width, image.height, aois)
            draw_preview(image, aois, labels).save(PREVIEW_DIR / f"{image_path.stem}_aois.png")
            Image.fromarray(labels + 1).save(LABEL_MAP_DIR / f"{image_path.stem}_labels.png")
            for row in label_stats(labels, aois):
                writer.writerow({"image_file": image_path.name, "image_width": image.width, "image_height": image.height, **row})
    with AOI_METHOD_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": 2,
                "method": LIMITED_RADIUS_METHOD,
                "interpretation": "AOIs are operational analysis regions, not objective semantic object boundaries.",
                "non_aoi_rule": "Pixels outside all foreground operational regions are assigned to non-AOI / background.",
                "overlap_rule": "Overlapping predefined regions are assigned by nearest region center; manual polygon AOIs use layer order.",
                "region_margin_norm": REGION_MARGIN_NORM,
                "region_min_size_norm": REGION_MIN_SIZE_NORM,
                "region_max_size_norm": REGION_MAX_SIZE_NORM,
                "region_max_iou_for_auto_selection": REGION_MAX_IOU,
                "default_total_categories": DEFAULT_TOTAL_AOIS,
                "auto_min_total_categories": AUTO_MIN_TOTAL_AOIS,
                "auto_max_total_categories": AUTO_MAX_TOTAL_AOIS,
                "image_count": len(images),
                "aoi_count_distribution": dict(sorted(summary_counts.items())),
                "method_counts": dict(sorted(summary_methods.items())),
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")


class ShapeEditor:
    HANDLE_SIZE = 7

    def __init__(self, images: list[Path], store: dict[str, Any], default_k: int | None, max_display: tuple[int, int]) -> None:
        if tk is None:
            raise RuntimeError("tkinter is not available in this Python installation")
        self.images = images
        self.store = store
        self.default_k = default_k
        self.max_display = max_display
        self.index = 0
        self.aois: list[AOI] = []
        self.selected = 1
        self.mode = "select"
        self.drag_action: str | None = None
        self.drag_start: tuple[float, float] | None = None
        self.original_box: tuple[float, float, float, float] | None = None
        self.original_points: list[tuple[float, float]] | None = None
        self.lasso_points: list[tuple[float, float]] = []
        self.lasso_line_id: int | None = None
        self.scale = 1.0
        self.display_size = (1, 1)
        self.tk_image: ImageTk.PhotoImage | None = None

        self.root = tk.Tk()
        self.root.title("AOI shape editor")
        self.header = tk.Label(self.root, anchor="w", justify="left")
        self.header.pack(fill="x")
        self.canvas = tk.Canvas(self.root, cursor="arrow", highlightthickness=0)
        self.canvas.pack()
        self.footer = tk.Label(
            self.root,
            anchor="w",
            justify="left",
            text=(
                "Drag shape/handles | E ellipse | R rect | L lasso/freehand | Del delete | Tab next AOI | "
                "] bring forward | [ send backward | T top | B bottom | "
                "A auto default | 2-9 auto count | +/- resize | Arrows nudge | S save | N/P navigate | Q quit"
            ),
        )
        self.footer.pack(fill="x")
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.root.bind("<Key>", self.on_key)
        self.load_current_image()

    def current_image_path(self) -> Path:
        return self.images[self.index]

    def load_current_image(self) -> None:
        image_path = self.current_image_path()
        self.aois = get_image_aois(self.store, image_path, self.default_k)
        self.selected = 1 if len(self.aois) > 1 else 0
        self.mode = "select"
        self.render()

    def save_current(self) -> None:
        self.aois = normalize_aois(self.aois)
        image_path = self.current_image_path()
        self.store["images"][image_path.name] = with_image_metadata(aois_to_record(self.aois), image_path)
        save_store(self.store)

    def norm_pos(self, event: Any) -> tuple[float, float]:
        return (
            clamp01(event.x / max(1, self.display_size[0])),
            clamp01(event.y / max(1, self.display_size[1])),
        )

    def display_box(self, aoi: AOI) -> tuple[float, float, float, float]:
        return (
            aoi.x * self.display_size[0],
            aoi.y * self.display_size[1],
            (aoi.x + aoi.w) * self.display_size[0],
            (aoi.y + aoi.h) * self.display_size[1],
        )

    def hit_test(self, x: float, y: float) -> tuple[int, str] | None:
        px = x * self.display_size[0]
        py = y * self.display_size[1]
        for idx in range(len(self.aois) - 1, 0, -1):
            aoi = self.aois[idx]
            x0, y0, x1, y1 = self.display_box(aoi)
            handles = {
                "nw": (x0, y0),
                "ne": (x1, y0),
                "sw": (x0, y1),
                "se": (x1, y1),
            }
            for name, (hx, hy) in handles.items():
                if abs(px - hx) <= self.HANDLE_SIZE + 3 and abs(py - hy) <= self.HANDLE_SIZE + 3:
                    return idx, name
            if x0 <= px <= x1 and y0 <= py <= y1:
                return idx, "move"
        return None

    def add_lasso_point(self, x: float, y: float) -> None:
        if self.lasso_points:
            last_x, last_y = self.lasso_points[-1]
            dx = (x - last_x) * self.display_size[0]
            dy = (y - last_y) * self.display_size[1]
            if math.hypot(dx, dy) < 3.0:
                return
        self.lasso_points.append((x, y))
        coords: list[float] = []
        for px, py in self.lasso_points:
            coords.extend([px * self.display_size[0], py * self.display_size[1]])
        if self.lasso_line_id is None:
            self.lasso_line_id = self.canvas.create_line(*coords, fill="white", width=3, smooth=True)
        else:
            self.canvas.coords(self.lasso_line_id, *coords)

    def add_polygon_aoi(self, points: list[tuple[float, float]]) -> None:
        if len(points) < 3 or len(self.aois) >= MAX_TOTAL_AOIS:
            return
        simplified = simplify_points(points)
        if len(simplified) < 3:
            simplified = points
        idx = len(self.aois)
        aoi = AOI(f"AOI{idx}", f"AOI {idx}", "polygon", points=simplified)
        update_polygon_bbox(aoi)
        self.aois.append(aoi)
        self.selected = idx

    def render(self) -> None:
        image_path = self.current_image_path()
        image = Image.open(image_path).convert("RGB")
        self.scale = min(self.max_display[0] / image.width, self.max_display[1] / image.height, 1.0)
        self.display_size = (max(1, round(image.width * self.scale)), max(1, round(image.height * self.scale)))
        display = image.resize(self.display_size, Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(display)
        self.canvas.configure(width=self.display_size[0], height=self.display_size[1])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        for idx, aoi in enumerate(normalize_aois(self.aois)):
            if aoi.is_background:
                continue
            x0, y0, x1, y1 = self.display_box(aoi)
            color = "#%02x%02x%02x" % PALETTE[idx % len(PALETTE)]
            width = 4 if idx == self.selected else 2
            if aoi.kind == "polygon" and aoi.points:
                coords: list[float] = []
                for px, py in aoi.points:
                    coords.extend([px * self.display_size[0], py * self.display_size[1]])
                self.canvas.create_polygon(*coords, outline=color, fill="", width=width)
            elif aoi.kind == "rect":
                self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=width)
            else:
                self.canvas.create_oval(x0, y0, x1, y1, outline=color, width=width)
            self.canvas.create_text(x0 + 6, y0 + 6, text=f"{aoi.aoi_id}: {aoi.label}", anchor="nw", fill="white", font=("TkDefaultFont", 10, "bold"))
            self.canvas.create_text(x0 + 7, y0 + 7, text=f"{aoi.aoi_id}: {aoi.label}", anchor="nw", fill="black", font=("TkDefaultFont", 10, "bold"))
            if idx == self.selected:
                for hx, hy in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
                    self.canvas.create_rectangle(hx - self.HANDLE_SIZE, hy - self.HANDLE_SIZE, hx + self.HANDLE_SIZE, hy + self.HANDLE_SIZE, fill=color, outline="white")
                center_x = x0 + (x1 - x0) / 2.0
                center_y = y0 + (y1 - y0) / 2.0
                self.canvas.create_oval(center_x - 4, center_y - 4, center_x + 4, center_y + 4, fill="white", outline="black")
        stack = "Stack bottom->top: " + " < ".join(
            f"{idx}:{aoi.aoi_id}" for idx, aoi in enumerate(normalize_aois(self.aois)) if not aoi.is_background
        )
        self.header.configure(
            text=(
                f"{self.index + 1}/{len(self.images)}  {image_path.name}  {image.width}x{image.height}  "
                f"mode={self.mode}  selected={self.aois[self.selected].aoi_id if self.aois else 'none'}\n"
                f"{stack}"
            )
        )

    def on_mouse_down(self, event: Any) -> None:
        x, y = self.norm_pos(event)
        self.drag_start = (x, y)
        self.original_points = None
        if self.mode == "lasso":
            if len(self.aois) >= MAX_TOTAL_AOIS:
                return
            self.drag_action = "lasso"
            self.lasso_points = []
            self.lasso_line_id = None
            self.add_lasso_point(x, y)
            return
        if self.mode in {"ellipse", "rect"}:
            if len(self.aois) >= MAX_TOTAL_AOIS:
                return
            kind = self.mode
            idx = len(self.aois)
            self.aois.append(AOI(f"AOI{idx}", f"AOI {idx}", kind, x, y, 0.01, 0.01))
            self.selected = idx
            self.drag_action = "se"
        else:
            hit = self.hit_test(x, y)
            if hit is None:
                return
            self.selected, self.drag_action = hit
        aoi = self.aois[self.selected]
        self.original_box = (aoi.x, aoi.y, aoi.w, aoi.h)
        self.original_points = list(aoi.points or [])
        self.render()

    def on_mouse_drag(self, event: Any) -> None:
        if self.drag_start is None or self.drag_action is None or self.selected <= 0:
            if self.drag_action == "lasso":
                x, y = self.norm_pos(event)
                self.add_lasso_point(x, y)
            return
        if self.drag_action == "lasso":
            x, y = self.norm_pos(event)
            self.add_lasso_point(x, y)
            return
        x, y = self.norm_pos(event)
        sx, sy = self.drag_start
        ox, oy, ow, oh = self.original_box or (0.0, 0.0, 0.1, 0.1)
        aoi = self.aois[self.selected]
        if self.drag_action == "move":
            aoi.x = clamp01(ox + x - sx)
            aoi.y = clamp01(oy + y - sy)
            aoi.x = min(aoi.x, 1.0 - aoi.w)
            aoi.y = min(aoi.y, 1.0 - aoi.h)
            if aoi.kind == "polygon" and self.original_points:
                dx = aoi.x - ox
                dy = aoi.y - oy
                aoi.points = [
                    (
                        min(max(0.0, px + dx), 1.0),
                        min(max(0.0, py + dy), 1.0),
                    )
                    for px, py in self.original_points
                ]
                update_polygon_bbox(aoi)
        else:
            x0, y0, x1, y1 = ox, oy, ox + ow, oy + oh
            if "n" in self.drag_action:
                y0 = y
            if "s" in self.drag_action:
                y1 = y
            if "w" in self.drag_action:
                x0 = x
            if "e" in self.drag_action:
                x1 = x
            left, right = sorted((clamp01(x0), clamp01(x1)))
            top, bottom = sorted((clamp01(y0), clamp01(y1)))
            aoi.x = left
            aoi.y = top
            aoi.w = max(0.01, right - left)
            aoi.h = max(0.01, bottom - top)
            if aoi.kind == "polygon" and self.original_points:
                scale_x = aoi.w / max(0.01, ow)
                scale_y = aoi.h / max(0.01, oh)
                aoi.points = [
                    (
                        clamp01(aoi.x + (px - ox) * scale_x),
                        clamp01(aoi.y + (py - oy) * scale_y),
                    )
                    for px, py in self.original_points
                ]
                update_polygon_bbox(aoi)
        self.render()

    def on_mouse_up(self, event: Any) -> None:
        if self.drag_action == "lasso":
            x, y = self.norm_pos(event)
            self.add_lasso_point(x, y)
            self.add_polygon_aoi(self.lasso_points)
            self.lasso_points = []
            self.lasso_line_id = None
            self.drag_start = None
            self.drag_action = None
            self.original_box = None
            self.original_points = None
            self.mode = "select"
            self.render()
            return
        if self.drag_action is not None:
            self.mark_selected_adjusted()
        self.drag_start = None
        self.drag_action = None
        self.original_box = None
        self.original_points = None
        if self.mode in {"ellipse", "rect"}:
            self.mode = "select"
        self.render()

    def selected_aoi(self) -> AOI | None:
        if 0 <= self.selected < len(self.aois):
            return self.aois[self.selected]
        return None

    def mark_selected_adjusted(self) -> None:
        aoi = self.selected_aoi()
        if aoi is None or aoi.is_background:
            return
        if aoi.method == LIMITED_RADIUS_METHOD:
            aoi.method = "manual_adjusted_limited_radius"
        elif not aoi.method:
            aoi.method = "manual"

    def resize_selected(self, factor: float) -> None:
        aoi = self.selected_aoi()
        if aoi is None or aoi.is_background:
            return
        cx, cy = aoi.x + aoi.w / 2.0, aoi.y + aoi.h / 2.0
        old_x, old_y, old_w, old_h = aoi.x, aoi.y, aoi.w, aoi.h
        aoi.w = max(0.01, min(1.0, aoi.w * factor))
        aoi.h = max(0.01, min(1.0, aoi.h * factor))
        aoi.x = min(max(0.0, cx - aoi.w / 2.0), 1.0 - aoi.w)
        aoi.y = min(max(0.0, cy - aoi.h / 2.0), 1.0 - aoi.h)
        if aoi.kind == "polygon" and aoi.points:
            scale_x = aoi.w / max(0.01, old_w)
            scale_y = aoi.h / max(0.01, old_h)
            aoi.points = [
                (
                    clamp01(aoi.x + (px - old_x) * scale_x),
                    clamp01(aoi.y + (py - old_y) * scale_y),
                )
                for px, py in aoi.points
            ]
            update_polygon_bbox(aoi)
        self.mark_selected_adjusted()
        self.render()

    def nudge_selected(self, dx: float, dy: float) -> None:
        aoi = self.selected_aoi()
        if aoi is None or aoi.is_background:
            return
        old_x, old_y = aoi.x, aoi.y
        aoi.x = min(max(0.0, aoi.x + dx), 1.0 - aoi.w)
        aoi.y = min(max(0.0, aoi.y + dy), 1.0 - aoi.h)
        if aoi.kind == "polygon" and aoi.points:
            actual_dx = aoi.x - old_x
            actual_dy = aoi.y - old_y
            aoi.points = [(clamp01(px + actual_dx), clamp01(py + actual_dy)) for px, py in aoi.points]
            update_polygon_bbox(aoi)
        self.mark_selected_adjusted()
        self.render()

    def move_selected_layer(self, target: str) -> None:
        if self.selected <= 0 or self.selected >= len(self.aois):
            return
        old_idx = self.selected
        if target == "forward":
            new_idx = min(len(self.aois) - 1, old_idx + 1)
        elif target == "backward":
            new_idx = max(1, old_idx - 1)
        elif target == "top":
            new_idx = len(self.aois) - 1
        elif target == "bottom":
            new_idx = 1
        else:
            return
        if new_idx == old_idx:
            return
        aoi = self.aois.pop(old_idx)
        self.aois.insert(new_idx, aoi)
        self.aois = normalize_aois(self.aois)
        self.selected = new_idx
        self.render()

    def on_key(self, event: Any) -> None:
        key = event.keysym.lower()
        char = event.char.lower() if event.char else ""
        if key in {str(value) for value in range(2, MAX_TOTAL_AOIS)}:
            self.aois = image_oriented_defaults(self.current_image_path(), int(key))
            self.selected = 1 if len(self.aois) > 1 else 0
            self.render()
        elif char == "a":
            self.aois = image_oriented_defaults(self.current_image_path(), self.default_k or DEFAULT_TOTAL_AOIS)
            self.selected = 1 if len(self.aois) > 1 else 0
            self.render()
        elif char == "e":
            self.mode = "ellipse"
            self.render()
        elif char == "r":
            self.mode = "rect"
            self.render()
        elif char == "l":
            self.mode = "lasso"
            self.render()
        elif key in {"delete", "backspace"} and self.selected > 0:
            del self.aois[self.selected]
            self.aois = normalize_aois(self.aois)
            self.selected = min(self.selected, len(self.aois) - 1)
            self.render()
        elif key == "tab":
            self.selected = 1 if self.selected >= len(self.aois) - 1 else self.selected + 1
            self.render()
        elif key == "plus" or char == "+":
            self.resize_selected(1.04)
        elif key == "minus" or char == "-":
            self.resize_selected(0.96)
        elif char == "]":
            self.move_selected_layer("forward")
        elif char == "[":
            self.move_selected_layer("backward")
        elif char == "t":
            self.move_selected_layer("top")
        elif char == "b":
            self.move_selected_layer("bottom")
        elif key == "left":
            self.nudge_selected(-0.005, 0.0)
        elif key == "right":
            self.nudge_selected(0.005, 0.0)
        elif key == "up":
            self.nudge_selected(0.0, -0.005)
        elif key == "down":
            self.nudge_selected(0.0, 0.005)
        elif char == "s":
            self.save_current()
        elif char == "n":
            self.save_current()
            self.index = min(len(self.images) - 1, self.index + 1)
            self.load_current_image()
        elif char == "p":
            self.save_current()
            self.index = max(0, self.index - 1)
            self.load_current_image()
        elif char == "q":
            self.save_current()
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create coarse, editable shape AOIs with one optional background/remainder AOI.")
    parser.add_argument(
        "command",
        nargs="?",
        default="edit",
        choices=["init", "edit", "batch"],
        help="init AOIs, edit AOIs, or regenerate AOI outputs. Defaults to edit.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        choices=range(2, MAX_TOTAL_AOIS + 1),
        help=f"default total AOI count, including background. Defaults to {DEFAULT_TOTAL_AOIS}; each image can still vary.",
    )
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--overwrite", action="store_true", help="replace existing saved AOI shapes during init")
    parser.add_argument("--from-legacy-seeds", action="store_true", help="convert existing seeds.json into ellipse AOIs")
    parser.add_argument("--max-width", type=int, default=1150, help="maximum editor image width")
    parser.add_argument("--max-height", type=int, default=780, help="maximum editor image height")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = list_images(args.image_dir)
    if not images:
        raise SystemExit(f"No supported images found in {args.image_dir}")

    if args.command == "init":
        store = ensure_records(images, args.k, overwrite=args.overwrite, from_legacy_seeds=args.from_legacy_seeds)
        write_outputs(images, store, args.k)
    elif args.command == "batch":
        store = ensure_records(images, args.k, overwrite=False)
        write_outputs(images, store, args.k)
    elif args.command == "edit":
        store = ensure_records(images, args.k, overwrite=False)
        editor = ShapeEditor(images, store, args.k, (args.max_width, args.max_height))
        editor.run()
        write_outputs(images, load_store(), args.k)


if __name__ == "__main__":
    main()
