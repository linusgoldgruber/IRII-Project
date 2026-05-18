from __future__ import annotations

import argparse
import csv
import io
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


DATASET_API = "https://datasets-server.huggingface.co/rows"
SEARCH_API = "https://datasets-server.huggingface.co/search"
DATASET_NAME = "asahi417/wikiart-all"
CONFIG = "default"
SPLIT = "test"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "input" / "candidate_images" / "wikiart_easy_aoi"

# Row ranges inferred from the published asahi417/wikiart-all group counts.
# These overlap the current study categories available in this mirror.
GROUP_RANGES = {
    "baroque": (0, 5483),
    "expressionism": (5483, 6369),
    "impressionism": (11852, 9895),
    "realism": (21761, 12460),
    "rococo": (34221, 3262),
    "romanticism": (37483, 12993),
    "surrealism": (50476, 4922),
    "symbolism": (55398, 7663),
}

DEFAULT_GROUPS = ("expressionism", "surrealism", "symbolism")

EASY_TAGS = {
    "bird",
    "boat",
    "book",
    "bottle",
    "building",
    "chair",
    "cloud",
    "face",
    "figure",
    "flower",
    "forehead",
    "fruit",
    "hand",
    "head",
    "horse",
    "house",
    "human",
    "leaf",
    "moon",
    "mountain",
    "person",
    "picture frame",
    "plant",
    "portrait",
    "sailing ship",
    "ship",
    "sun",
    "table",
    "tree",
    "vehicle",
    "watercraft",
    "window",
}

EASY_GENRES = {
    "abstract",
    "animal painting",
    "figurative",
    "flower painting",
    "landscape",
    "portrait",
    "self-portrait",
    "still life",
    "symbolic painting",
}

ABSTRACT_SPARSE_TITLE_TERMS = {
    "abstract",
    "composition",
    "figure",
    "flower",
    "flowers",
    "head",
    "portrait",
    "self portrait",
    "still life",
    "vase",
}

ABSTRACT_SPARSE_TAG_TERMS = {
    "face",
    "figure",
    "flower",
    "forehead",
    "head",
    "human",
    "leaf",
    "plant",
    "portrait",
    "still life",
    "sunflower",
    "vase",
}

ABSTRACT_SPARSE_STYLE_TERMS = {
    "abstract expressionism",
    "expressionism",
    "surrealism",
    "symbolism",
}

ABSTRACT_SPARSE_GENRES = {
    "abstract",
    "figurative",
    "flower painting",
    "portrait",
    "self-portrait",
    "still life",
    "symbolic painting",
}

ABSTRACT_SPARSE_PENALTY_TERMS = {
    "alps",
    "architecture",
    "boat",
    "boats-and-ships",
    "building",
    "cityscape",
    "cloud",
    "crowd",
    "forest",
    "forests-and-trees",
    "harbor",
    "house",
    "landscape",
    "marina",
    "mountain",
    "port",
    "sailing ship",
    "scene",
    "ship",
    "tree",
    "vehicle",
    "watercraft",
}

AVOID_GENRES = {
    "battle painting",
    "cityscape",
    "genre painting",
    "history painting",
    "religious painting",
}


@dataclass
class Candidate:
    row_idx: int
    group: str
    title: str
    artist: str
    url_slug: str
    genres: list[str]
    styles: list[str]
    tags: list[str]
    width: int
    height: int
    image_url: str
    score: float
    filename: str = ""


def request_json(url: str, params: dict[str, str | int], retries: int = 5) -> dict:
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(full_url, headers={"User-Agent": "irii-study/1.0"})
            with urllib.request.urlopen(req, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                time.sleep(12.0 * (attempt + 1))
            else:
                time.sleep(1.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} attempts: {full_url}") from last_error


def fetch_rows(offset: int, length: int) -> list[dict]:
    payload = request_json(
        DATASET_API,
        {
            "dataset": DATASET_NAME,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        },
    )
    return payload.get("rows", [])


def search_rows(query: str, offset: int, length: int) -> list[dict]:
    payload = request_json(
        SEARCH_API,
        {
            "dataset": DATASET_NAME,
            "config": CONFIG,
            "split": SPLIT,
            "query": query,
            "offset": offset,
            "length": length,
        },
    )
    return payload.get("rows", [])


def clean_slug(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:70] or fallback


def score_row(row_idx: int, row: dict, profile: str = "easy_aoi") -> float:
    genres = [str(v).lower() for v in row.get("genres") or []]
    styles = [str(v).lower() for v in row.get("styles") or []]
    tags = [str(v).lower() for v in row.get("tags") or []]
    title = str(row.get("title") or "").lower()
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)

    score = 0.0
    if 280 <= width <= 900 and 280 <= height <= 900:
        score += 1.0
    if 0.55 <= (width / max(height, 1)) <= 1.8:
        score += 0.8
    if 1 <= len(genres) <= 2:
        score += 1.0
    if 1 <= len(tags) <= 6:
        score += 1.2
    if not tags:
        score += 0.2

    score += 1.2 * sum(genre in EASY_GENRES for genre in genres)
    score += 0.7 * sum(any(key in tag for key in EASY_TAGS) for tag in tags)
    score += 0.4 * sum(key in title for key in EASY_TAGS)
    score -= 1.2 * sum(genre in AVOID_GENRES for genre in genres)
    score -= 0.18 * max(0, len(tags) - 6)
    score -= 0.25 * max(0, len(genres) - 2)

    # Prefer rows beyond the earliest items because the current bank appears to
    # have been assembled from early category slices.
    group_start = min(start for start, _count in GROUP_RANGES.values())
    score += min(1.5, max(0.0, (row_idx - group_start) / 20000.0))
    if profile == "abstract_sparse":
        text_bits = genres + styles + tags + [title]
        score += 2.6 * sum(genre in ABSTRACT_SPARSE_GENRES for genre in genres)
        score += 1.7 * sum(style in ABSTRACT_SPARSE_STYLE_TERMS for style in styles)
        score += 1.4 * sum(term in title for term in ABSTRACT_SPARSE_TITLE_TERMS)
        score += 0.9 * sum(any(term in tag for term in ABSTRACT_SPARSE_TAG_TERMS) for tag in tags)
        score -= 1.3 * sum(any(term in bit for term in ABSTRACT_SPARSE_PENALTY_TERMS) for bit in text_bits)
        score -= 0.55 * max(0, len(tags) - 5)
        score -= 0.8 * max(0, len(genres) - 2)
        if width > height * 1.55:
            score -= 1.8
        if len(tags) <= 5:
            score += 1.3
        if not tags:
            score += 0.8
    return score


def row_to_candidate(row_record: dict, profile: str = "easy_aoi") -> Candidate | None:
    row_idx = int(row_record["row_idx"])
    row = row_record.get("row") or {}
    image = row.get("image") or {}
    image_url = image.get("src")
    if not image_url:
        return None
    width = int(row.get("width") or image.get("width") or 0)
    height = int(row.get("height") or image.get("height") or 0)
    if width < 250 or height < 250:
        return None
    return Candidate(
        row_idx=row_idx,
        group=str(row.get("group") or ""),
        title=str(row.get("title") or ""),
        artist=str(row.get("artistName") or ""),
        url_slug=str(row.get("url") or ""),
        genres=[str(v) for v in row.get("genres") or []],
        styles=[str(v) for v in row.get("styles") or []],
        tags=[str(v) for v in row.get("tags") or []],
        width=width,
        height=height,
        image_url=str(image_url),
        score=score_row(row_idx, row, profile),
    )


def existing_manifest_keys(root: Path) -> set[str]:
    keys: set[str] = set()
    for manifest_path in (root / "input" / "candidate_images").glob("*/manifest.csv"):
        try:
            with manifest_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    keys.add(row.get("url_slug", ""))
                    keys.add(row.get("row_idx", ""))
        except OSError:
            continue
    return {key for key in keys if key}


def collect_candidates(
    groups: list[str],
    per_group: int,
    scan_per_group: int,
    page_size: int,
    scan_offset: int,
    request_delay_s: float,
    profile: str,
    exclude_keys: set[str],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for group in groups:
        start, count = GROUP_RANGES[group]
        group_rows: list[Candidate] = []
        first = start + min(scan_offset, max(0, count - 1))
        limit = min(scan_per_group, max(0, count - scan_offset))
        for offset in range(first, first + limit, page_size):
            rows = fetch_rows(offset, min(page_size, first + limit - offset))
            if request_delay_s > 0:
                time.sleep(request_delay_s)
            for row_record in rows:
                candidate = row_to_candidate(row_record, profile)
                if candidate is not None:
                    if str(candidate.row_idx) in exclude_keys or candidate.url_slug in exclude_keys:
                        continue
                    group_rows.append(candidate)
        group_rows.sort(key=lambda c: c.score, reverse=True)
        candidates.extend(group_rows[:per_group])
        print(f"{group}: scanned {len(group_rows)} usable rows, selected {min(per_group, len(group_rows))}")
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def collect_search_candidates(
    queries: list[str],
    per_query: int,
    scan_per_query: int,
    page_size: int,
    request_delay_s: float,
    profile: str,
    exclude_keys: set[str],
) -> list[Candidate]:
    candidates_by_key: dict[str, Candidate] = {}
    for query in queries:
        query_rows: list[Candidate] = []
        for offset in range(0, scan_per_query, page_size):
            rows = search_rows(query, offset, min(page_size, scan_per_query - offset))
            if request_delay_s > 0:
                time.sleep(request_delay_s)
            for row_record in rows:
                candidate = row_to_candidate(row_record, profile)
                if candidate is None:
                    continue
                if str(candidate.row_idx) in exclude_keys or candidate.url_slug in exclude_keys:
                    continue
                query_rows.append(candidate)
        query_rows.sort(key=lambda c: c.score, reverse=True)
        for candidate in query_rows[:per_query]:
            key = candidate.url_slug or str(candidate.row_idx)
            current = candidates_by_key.get(key)
            if current is None or candidate.score > current.score:
                candidates_by_key[key] = candidate
        print(f"search {query!r}: scanned {len(query_rows)} usable rows, selected {min(per_query, len(query_rows))}")
    candidates = list(candidates_by_key.values())
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def download_image(candidate: Candidate, output_dir: Path, index: int) -> Path:
    slug = clean_slug(candidate.url_slug or candidate.title, f"row_{candidate.row_idx}")
    filename = f"candidate_{index:03d}_{candidate.group}_{slug}.jpg"
    path = output_dir / "images" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(candidate.image_url, headers={"User-Agent": "irii-study/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        raw = response.read()
    with Image.open(io.BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.save(path, quality=92, optimize=True)
    candidate.filename = filename
    return path


def write_manifest(candidates: list[Candidate], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "manifest.csv"
    fieldnames = [
        "filename",
        "score",
        "row_idx",
        "group",
        "title",
        "artist",
        "url_slug",
        "genres",
        "styles",
        "tags",
        "width",
        "height",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "filename": candidate.filename,
                    "score": round(candidate.score, 3),
                    "row_idx": candidate.row_idx,
                    "group": candidate.group,
                    "title": candidate.title,
                    "artist": candidate.artist,
                    "url_slug": candidate.url_slug,
                    "genres": "; ".join(candidate.genres),
                    "styles": "; ".join(candidate.styles),
                    "tags": "; ".join(candidate.tags),
                    "width": candidate.width,
                    "height": candidate.height,
                }
            )


def make_contact_sheet(candidates: list[Candidate], output_dir: Path, columns: int = 5) -> None:
    image_dir = output_dir / "images"
    thumb_w, thumb_h = 220, 170
    label_h = 48
    rows = (len(candidates) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for i, candidate in enumerate(candidates):
        x = (i % columns) * thumb_w
        y = (i // columns) * (thumb_h + label_h)
        path = image_dir / candidate.filename
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            px = x + (thumb_w - image.width) // 2
            py = y + (thumb_h - image.height) // 2
            sheet.paste(image, (px, py))
        label = f"{i + 1:02d} {candidate.group} | {candidate.title[:36]}"
        draw.text((x + 6, y + thumb_h + 4), label, fill="black", font=font)
        draw.text((x + 6, y + thumb_h + 22), f"score {candidate.score:.2f} row {candidate.row_idx}", fill="black", font=font)

    sheet.save(output_dir / "contact_sheet.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WikiArt candidate images likely to have easy AOIs.")
    parser.add_argument("--per-group", type=int, default=14)
    parser.add_argument("--scan-per-group", type=int, default=550)
    parser.add_argument("--scan-offset", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--request-delay-s", type=float, default=0.5)
    parser.add_argument("--profile", choices=["easy_aoi", "abstract_sparse"], default="easy_aoi")
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument("--search-queries", nargs="+", default=None)
    parser.add_argument("--per-query", type=int, default=12)
    parser.add_argument("--scan-per-query", type=int, default=500)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        choices=sorted(GROUP_RANGES),
        help="WikiArt movement groups to scan.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    exclude_keys = set() if args.allow_duplicates else existing_manifest_keys(PROJECT_ROOT)
    if args.search_queries:
        candidates = collect_search_candidates(
            args.search_queries,
            args.per_query,
            args.scan_per_query,
            args.page_size,
            args.request_delay_s,
            args.profile,
            exclude_keys,
        )
    else:
        candidates = collect_candidates(
            args.groups,
            args.per_group,
            args.scan_per_group,
            args.page_size,
            args.scan_offset,
            args.request_delay_s,
            args.profile,
            exclude_keys,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[Candidate] = []
    for index, candidate in enumerate(candidates, start=1):
        try:
            download_image(candidate, args.output_dir, index)
        except Exception as exc:
            print(f"skip row {candidate.row_idx}: {exc}")
            continue
        selected.append(candidate)

    write_manifest(selected, args.output_dir)
    if selected:
        make_contact_sheet(selected, args.output_dir)
    print(f"downloaded {len(selected)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
