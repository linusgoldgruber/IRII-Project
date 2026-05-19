from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STIMULI_CSV = PROJECT_ROOT / "input" / "main" / "stimuli.csv"
IMAGE_DIR = PROJECT_ROOT / "input" / "main" / "images"
BACKUP_DIR = PROJECT_ROOT / "input" / "main" / "stimuli_backups"
AUDIT_ROOT = PROJECT_ROOT / "output" / "prompt_audit"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

PROMPT_COLUMNS = [
    "condition_congruent",
    "condition_semi_congruent",
    "condition_incongruent",
    "condition_congruent_en",
    "condition_semi_congruent_en",
    "condition_incongruent_en",
]

SUGGESTION_COLUMNS = [
    "approved",
    "stimulus_id",
    "image_file",
    "review_status",
    "confidence",
    "major_issue_count",
    "metadata_artist",
    "metadata_title",
    "metadata_style",
    "high_confidence_visible",
    "medium_confidence_likely",
    "do_not_mention",
    "current_condition_congruent",
    "current_condition_semi_congruent",
    "current_condition_incongruent",
    "current_condition_congruent_en",
    "current_condition_semi_congruent_en",
    "current_condition_incongruent_en",
    "suggested_condition_congruent",
    "suggested_condition_semi_congruent",
    "suggested_condition_incongruent",
    "suggested_condition_congruent_en",
    "suggested_condition_semi_congruent_en",
    "suggested_condition_incongruent_en",
    "audit_notes",
]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def backup_stimuli(reason: str) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_path = BACKUP_DIR / f"stimuli_{reason}_{timestamp}.csv"
    shutil.copy2(STIMULI_CSV, backup_path)
    return backup_path


def load_metadata() -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    manifest_paths = [
        PROJECT_ROOT / "input" / "candidate_images" / "wikiart_easy_aoi" / "manifest.csv",
        PROJECT_ROOT / "input" / "candidate_images" / "wikiart_easy_aoi_more" / "manifest.csv",
        PROJECT_ROOT / "input" / "candidate_images" / "wikiart_abstract_sparse" / "manifest.csv",
        PROJECT_ROOT / "input" / "practice" / "images" / "wikiart_practice" / "manifest.csv",
    ]
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                filename = (row.get("filename") or row.get("image_file") or "").strip()
                if filename:
                    metadata[filename] = {
                        "title": (row.get("title") or "").strip(),
                        "artist": (row.get("artist") or "").strip(),
                        "url_slug": (row.get("url_slug") or "").strip(),
                        "style": (row.get("style") or row.get("group") or "").strip(),
                        "genre": (row.get("genre") or row.get("genres") or "").strip(),
                        "tags": (row.get("tags") or "").strip(),
                    }
    return metadata


def metadata_for_row(row: dict[str, str], metadata: dict[str, dict[str, str]]) -> dict[str, str]:
    original_filename = (row.get("original_filename") or "").strip()
    result = dict(metadata.get(original_filename, {}))
    if not result.get("style"):
        result["style"] = (row.get("original_style_folder") or "").strip()
    if not result.get("title"):
        title = re.sub(r"^(candidate_\d+_|practice_wikiart_\d+_)", "", original_filename)
        title = re.sub(r"\.(jpg|jpeg|png|ppm)$", "", title, flags=re.I)
        title = re.sub(r"_(\d{4}|c_\d{4}|before_\d{4})$", "", title)
        result["title"] = title.replace("_", " ").strip()
    if not result.get("artist"):
        result["artist"] = ""
    return result


def encode_image_data_url(image_path: Path, max_side: int = 1200) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def build_model_prompt(row: dict[str, str], meta: dict[str, str]) -> str:
    current = "\n".join(f"{column}: {row.get(column, '')}" for column in PROMPT_COLUMNS)
    return f"""
You are auditing prompts for an eye-tracking study with abstract, surrealist, expressionist, and related artworks.

Goal:
Participants read a description before seeing the image. The description should guide visual search and imagination.
We need a plausible range from accurate to inaccurate. Even the incongruent description must still make the participant search for mentioned visual content, but it must be inaccurate enough to create low perceived congruency.

Use the attached image and metadata. Do not over-trust metadata if it conflicts with visible content.

Metadata:
- stimulus_id: {row.get("stimulus_id", "")}
- image_file: {row.get("image_file", "")}
- original_filename: {row.get("original_filename", "")}
- original_relative_path: {row.get("original_relative_path", "")}
- style_folder: {row.get("original_style_folder", "")}
- artist: {meta.get("artist", "")}
- title: {meta.get("title", "")}
- metadata_style: {meta.get("style", "")}
- genre: {meta.get("genre", "")}
- tags: {meta.get("tags", "")}

Current prompts:
{current}

Do this in two passes:

Pass 1: Objective inventory only.
- high_confidence_visible: colors, shapes, positions, number of salient objects, and clearly identifiable objects.
- medium_confidence_likely: plausible interpretations, but do not use these as object claims in final prompts.
- do_not_mention: things that are uncertain, absent, or too interpretive.

Pass 2: Rewrite the prompts.
Rules:
- Congruent: accurate, concrete, gaze-guiding, not overly detailed.
- Semi-congruent: plausible and partially matching, same broad visual domain, but meaningfully inaccurate.
- Incongruent: plausible for the artwork style, but clearly inaccurate in 2-3 major visual features. Do not describe a totally different scene.
- For abstract/surreal images, prefer color/shape/position/form language over object names unless the object is unmistakable.
- German and English must be semantically parallel.
- Keep prompts concise: 1-2 sentences each.
- Avoid tiny wording-only differences between congruency levels.

Return only valid JSON with this exact structure:
{{
  "review_status": "ok" | "needs_human_review",
  "confidence": "high" | "medium" | "low",
  "major_issue_count": 0,
  "inventory": {{
    "high_confidence_visible": ["..."],
    "medium_confidence_likely": ["..."],
    "do_not_mention": ["..."]
  }},
  "suggested": {{
    "condition_congruent": "...",
    "condition_semi_congruent": "...",
    "condition_incongruent": "...",
    "condition_congruent_en": "...",
    "condition_semi_congruent_en": "...",
    "condition_incongruent_en": "..."
  }},
  "audit_notes": ["..."]
}}
""".strip()


def call_openai(prompt: str, image_data_url: str, model: str, api_key: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = extract_output_text(body)
    return parse_json_text(text)


def extract_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "\n".join(chunks).strip()


def parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def audit_to_suggestion_row(row: dict[str, str], meta: dict[str, str], audit: dict[str, Any]) -> dict[str, str]:
    inventory = audit.get("inventory") or {}
    suggested = audit.get("suggested") or {}
    notes = audit.get("audit_notes") or []
    result: dict[str, str] = {
        "approved": "",
        "stimulus_id": row.get("stimulus_id", ""),
        "image_file": row.get("image_file", ""),
        "review_status": str(audit.get("review_status", "needs_human_review")),
        "confidence": str(audit.get("confidence", "low")),
        "major_issue_count": str(audit.get("major_issue_count", "")),
        "metadata_artist": meta.get("artist", ""),
        "metadata_title": meta.get("title", ""),
        "metadata_style": meta.get("style", ""),
        "high_confidence_visible": "; ".join(map(str, inventory.get("high_confidence_visible", []))),
        "medium_confidence_likely": "; ".join(map(str, inventory.get("medium_confidence_likely", []))),
        "do_not_mention": "; ".join(map(str, inventory.get("do_not_mention", []))),
        "audit_notes": "; ".join(map(str, notes)),
    }
    for column in PROMPT_COLUMNS:
        result[f"current_{column}"] = row.get(column, "")
        result[f"suggested_{column}"] = str(suggested.get(column, row.get(column, "")))
    return result


def offline_suggestion_row(row: dict[str, str], meta: dict[str, str]) -> dict[str, str]:
    audit = {
        "review_status": "not_audited",
        "confidence": "",
        "major_issue_count": "",
        "inventory": {
            "high_confidence_visible": [],
            "medium_confidence_likely": [],
            "do_not_mention": ["No model audit was run for this row."],
        },
        "suggested": {column: row.get(column, "") for column in PROMPT_COLUMNS},
        "audit_notes": ["Offline packet only. Run with OPENAI_API_KEY for automated suggestions."],
    }
    return audit_to_suggestion_row(row, meta, audit)


def write_review_html(out_path: Path, suggestion_rows: list[dict[str, str]]) -> None:
    rows_html: list[str] = []
    for item in suggestion_rows:
        image_rel = Path("..") / ".." / "input" / "main" / "images" / item["image_file"]
        prompt_cells = []
        for prefix, title in (("current", "Current"), ("suggested", "Suggested")):
            values = "".join(
                f"<dt>{html.escape(column.replace('condition_', ''))}</dt>"
                f"<dd>{html.escape(item.get(f'{prefix}_{column}', ''))}</dd>"
                for column in PROMPT_COLUMNS
            )
            prompt_cells.append(f"<section><h4>{title}</h4><dl>{values}</dl></section>")
        rows_html.append(
            f"""
            <article class="card">
              <div class="media">
                <img src="{html.escape(str(image_rel))}" alt="{html.escape(item['image_file'])}">
              </div>
              <div class="content">
                <h2>{html.escape(item['stimulus_id'])} - {html.escape(item['image_file'])}</h2>
                <p><b>Status:</b> {html.escape(item['review_status'])} |
                   <b>Confidence:</b> {html.escape(item['confidence'])} |
                   <b>Issues:</b> {html.escape(item['major_issue_count'])}</p>
                <p><b>Metadata:</b> {html.escape(item['metadata_artist'])} - {html.escape(item['metadata_title'])}
                   ({html.escape(item['metadata_style'])})</p>
                <p><b>High confidence visible:</b> {html.escape(item['high_confidence_visible'])}</p>
                <p><b>Medium confidence likely:</b> {html.escape(item['medium_confidence_likely'])}</p>
                <p><b>Do not mention:</b> {html.escape(item['do_not_mention'])}</p>
                <p><b>Audit notes:</b> {html.escape(item['audit_notes'])}</p>
                <div class="prompts">{''.join(prompt_cells)}</div>
              </div>
            </article>
            """
        )
    out_path.write_text(
        f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stimulus Prompt Audit</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; background: #f7f7f7; color: #111; }}
.card {{ display: grid; grid-template-columns: minmax(260px, 34vw) 1fr; gap: 20px; background: white; border: 1px solid #ddd; border-radius: 6px; padding: 16px; margin-bottom: 18px; }}
.media img {{ width: 100%; max-height: 620px; object-fit: contain; background: #222; }}
h2 {{ margin-top: 0; }}
.prompts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
dt {{ font-weight: 700; margin-top: 10px; }}
dd {{ margin-left: 0; white-space: pre-wrap; }}
@media (max-width: 900px) {{ .card, .prompts {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Stimulus Prompt Audit</h1>
<p>Approve suggested rows in <code>suggestions.csv</code> by setting <code>approved</code> to <code>yes</code>, then apply with the script.</p>
{''.join(rows_html)}
</body>
</html>
""".strip(),
        encoding="utf-8",
    )


def run_audit(args: argparse.Namespace) -> int:
    rows, _fieldnames = read_csv(STIMULI_CSV)
    metadata = load_metadata()
    selected = rows[args.start - 1 :]
    if args.limit:
        selected = selected[: args.limit]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = AUDIT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    suggestions_path = out_dir / "suggestions.csv"
    audit_jsonl_path = out_dir / "audit.jsonl"
    review_html_path = out_dir / "review.html"

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    use_model = bool(api_key) and not args.offline
    if not use_model:
        print("No model audit will run. Creating offline review packet only.", file=sys.stderr)

    suggestion_rows: list[dict[str, str]] = []
    with audit_jsonl_path.open("w", encoding="utf-8") as jsonl:
        for idx, row in enumerate(selected, start=args.start):
            meta = metadata_for_row(row, metadata)
            image_path = IMAGE_DIR / (row.get("image_file") or "")
            print(f"[{idx}/{len(rows)}] {row.get('image_file', '')}")
            if use_model and image_path.exists():
                prompt = build_model_prompt(row, meta)
                image_data_url = encode_image_data_url(image_path)
                try:
                    audit = call_openai(prompt, image_data_url, args.model, api_key)
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                    audit = {
                        "review_status": "needs_human_review",
                        "confidence": "low",
                        "major_issue_count": "",
                        "inventory": {
                            "high_confidence_visible": [],
                            "medium_confidence_likely": [],
                            "do_not_mention": ["Model call failed."],
                        },
                        "suggested": {column: row.get(column, "") for column in PROMPT_COLUMNS},
                        "audit_notes": [f"Model call failed: {exc}"],
                    }
                if args.sleep_s:
                    time.sleep(args.sleep_s)
            else:
                audit = {
                    "review_status": "needs_human_review" if not image_path.exists() else "not_audited",
                    "confidence": "",
                    "major_issue_count": "",
                    "inventory": {
                        "high_confidence_visible": [],
                        "medium_confidence_likely": [],
                        "do_not_mention": ["No model audit was run for this row."],
                    },
                    "suggested": {column: row.get(column, "") for column in PROMPT_COLUMNS},
                    "audit_notes": ["Offline packet only. Run with OPENAI_API_KEY for automated suggestions."],
                }
            jsonl.write(json.dumps({"row": row, "metadata": meta, "audit": audit}, ensure_ascii=False) + "\n")
            suggestion_rows.append(audit_to_suggestion_row(row, meta, audit))

    write_csv(suggestions_path, suggestion_rows, SUGGESTION_COLUMNS)
    write_review_html(review_html_path, suggestion_rows)
    print(f"Wrote {suggestions_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {review_html_path.relative_to(PROJECT_ROOT)}")
    return 0


def apply_approved(args: argparse.Namespace) -> int:
    suggestions_path = Path(args.apply_approved).resolve()
    suggestion_rows, _suggestion_fields = read_csv(suggestions_path)
    stimuli_rows, stimuli_fields = read_csv(STIMULI_CSV)
    rows_by_id = {row.get("stimulus_id", ""): row for row in stimuli_rows}
    approved = [
        row for row in suggestion_rows if row.get("approved", "").strip().lower() in {"y", "yes", "true", "1", "approved"}
    ]
    if not approved:
        print("No approved rows found. Set approved=yes in suggestions.csv first.")
        return 0
    backup_path = backup_stimuli("before_applying_prompt_audit")
    changed = 0
    for suggestion in approved:
        stimulus_id = suggestion.get("stimulus_id", "")
        target = rows_by_id.get(stimulus_id)
        if target is None:
            continue
        for column in PROMPT_COLUMNS:
            value = suggestion.get(f"suggested_{column}", "")
            if value:
                target[column] = value
        target["notes"] = append_note(target.get("notes", ""), "prompt_audit_reviewed")
        target["needs_human_review"] = "False"
        changed += 1
    write_csv(STIMULI_CSV, stimuli_rows, stimuli_fields)
    print(f"Applied {changed} approved rows.")
    print(f"Backup: {backup_path.relative_to(PROJECT_ROOT)}")
    return 0


def append_note(existing: str, note: str) -> str:
    if note in existing:
        return existing
    return f"{existing}; {note}" if existing else note


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit image-description prompts and create approval-based prompt suggestions."
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--start", type=int, default=1, help="1-based first row to audit.")
    parser.add_argument("--limit", type=int, default=0, help="Number of rows to audit. 0 means all from start.")
    parser.add_argument("--sleep-s", type=float, default=0.2, help="Pause between API calls.")
    parser.add_argument("--offline", action="store_true", help="Create review packet without calling a model.")
    parser.add_argument("--apply-approved", help="Path to a suggestions.csv with approved=yes rows to apply.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.apply_approved:
        return apply_approved(args)
    return run_audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
