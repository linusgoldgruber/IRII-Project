from __future__ import annotations

import csv
import math
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from psychopy import core, event, gui, visual
import study_config as cfg

N_TRIALS_REAL = cfg.N_TRIALS_REAL
N_TRIALS_RG = cfg.N_TRIALS_RG
N_TRIALS_TEST = cfg.N_TRIALS_TEST
N_PRACTICE_TRIALS = cfg.N_PRACTICE_TRIALS
CHECK_RATE = cfg.CHECK_RATE
MIN_IMAGE_VIEW_S = cfg.MIN_IMAGE_VIEW_S
BREAK_AFTER_TRIALS = cfg.BREAK_AFTER_TRIALS
BREAK_MIN_S = cfg.BREAK_MIN_S
QUIT_DOUBLE_PRESS_WINDOW_S = cfg.QUIT_DOUBLE_PRESS_WINDOW_S
FAST_SKIP_DOUBLE_PRESS_WINDOW_S = cfg.FAST_SKIP_DOUBLE_PRESS_WINDOW_S
GAZE_SAMPLE_INTERVAL_S = cfg.GAZE_SAMPLE_INTERVAL_S
RATING_MIN = cfg.RATING_MIN
RATING_MAX = cfg.RATING_MAX
RATING_DEFAULT = cfg.RATING_DEFAULT
RATING_SPEED_BASE = cfg.RATING_SPEED_BASE
RATING_SPEED_ACCEL = cfg.RATING_SPEED_ACCEL
RATING_SPEED_MAX = cfg.RATING_SPEED_MAX
MAX_MAIN_STIMULI = cfg.MAX_MAIN_STIMULI

PREPARED_DESCRIPTION_CONDITIONS = ("congruent", "ambiguous", "incongruent")
PREPARED_METADATA_CONDITION_BY_LABEL = {
    "congruent": "congruent",
    "ambiguous": "semi_congruent",
    "incongruent": "incongruent",
}


class UserAbort(Exception):
    """Raised when participant requests abort (Q twice)."""


class QuitState:
    """Track Q-press timing for two-press quit confirmation."""

    def __init__(self, confirm_window_s: float = QUIT_DOUBLE_PRESS_WINDOW_S) -> None:
        self.confirm_window_s = confirm_window_s
        self.first_q_session: float | None = None

    def process_keys(self, keys: list[str], now_session: float) -> None:
        if "q" not in keys:
            return
        if self.first_q_session is None:
            self.first_q_session = now_session
            return
        if now_session - self.first_q_session <= self.confirm_window_s:
            raise UserAbort
        self.first_q_session = now_session

    def active_message(self, now_session: float) -> str:
        if self.first_q_session is None:
            return ""
        if now_session - self.first_q_session <= self.confirm_window_s:
            return "" #"Press Q again to quit"
        self.first_q_session = None
        return ""


def get_run_info() -> tuple[str, str, str, int | None]:
    dlg = gui.Dlg(title="Congruency Rating Study")
    dlg.addText("Choose run mode and participant info.")
    dlg.addField("mode", choices=["real", "test", "rg"], initial="real")
    dlg.addField("participant", "")
    dlg.addField("session", "001")
    dlg.addField("main_trials (blank=mode default)", "")
    values = dlg.show()
    if not dlg.OK or values is None:
        core.quit()

    mode = str(values[0]).strip().lower() or "real"
    if mode not in {"real", "test", "rg"}:
        mode = "real"
    participant = str(values[1]).strip() or "test"
    session = str(values[2]).strip() or "001"
    trials_text = str(values[3]).strip()
    n_main_trials_override: int | None = None
    if trials_text:
        try:
            parsed = int(trials_text)
            if parsed > 0:
                n_main_trials_override = parsed
        except ValueError:
            n_main_trials_override = None
    return mode, participant, session, n_main_trials_override


def load_trials(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing trial file: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"trial_id", "description", "image_file"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Trial file must include columns {sorted(required)}. Missing: {sorted(missing)}"
            )
        rows = [
            {
                "trial_id": (row.get("trial_id") or "").strip(),
                "description": (row.get("description") or "").strip(),
                "image_file": (row.get("image_file") or "").strip(),
            }
            for row in reader
        ]

    rows = [row for row in rows if row["description"] and row["image_file"]]
    if not rows:
        raise ValueError(f"No valid rows in trial file: {path}")
    return rows


def load_prepared_trials(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared stimulus file: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        image_field = "image_file" if "image_file" in fieldnames else "new_filename" if "new_filename" in fieldnames else ""
        required = {
            "stimulus_id",
            "original_filename",
            "original_relative_path",
            "original_style_folder",
            "condition_congruent",
            "condition_semi_congruent",
            "condition_incongruent",
        }
        missing = required - fieldnames
        if not image_field:
            missing = sorted(set(missing) | {"image_file"})
        else:
            missing = sorted(missing)
        if missing:
            allowed = sorted(required | {"image_file", "new_filename"})
            raise ValueError(
                "Prepared stimulus file must include columns "
                f"{allowed}. Missing: {missing}"
            )

        rows: list[dict[str, str]] = []
        for row in reader:
            stimulus_id = (row.get("stimulus_id") or "").strip()
            image_file = (row.get(image_field) or "").strip()
            original_filename = (row.get("original_filename") or "").strip()
            original_relative_path = (row.get("original_relative_path") or "").strip()
            original_style_folder = (row.get("original_style_folder") or "").strip()
            congruent = (row.get("condition_congruent") or "").strip()
            ambiguous = (row.get("condition_semi_congruent") or "").strip()
            incongruent = (row.get("condition_incongruent") or "").strip()
            needs_review = (row.get("needs_human_review") or "").strip().lower()

            if not stimulus_id or not image_file:
                continue
            if not (congruent and ambiguous and incongruent):
                continue
            if needs_review in {"1", "true", "yes", "y"}:
                continue

            rows.append(
                {
                    "stimulus_id": stimulus_id,
                    "image_file": image_file,
                    "original_filename": original_filename,
                    "original_relative_path": original_relative_path,
                    "original_style_folder": original_style_folder,
                    "description_congruent": congruent,
                    "description_ambiguous": ambiguous,
                    "description_incongruent": incongruent,
                }
            )

    if not rows:
        raise ValueError(f"No usable prepared stimuli found in: {path}")
    if len(rows) > MAX_MAIN_STIMULI:
        raise ValueError(
            f"Prepared stimulus file contains {len(rows)} usable rows, which exceeds the "
            f"maximum of {MAX_MAIN_STIMULI}."
        )
    return rows


def build_condition_pool(n_trials: int) -> list[str]:
    labels = list(PREPARED_DESCRIPTION_CONDITIONS)
    base, remainder = divmod(n_trials, len(labels))
    pool = [label for label in labels for _ in range(base)]

    extras = labels.copy()
    random.shuffle(extras)
    pool.extend(extras[:remainder])
    random.shuffle(pool)
    return pool


def build_prepared_trial_sequence(
    prepared_trials: list[dict[str, str]],
    n_trials: int,
) -> list[dict[str, str]]:
    if n_trials > len(prepared_trials):
        raise ValueError(
            f"Requested {n_trials} main trials, but only {len(prepared_trials)} prepared stimuli are available."
        )
    selected_stimuli = random.sample(prepared_trials, n_trials)
    condition_pool = build_condition_pool(n_trials)

    sequence: list[dict[str, str]] = []
    for stimulus, condition in zip(selected_stimuli, condition_pool):
        metadata_condition = PREPARED_METADATA_CONDITION_BY_LABEL[condition]
        sequence.append(
            {
                "stimulus_id": stimulus["stimulus_id"],
                "trial_id": stimulus["stimulus_id"],
                "description_condition": condition,
                "metadata_condition": metadata_condition,
                "description": stimulus[f"description_{condition}"],
                "image_file": stimulus["image_file"],
                "original_filename": stimulus["original_filename"],
                "original_relative_path": stimulus["original_relative_path"],
                "original_style_folder": stimulus["original_style_folder"],
            }
        )
    return sequence


def sanitize_path_component(value: str, default: str = "unknown") -> str:
    text = value.strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or default


def build_run_timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]


def build_run_output_layout(
    project_root: Path,
    participant: str,
    run_timestamp: str,
) -> dict[str, Path | str]:
    output_root = project_root / "output"
    participant_slug = sanitize_path_component(participant)
    participant_dir = output_root / f"participant_{participant_slug}"
    run_dir = participant_dir / run_timestamp
    return {
        "output_root": output_root,
        "participant_slug": participant_slug,
        "participant_dir": participant_dir,
        "run_dir": run_dir,
        "main_trials_csv": run_dir / "main_trials.csv",
        "main_gaze_csv": run_dir / "main_gaze.csv",
        "practice_trials_csv": run_dir / "practice_trials.csv",
        "practice_gaze_csv": run_dir / "practice_gaze.csv",
        "main_sequence_csv": run_dir / "main_sequence.csv",
        "qc_png": run_dir / "qc.png",
        "participant_index_csv": output_root / "participants_latest.csv",
    }


def upsert_latest_row(
    csv_path: Path,
    row: dict[str, str | int | float],
    fieldnames: list[str],
    key_field: str,
) -> None:
    existing_rows: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for existing_row in reader:
                if existing_row.get(key_field) != str(row.get(key_field, "")):
                    existing_rows.append({k: v for k, v in existing_row.items() if k is not None})

    existing_rows.append({field: row.get(field, "") for field in fieldnames})
    existing_rows.sort(key=lambda item: item.get(key_field, ""))
    write_rows(csv_path, existing_rows, fieldnames)


def build_trial_sequence(base_trials: list[dict[str, str]], n_trials: int) -> list[dict[str, str]]:
    pool = base_trials.copy()
    random.shuffle(pool)

    sequence: list[dict[str, str]] = []
    while len(sequence) < n_trials:
        if not pool:
            pool = base_trials.copy()
            random.shuffle(pool)
        sequence.append(pool.pop())
    return sequence


def choose_check_trials(n_trials: int, rate: float) -> set[int]:
    n_checks = max(1, int(round(n_trials * rate)))
    n_checks = min(n_checks, n_trials)
    return set(random.sample(range(1, n_trials + 1), n_checks))


def append_rows(csv_path: Path, rows: list[dict[str, str | int | float]], fieldnames: list[str]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_rows(csv_path: Path, rows: list[dict[str, str | int | float]], fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_image_path(image_dir: Path, image_file: str) -> Path | None:
    requested = image_dir / image_file
    if requested.exists():
        return requested

    if not image_dir.exists():
        return None

    lowered_name = image_file.lower()
    for candidate in image_dir.iterdir():
        if candidate.is_file() and candidate.name.lower() == lowered_name:
            return candidate

    stem = Path(image_file).stem
    candidates = [candidate for candidate in image_dir.iterdir() if candidate.is_file() and candidate.stem == stem]
    if not candidates:
        return None

    requested_suffix = Path(image_file).suffix.lower()
    preferred_suffixes = [suffix for suffix in [requested_suffix, ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".gif", ".tif", ".tiff", ".webp"] if suffix]

    def sort_key(candidate: Path) -> tuple[int, str]:
        suffix = candidate.suffix.lower()
        try:
            rank = preferred_suffixes.index(suffix)
        except ValueError:
            rank = len(preferred_suffixes)
        return rank, candidate.name.lower()

    candidates.sort(key=sort_key)
    return candidates[0]


def load_displayable_image_array(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        return np.asarray(pil_image)


def create_image_stim(
    win: visual.Window,
    image_path: Path,
    size: tuple[float, float] = (1.3, 0.9),
    units: str = "height",
) -> visual.ImageStim:
    try:
        image_data = load_displayable_image_array(image_path)
        return visual.ImageStim(win, image=image_data, size=size, units=units)
    except Exception:
        return visual.ImageStim(win, image=str(image_path), size=size, units=units)


def build_image_cache(
    win: visual.Window,
    trials: list[dict[str, str]],
    image_dir: Path,
) -> dict[str, visual.ImageStim]:
    cache: dict[str, visual.ImageStim] = {}
    unresolved: list[str] = []
    for trial in trials:
        image_file = trial["image_file"]
        if image_file in cache:
            continue
        image_path = resolve_image_path(image_dir, image_file)
        if image_path is None:
            unresolved.append(image_file)
            continue
        try:
            cache[image_file] = create_image_stim(win, image_path)
        except Exception as exc:
            unresolved.append(f"{image_file} ({exc})")
    if unresolved:
        raise ValueError(
            f"Unable to load {len(unresolved)} image file(s) from {image_dir}: {', '.join(unresolved[:10])}"
            + (" ..." if len(unresolved) > 10 else "")
        )
    return cache


def read_rows_for_run(csv_path: Path, run_id: str, phase: str | None = None) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("run_id") == run_id]
    if phase is None:
        return rows
    return [row for row in rows if row.get("phase") == phase]


def get_condition_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        condition = (row.get("description_condition") or "").strip()
        if condition in PREPARED_DESCRIPTION_CONDITIONS:
            counts[condition] += 1
    return {label: counts.get(label, 0) for label in PREPARED_DESCRIPTION_CONDITIONS}


def build_latest_participant_row(
    project_root: Path,
    participant: str,
    participant_dir: Path,
    run_dir: Path,
    mode: str,
    session: str,
    run_id: str,
    run_started_at: str,
    run_finished_at: str,
    run_duration_s: float,
    run_status: str,
    run_error: str,
    main_output_csv: Path,
    main_gaze_csv: Path,
    practice_output_csv: Path,
    practice_gaze_csv: Path,
    main_sequence_csv: Path,
    qc_png: Path | None,
    n_main_trials_expected: int,
    n_practice_trials_expected: int,
    record_cursor_samples: bool,
    main_trial_source: Path,
    practice_trial_source: Path,
) -> dict[str, str | int | float]:
    main_rows = read_rows_for_run(main_output_csv, run_id, phase="main")
    practice_rows = read_rows_for_run(practice_output_csv, run_id, phase="practice")
    main_gaze_rows = read_rows_for_run(main_gaze_csv, run_id) if record_cursor_samples else []
    practice_gaze_rows = (
        read_rows_for_run(practice_gaze_csv, run_id) if record_cursor_samples else []
    )

    return {
        "participant": participant,
        "participant_folder": str(participant_dir.relative_to(project_root)),
        "run_folder": str(run_dir.relative_to(project_root)),
        "run_id": run_id,
        "mode": mode,
        "session": session,
        "status": run_status,
        "error_message": run_error,
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "run_duration_s": round(run_duration_s, 3),
        "n_main_trials_expected": n_main_trials_expected,
        "n_main_trials_recorded": len(main_rows),
        "n_practice_trials_expected": n_practice_trials_expected,
        "n_practice_trials_recorded": len(practice_rows),
        "n_main_gaze_rows": len(main_gaze_rows),
        "n_practice_gaze_rows": len(practice_gaze_rows),
        "record_cursor_samples": int(bool(record_cursor_samples)),
        "main_trials_csv": str(main_output_csv.relative_to(project_root)),
        "main_gaze_csv": str(main_gaze_csv.relative_to(project_root)),
        "practice_trials_csv": str(practice_output_csv.relative_to(project_root)),
        "practice_gaze_csv": str(practice_gaze_csv.relative_to(project_root)),
        "main_sequence_csv": str(main_sequence_csv.relative_to(project_root)),
        "qc_png": str(qc_png.relative_to(project_root)) if qc_png and qc_png.exists() else "",
        "main_stimulus_source": str(main_trial_source.relative_to(project_root)),
        "practice_stimulus_source": str(practice_trial_source.relative_to(project_root)),
    }


def to_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def create_qc_plot(
    run_id: str,
    mode: str,
    main_rows: list[dict[str, str]],
    gaze_rows: list[dict[str, str]],
    output_dir: Path,
    expected_main_trials: int,
    expect_cursor_samples: bool,
) -> tuple[Path | None, list[str]]:
    summary: list[str] = []
    if not main_rows:
        return None, ["No main-trial rows found for this run."]

    main_rows_sorted = sorted(main_rows, key=lambda r: int(r.get("trial_index", "0") or 0))
    trial_idx = [int(r.get("trial_index", "0") or 0) for r in main_rows_sorted]
    image_view = [to_float(r.get("image_view_time_s")) for r in main_rows_sorted]
    rating = [to_float(r.get("rating")) for r in main_rows_sorted]
    rating_rt = [to_float(r.get("rating_rt_s")) for r in main_rows_sorted]
    condition_counts = get_condition_counts(main_rows_sorted)
    stimulus_ids = [
        (row.get("stimulus_id") or row.get("trial_id") or "").strip()
        for row in main_rows_sorted
    ]
    stimulus_ids = [stimulus_id for stimulus_id in stimulus_ids if stimulus_id]

    valid_image_view = [v for v in image_view if v is not None]
    valid_rating = [v for v in rating if v is not None]
    valid_rating_rt = [v for v in rating_rt if v is not None]

    summary.append(f"Main trials recorded: {len(main_rows_sorted)}/{expected_main_trials}")
    if valid_image_view:
        summary.append(
            f"Image view time (s): mean {sum(valid_image_view)/len(valid_image_view):.2f}, "
            f"min {min(valid_image_view):.2f}, max {max(valid_image_view):.2f}"
        )
    if valid_rating:
        summary.append(
            f"Rating: mean {sum(valid_rating)/len(valid_rating):.2f}, "
            f"min {min(valid_rating):.2f}, max {max(valid_rating):.2f}"
        )
    if valid_rating_rt:
        summary.append(
            f"Rating RT (s): mean {sum(valid_rating_rt)/len(valid_rating_rt):.2f}, "
            f"min {min(valid_rating_rt):.2f}, max {max(valid_rating_rt):.2f}"
        )
    if condition_counts:
        summary.append(
            "Condition counts: "
            + ", ".join(f"{label} {condition_counts[label]}" for label in PREPARED_DESCRIPTION_CONDITIONS)
        )
    if stimulus_ids:
        unique_stimuli = len(set(stimulus_ids))
        if unique_stimuli == len(stimulus_ids):
            summary.append(f"Stimulus IDs: all unique ({unique_stimuli})")
        else:
            summary.append(
                f"Warning: repeated stimulus IDs detected ({unique_stimuli}/{len(stimulus_ids)} unique)."
            )

    if len(main_rows_sorted) != expected_main_trials:
        summary.append("Warning: trial count differs from expected.")

    short_trials = [v for v in valid_image_view if v < (MIN_IMAGE_VIEW_S - 0.01)]
    if mode != "test" and short_trials:
        summary.append(
            f"Warning: {len(short_trials)} trial(s) were shorter than {MIN_IMAGE_VIEW_S:.1f}s."
        )
    elif mode == "test" and short_trials:
        summary.append(f"Info: {len(short_trials)} short trials (likely fast-skip in test mode).")

    sample_count_by_trial: dict[int, int] = defaultdict(int)
    for row in gaze_rows:
        trial = int(row.get("trial_index", "0") or 0)
        sample_count_by_trial[trial] += 1
    sample_counts = [sample_count_by_trial.get(t, 0) for t in trial_idx]

    if expect_cursor_samples:
        if sum(sample_counts) == 0:
            summary.append("Warning: no cursor-proxy samples recorded.")
        else:
            summary.append(
                f"Cursor samples: total {sum(sample_counts)}, "
                f"mean/trial {sum(sample_counts)/max(1, len(sample_counts)):.1f}"
            )
    else:
        summary.append("Cursor-proxy samples: disabled for this mode.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)
        qc_path = output_dir / "qc.png"

        fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), constrained_layout=True)
        fig.suptitle(f"Run QC - {run_id}", fontsize=13)

        ax1 = axes[0]
        ax1.plot(trial_idx, image_view, marker="o", linewidth=1.5, color="#1f77b4", label="image_view_time_s")
        ax1.axhline(MIN_IMAGE_VIEW_S, linestyle="--", linewidth=1.0, color="#888888", label="min image time")
        ax1.set_ylabel("Image View Time (s)")
        ax1.set_xlabel("Trial")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper left", fontsize=8)

        ax1b = ax1.twinx()
        ax1b.plot(trial_idx, rating, marker="x", linewidth=1.2, color="#d62728", label="rating")
        ax1b.set_ylabel("Rating")
        ax1b.set_ylim(0.8, 5.2)

        ax2 = axes[1]
        if sample_counts and sum(sample_counts) > 0:
            ax2.bar(trial_idx, sample_counts, color="#2ca02c")
            ax2.set_ylabel("Cursor Samples")
        else:
            ax2.text(0.5, 0.5, "No cursor-proxy samples in this run", ha="center", va="center")
            ax2.set_yticks([])
        ax2.set_xlabel("Trial")
        ax2.grid(alpha=0.25, axis="y")

        fig.savefig(qc_path, dpi=140)
        plt.close(fig)
        return qc_path, summary
    except Exception as exc:
        summary.append(f"QC plot generation failed: {exc}")
        return None, summary


def build_validity_checklist(
    main_rows: list[dict[str, str]],
    gaze_rows: list[dict[str, str]],
    mode: str,
    expected_main_trials: int,
    expect_cursor_samples: bool,
) -> list[str]:
    checks: list[str] = []
    if not main_rows:
        return ["[FAIL] No main-trial rows found for this run."]

    rows = sorted(main_rows, key=lambda r: int(r.get("trial_index", "0") or 0))
    n_rows = len(rows)
    checks.append(
        f"{'[PASS]' if n_rows == expected_main_trials else '[FAIL]'} Trial count: {n_rows}/{expected_main_trials}"
    )

    trial_idx = [int(r.get("trial_index", "0") or 0) for r in rows]
    contiguous = trial_idx == list(range(1, n_rows + 1))
    checks.append(
        f"{'[PASS]' if contiguous else '[WARN]'} Trial index continuity: "
        f"{'contiguous' if contiguous else 'gaps/duplicates detected'}"
    )

    image_view = [to_float(r.get("image_view_time_s")) for r in rows]
    rating = [to_float(r.get("rating")) for r in rows]
    rating_rt = [to_float(r.get("rating_rt_s")) for r in rows]
    condition_counts = get_condition_counts(rows)
    stimulus_ids = [
        (row.get("stimulus_id") or row.get("trial_id") or "").strip() for row in rows
    ]
    stimulus_ids = [stimulus_id for stimulus_id in stimulus_ids if stimulus_id]

    missing_iv = sum(v is None for v in image_view)
    missing_rating = sum(v is None for v in rating)
    missing_rt = sum(v is None for v in rating_rt)
    missing_total = missing_iv + missing_rating + missing_rt
    checks.append(
        f"{'[PASS]' if missing_total == 0 else '[FAIL]'} Missing values in key fields: {missing_total}"
    )

    valid_iv = [v for v in image_view if v is not None]
    if valid_iv:
        short_count = sum(v < (MIN_IMAGE_VIEW_S - 0.01) for v in valid_iv)
        if mode != "test":
            checks.append(
                f"{'[PASS]' if short_count == 0 else '[FAIL]'} Min image-time rule ({MIN_IMAGE_VIEW_S:.1f}s): "
                f"{len(valid_iv) - short_count}/{len(valid_iv)} pass"
            )
        else:
            checks.append(
                f"{'[WARN]' if short_count > 0 else '[PASS]'} Test-mode short trials (<{MIN_IMAGE_VIEW_S:.1f}s): "
                f"{short_count}"
            )

    valid_rating = [v for v in rating if v is not None]
    out_of_range = sum((v < RATING_MIN - 1e-6) or (v > RATING_MAX + 1e-6) for v in valid_rating)
    checks.append(
        f"{'[PASS]' if out_of_range == 0 else '[FAIL]'} Rating range [{RATING_MIN:.0f},{RATING_MAX:.0f}]: "
        f"{len(valid_rating) - out_of_range}/{len(valid_rating)} valid"
    )

    if len(valid_rating) >= 2:
        mean_r = sum(valid_rating) / len(valid_rating)
        var_r = sum((v - mean_r) ** 2 for v in valid_rating) / len(valid_rating)
        sd_r = math.sqrt(var_r)
        checks.append(
            f"{'[PASS]' if sd_r >= 0.15 else '[WARN]'} Rating variability (SD): {sd_r:.3f}"
        )

    mismatch_triggered = sum((r.get("mismatch_check_triggered") or "0").strip() == "1" for r in rows)
    expected_checks = max(1, int(round(expected_main_trials * CHECK_RATE)))
    checks.append(
        f"{'[PASS]' if abs(mismatch_triggered - expected_checks) <= 1 else '[WARN]'} "
        f"Manipulation checks: {mismatch_triggered} (target ~{expected_checks})"
    )

    if condition_counts:
        condition_values = list(condition_counts.values())
        condition_balanced = max(condition_values) - min(condition_values) <= 1
        checks.append(
            f"{'[PASS]' if condition_balanced else '[WARN]'} Condition balance: "
            + ", ".join(f"{label}={condition_counts[label]}" for label in PREPARED_DESCRIPTION_CONDITIONS)
        )

    if stimulus_ids:
        unique_stimuli = len(set(stimulus_ids))
        if unique_stimuli == len(stimulus_ids):
            checks.append(f"[PASS] Unique stimulus IDs: {unique_stimuli}/{len(stimulus_ids)}")
        else:
            checks.append(
                f"[WARN] Unique stimulus IDs: {unique_stimuli}/{len(stimulus_ids)} "
                "(repeats allowed only when trial count exceeds the available pool)"
            )

    if expect_cursor_samples:
        sample_count_by_trial: dict[int, int] = defaultdict(int)
        for row in gaze_rows:
            sample_count_by_trial[int(row.get("trial_index", "0") or 0)] += 1
        counts = [sample_count_by_trial.get(t, 0) for t in trial_idx]
        total_actual = sum(counts)
        total_expected = int(round(sum(v for v in valid_iv) * cfg.CURSOR_TARGET_HZ))
        if total_expected > 0:
            ratio = total_actual / total_expected
            tag = "[PASS]" if ratio >= 0.98 else "[WARN]"
            checks.append(
                f"{tag} Cursor sample count vs expected@{cfg.CURSOR_TARGET_HZ:.0f}Hz: "
                f"{total_actual}/{total_expected} ({ratio:.2%})"
            )
        else:
            checks.append("[WARN] Cursor sample validation skipped (no valid image durations).")
    else:
        checks.append("[PASS] Cursor-proxy sampling disabled by design in this mode.")

    return checks


def resample_cursor_samples_fixed_hz(
    raw_points: list[tuple[float, float, float, float]],
    onset_session: float,
    onset_unix: float,
    offset_session: float,
    run_id: str,
    participant: str,
    session: str,
    trial_index: int,
    trial_id: str,
    image_file: str,
    target_hz: float,
) -> list[dict[str, str | int | float]]:
    duration = max(0.0, offset_session - onset_session)
    if target_hz <= 0:
        return []
    n_samples = max(1, int(round(duration * target_hz)))
    if not raw_points:
        return []

    raw_points = sorted(raw_points, key=lambda p: p[0])
    rows: list[dict[str, str | int | float]] = []
    j = 0

    for i in range(n_samples):
        target_session = onset_session + (i / target_hz)

        while j + 1 < len(raw_points) and raw_points[j + 1][0] <= target_session:
            j += 1

        if j + 1 < len(raw_points):
            t0, _, x0, y0 = raw_points[j]
            t1, _, x1, y1 = raw_points[j + 1]
            if t1 > t0:
                alpha = (target_session - t0) / (t1 - t0)
                alpha = min(1.0, max(0.0, alpha))
                x = x0 + alpha * (x1 - x0)
                y = y0 + alpha * (y1 - y0)
            else:
                x, y = x0, y0
        else:
            _, _, x, y = raw_points[-1]

        target_unix = onset_unix + (target_session - onset_session)
        rows.append(
            {
                "run_id": run_id,
                "participant": participant,
                "session": session,
                "trial_index": trial_index,
                "trial_id": trial_id,
                "image_file": image_file,
                "sample_index": i + 1,
                "sample_session_s": round(target_session, 6),
                "sample_unix_s": round(target_unix, 6),
                "sample_from_image_onset_s": round(target_session - onset_session, 6),
                "gaze_x_pix": round(float(x), 3),
                "gaze_y_pix": round(float(y), 3),
                "units": "pix",
            }
        )

    return rows


def wait_for_continue(
    win: visual.Window,
    stim: visual.BaseVisualStim,
    session_clock: core.MonotonicClock,
    quit_state: QuitState,
    quit_hint: visual.TextStim,
    progress_stim: visual.TextStim | None = None,
    continue_key: str = "space",
) -> tuple[float, float, float, float, float]:
    event.clearEvents(eventType="keyboard")
    onset_session = session_clock.getTime()
    onset_unix = time.time()

    while True:
        now_session = session_clock.getTime()
        quit_hint.text = quit_state.active_message(now_session)

        if progress_stim is not None:
            progress_stim.draw()
        stim.draw()
        if quit_hint.text:
            quit_hint.draw()
        win.flip()

        keys = event.getKeys(keyList=[continue_key, "q"])
        now_session = session_clock.getTime()
        quit_state.process_keys(keys, now_session)
        if continue_key in keys:
            offset_session = session_clock.getTime()
            offset_unix = time.time()
            duration = offset_session - onset_session
            return duration, onset_session, offset_session, onset_unix, offset_unix


def show_image_until_continue(
    win: visual.Window,
    image_stim: visual.ImageStim | None,
    image_path: Path,
    session_clock: core.MonotonicClock,
    quit_state: QuitState,
    quit_hint: visual.TextStim,
    progress_stim: visual.TextStim,
    min_view_s: float,
    show_min_view_countdown: bool,
    mouse: event.Mouse,
    run_id: str,
    participant: str,
    session: str,
    trial_index: int,
    trial_id: str,
    image_file: str,
    fallback_text: visual.TextStim,
    allow_fast_skip: bool = False,
    record_cursor_samples: bool = True,
    continue_key: str = "space",
) -> tuple[float, float, float, float, float, list[dict[str, str | int | float]]]:
    event.clearEvents(eventType="keyboard")
    onset_session = session_clock.getTime()
    onset_unix = time.time()

    if image_stim is not None:
        draw_stim: visual.BaseVisualStim = image_stim
    else:
        fallback_text.text = (
            "Image file missing:\n"
            f"{image_path.name}\n\n"
            "Press SPACE to continue."
        )
        draw_stim = fallback_text

    min_wait_stim = visual.TextStim(
        win,
        text="",
        color="white",
        height=0.03,
        pos=(0, -0.42),
    )

    raw_cursor_points: list[tuple[float, float, float, float]] = []
    first_f_session: float | None = None

    if record_cursor_samples:
        x0, y0 = mouse.getPos()
        raw_cursor_points.append(
            (
                onset_session,
                onset_unix,
                float(x0) * float(win.size[1]),
                float(y0) * float(win.size[1]),
            )
        )

    def finalize_return(offset_session: float, offset_unix: float) -> tuple[
        float, float, float, float, float, list[dict[str, str | int | float]]
    ]:
        duration = offset_session - onset_session
        if record_cursor_samples:
            sample_rows = resample_cursor_samples_fixed_hz(
                raw_points=raw_cursor_points,
                onset_session=onset_session,
                onset_unix=onset_unix,
                offset_session=offset_session,
                run_id=run_id,
                participant=participant,
                session=session,
                trial_index=trial_index,
                trial_id=trial_id,
                image_file=image_file,
                target_hz=cfg.CURSOR_TARGET_HZ,
            )
        else:
            sample_rows = []
        return duration, onset_session, offset_session, onset_unix, offset_unix, sample_rows

    while True:
        now_session = session_clock.getTime()
        elapsed = now_session - onset_session
        wait_left = max(0.0, min_view_s - elapsed)

        if wait_left > 0:
            if show_min_view_countdown:
                min_wait_stim.text = f"Keep viewing for {wait_left:0.1f}s before continuing"
            else:
                min_wait_stim.text = ""
        else:
            min_wait_stim.text = "Press SPACE to continue"

        quit_hint.text = quit_state.active_message(now_session)

        progress_stim.draw()
        draw_stim.draw()
        min_wait_stim.draw()
        if quit_hint.text:
            quit_hint.draw()
        win.flip()

        sample_session = session_clock.getTime()
        if record_cursor_samples:
            gaze_x, gaze_y = mouse.getPos()
            gaze_x_pix = float(gaze_x) * float(win.size[1])
            gaze_y_pix = float(gaze_y) * float(win.size[1])
            sample_unix = time.time()
            raw_cursor_points.append((sample_session, sample_unix, gaze_x_pix, gaze_y_pix))

        keys = event.getKeys(keyList=[continue_key, "q", "f"])
        now_session = session_clock.getTime()
        quit_state.process_keys(keys, now_session)

        if allow_fast_skip:
            for key in keys:
                if key != "f":
                    continue
                if (
                    first_f_session is not None
                    and (now_session - first_f_session) <= FAST_SKIP_DOUBLE_PRESS_WINDOW_S
                ):
                    return finalize_return(session_clock.getTime(), time.time())
                first_f_session = now_session

        if continue_key in keys and (now_session - onset_session) >= min_view_s:
            return finalize_return(session_clock.getTime(), time.time())


def collect_rating(
    win: visual.Window,
    question_stim: visual.TextStim,
    controls_stim: visual.TextStim,
    slider: visual.Slider,
    confirm_rect: visual.Rect,
    confirm_label: visual.TextStim,
    quit_state: QuitState,
    quit_hint: visual.TextStim,
    progress_stim: visual.TextStim,
    key_state: object,
    session_clock: core.MonotonicClock,
) -> tuple[float, float, float, float, float, float]:
    slider.reset()
    event.clearEvents(eventType="keyboard")
    onset_session = session_clock.getTime()
    onset_unix = time.time()
    rating_value = RATING_DEFAULT
    slider.markerPos = rating_value
    slider.rating = rating_value
    hold_left_s = 0.0
    hold_right_s = 0.0
    last_session = onset_session

    while True:
        now_session = session_clock.getTime()
        dt = min(max(now_session - last_session, 0.0), 0.05)
        last_session = now_session

        quit_hint.text = quit_state.active_message(now_session)
        keys = event.getKeys(keyList=["space", "q"])
        now_session = session_clock.getTime()
        quit_state.process_keys(keys, now_session)

        left_pressed = bool(key_state[event.pyglet.window.key.LEFT])
        right_pressed = bool(key_state[event.pyglet.window.key.RIGHT])

        if left_pressed and not right_pressed:
            hold_left_s += dt
            hold_right_s = 0.0
            speed = min(RATING_SPEED_MAX, RATING_SPEED_BASE + RATING_SPEED_ACCEL * hold_left_s)
            rating_value -= speed * dt
        elif right_pressed and not left_pressed:
            hold_right_s += dt
            hold_left_s = 0.0
            speed = min(RATING_SPEED_MAX, RATING_SPEED_BASE + RATING_SPEED_ACCEL * hold_right_s)
            rating_value += speed * dt
        else:
            hold_left_s = 0.0
            hold_right_s = 0.0

        rating_value = max(RATING_MIN, min(RATING_MAX, rating_value))
        slider.markerPos = rating_value
        slider.rating = rating_value
        confirm_rect.fillColor = "#2f8f46"

        progress_stim.draw()
        question_stim.draw()
        controls_stim.draw()
        slider.draw()
        confirm_rect.draw()
        confirm_label.draw()
        if quit_hint.text:
            quit_hint.draw()
        win.flip()

        if "space" in keys:
            offset_session = session_clock.getTime()
            offset_unix = time.time()
            return (
                float(rating_value),
                offset_session - onset_session,
                onset_session,
                offset_session,
                onset_unix,
                offset_unix,
            )


def collect_mismatch_check(
    win: visual.Window,
    question: str,
    quit_state: QuitState,
    quit_hint: visual.TextStim,
    progress_stim: visual.TextStim,
    session_clock: core.MonotonicClock,
) -> tuple[str, float, float, float, float, float]:
    event.clearEvents(eventType="keyboard")
    onset_session = session_clock.getTime()
    onset_unix = time.time()

    question_stim = visual.TextStim(
        win,
        text=question,
        color="white",
        height=0.04,
        pos=(0, 0.34),
        wrapWidth=1.5,
    )
    instruction_stim = visual.TextStim(
        win,
        text="Type your answer and press ENTER to confirm.",
        color="#cccccc",
        height=0.03,
        pos=(0, -0.35),
    )
    answer_stim = visual.TextStim(
        win,
        text="",
        color="white",
        height=0.04,
        pos=(0, 0.02),
        wrapWidth=1.5,
    )

    text_chars: list[str] = []

    while True:
        now_session = session_clock.getTime()
        quit_hint.text = quit_state.active_message(now_session)

        progress_stim.draw()
        question_stim.draw()
        answer_stim.text = "".join(text_chars) if text_chars else "_"
        answer_stim.draw()
        instruction_stim.draw()
        if quit_hint.text:
            quit_hint.draw()
        win.flip()

        keys = event.getKeys()
        if not keys:
            continue

        now_session = session_clock.getTime()
        quit_state.process_keys(keys, now_session)

        for key in keys:
            if key == "return":
                answer = "".join(text_chars).strip()
                if answer:
                    offset_session = session_clock.getTime()
                    offset_unix = time.time()
                    return (
                        answer,
                        offset_session - onset_session,
                        onset_session,
                        offset_session,
                        onset_unix,
                        offset_unix,
                    )
            elif key == "backspace":
                if text_chars:
                    text_chars.pop()
            elif key == "space":
                text_chars.append(" ")
            elif key == "minus":
                text_chars.append("-")
            elif key == "period":
                text_chars.append(".")
            elif key == "comma":
                text_chars.append(",")
            elif key == "apostrophe":
                text_chars.append("'")
            elif key == "slash":
                text_chars.append("/")
            elif len(key) == 1:
                text_chars.append(key)


def run_break(
    win: visual.Window,
    break_index: int,
    total_breaks: int,
    session_clock: core.MonotonicClock,
    quit_state: QuitState,
    quit_hint: visual.TextStim,
    allow_fast_skip: bool = False,
) -> None:
    event.clearEvents(eventType="keyboard")
    break_start = session_clock.getTime()
    first_f_session: float | None = None

    break_text = visual.TextStim(
        win,
        text="",
        color="white",
        height=0.042,
        wrapWidth=1.5,
    )

    while True:
        now_session = session_clock.getTime()
        elapsed = now_session - break_start
        remaining = max(0.0, BREAK_MIN_S - elapsed)
        quit_hint.text = quit_state.active_message(now_session)

        if remaining > 0:
            continue_text = f"Break continues for {remaining:0.0f}s"
        else:
            continue_text = "Press SPACE when you are ready to continue"

        break_text.text = (
            f"Break {break_index}/{total_breaks}\n\n"
            "Please rest your eyes, blink, and stretch your shoulders.\n"
            "Keep your seating position stable for the next block.\n\n"
            f"{continue_text}"
        )

        break_text.draw()
        if quit_hint.text:
            quit_hint.draw()
        win.flip()

        keys = event.getKeys(keyList=["space", "q", "f"])
        now_session = session_clock.getTime()
        quit_state.process_keys(keys, now_session)

        if allow_fast_skip:
            for key in keys:
                if key != "f":
                    continue
                if (
                    first_f_session is not None
                    and (now_session - first_f_session) <= FAST_SKIP_DOUBLE_PRESS_WINDOW_S
                ):
                    return
                first_f_session = now_session

        if "space" in keys and remaining <= 0:
            return


def wait_for_space_screen(
    win: visual.Window,
    session_clock: core.MonotonicClock,
    quit_state: QuitState,
    quit_hint: visual.TextStim,
    text_stim: visual.TextStim,
    image_stim: visual.ImageStim | None = None,
) -> None:
    event.clearEvents(eventType="keyboard")
    while True:
        now_session = session_clock.getTime()
        quit_hint.text = quit_state.active_message(now_session)
        if image_stim is not None:
            image_stim.draw()
        text_stim.draw()
        if quit_hint.text:
            quit_hint.draw()
        win.flip()
        keys = event.getKeys(keyList=["space", "q"])
        now_session = session_clock.getTime()
        quit_state.process_keys(keys, now_session)
        if "space" in keys:
            return


def run_phase(
    phase_name: str,
    win: visual.Window,
    mouse: event.Mouse,
    key_state: object,
    session_clock: core.MonotonicClock,
    quit_state: QuitState,
    quit_hint: visual.TextStim,
    run_id: str,
    participant: str,
    session: str,
    trials: list[dict[str, str]],
    image_dir: Path,
    image_cache: dict[str, visual.ImageStim],
    output_csv: Path,
    gaze_csv: Path,
    trial_fieldnames: list[str],
    gaze_fieldnames: list[str],
    record_cursor_samples: bool,
    sequence_csv: Path | None = None,
    sequence_fieldnames: list[str] | None = None,
    check_trials: set[int] | None = None,
    enable_breaks: bool = False,
    show_min_view_countdown: bool = True,
    allow_fast_skip: bool = False,
) -> None:
    prime_stim = visual.TextStim(win, color="white", height=0.05, wrapWidth=1.5)
    fallback_stim = visual.TextStim(win, color="white", height=0.045, wrapWidth=1.5)
    progress_stim = visual.TextStim(
        win,
        text="",
        color="#cccccc",
        height=0.03,
        pos=(-0.62, 0.46),
        alignText="left",
        anchorHoriz="left",
    )

    question_stim = visual.TextStim(
        win,
        text="How well do you feel like the description fits the image you were presented with?",
        color="white",
        height=0.04,
        pos=(0, 0.35),
        wrapWidth=1.5,
    )
    controls_stim = visual.TextStim(
        win,
        text="Hold LEFT/RIGHT to move the slider. Press SPACE to confirm.",
        color="#cccccc",
        height=0.028,
        pos=(0, 0.23),
        wrapWidth=1.5,
    )
    slider = visual.Slider(
        win,
        ticks=[1, 2, 3, 4, 5],
        labels=["Doesn't fit at all", "", "", "", "Fits perfectly"],
        granularity=0,
        style=["rating"],
        size=(1.1, 0.1),
        pos=(0, 0.02),
        color="white",
        labelHeight=0.03,
    )
    confirm_rect = visual.Rect(
        win,
        width=0.34,
        height=0.11,
        pos=(0, -0.30),
        fillColor="#666666",
        lineColor="white",
    )
    confirm_label = visual.TextStim(
        win,
        text="SPACE = Confirm",
        color="white",
        height=0.034,
        pos=(0, -0.30),
    )

    total_breaks = len(BREAK_AFTER_TRIALS) if enable_breaks else 0
    break_counter = 0

    if sequence_csv is not None:
        if sequence_fieldnames is None:
            raise ValueError("sequence_fieldnames must be provided when sequence_csv is set.")
        sequence_rows = [
            {
                "run_id": run_id,
                "phase": phase_name,
                "participant": participant,
                "session": session,
                "trial_index": idx,
                "stimulus_id": trial.get("stimulus_id") or trial.get("trial_id", ""),
                "description_condition": trial.get("description_condition", ""),
                "metadata_condition": trial.get("metadata_condition", ""),
                "description": trial.get("description", ""),
                "image_file": trial.get("image_file", ""),
                "original_filename": trial.get("original_filename", ""),
                "original_style_folder": trial.get("original_style_folder", ""),
                "original_relative_path": trial.get("original_relative_path", ""),
            }
            for idx, trial in enumerate(trials, start=1)
        ]
        write_rows(sequence_csv, sequence_rows, sequence_fieldnames)

    for idx, trial in enumerate(trials, start=1):
        progress_stim.text = f"{phase_name.capitalize()} Trial {idx}/{len(trials)}"

        trial_start_session = session_clock.getTime()
        trial_start_unix = time.time()

        prime_stim.text = (
            f"{phase_name.capitalize()} Trial {idx}/{len(trials)}\n\n"
            f"Description:\n{trial['description']}\n\n"
            "Press SPACE to continue."
        )
        (
            description_time,
            desc_onset_session,
            desc_offset_session,
            desc_onset_unix,
            desc_offset_unix,
        ) = wait_for_continue(
            win,
            prime_stim,
            session_clock,
            quit_state,
            quit_hint,
            progress_stim,
        )

        image_path = resolve_image_path(image_dir, trial["image_file"]) or (image_dir / trial["image_file"])
        image_stim = image_cache.get(trial["image_file"])
        (
            image_view_time,
            image_onset_session,
            image_offset_session,
            image_onset_unix,
            image_offset_unix,
            image_samples,
        ) = show_image_until_continue(
            win,
            image_stim,
            image_path,
            session_clock,
            quit_state,
            quit_hint,
            progress_stim,
            MIN_IMAGE_VIEW_S,
            show_min_view_countdown,
            mouse,
            run_id,
            participant,
            session,
            idx,
            trial["trial_id"],
            trial["image_file"],
            fallback_stim,
            allow_fast_skip=allow_fast_skip,
            record_cursor_samples=record_cursor_samples,
        )
        if record_cursor_samples and image_samples:
            append_rows(gaze_csv, image_samples, gaze_fieldnames)

        (
            rating,
            rating_rt,
            rating_onset_session,
            rating_offset_session,
            rating_onset_unix,
            rating_offset_unix,
            ) = collect_rating(
                win,
                question_stim,
                controls_stim,
                slider,
                confirm_rect,
                confirm_label,
                quit_state,
                quit_hint,
                progress_stim,
                key_state,
                session_clock,
            )

        mismatch_triggered = 0
        mismatch_response = ""
        mismatch_rt = ""
        mismatch_onset_session = ""
        mismatch_offset_session = ""
        mismatch_onset_unix = ""
        mismatch_offset_unix = ""

        if check_trials and idx in check_trials:
            mismatch_triggered = 1
            (
                mismatch_response,
                mismatch_rt_val,
                mismatch_onset_session_val,
                mismatch_offset_session_val,
                mismatch_onset_unix_val,
                mismatch_offset_unix_val,
            ) = collect_mismatch_check(
                win,
                "Which aspect of the description do you think does not match the image?",
                quit_state,
                quit_hint,
                progress_stim,
                session_clock,
            )
            mismatch_rt = round(mismatch_rt_val, 4)
            mismatch_onset_session = round(mismatch_onset_session_val, 6)
            mismatch_offset_session = round(mismatch_offset_session_val, 6)
            mismatch_onset_unix = round(mismatch_onset_unix_val, 6)
            mismatch_offset_unix = round(mismatch_offset_unix_val, 6)

        trial_row = {
            "run_id": run_id,
            "phase": phase_name,
            "participant": participant,
            "session": session,
            "trial_index": idx,
            "trial_id": trial["trial_id"],
            "stimulus_id": trial.get("stimulus_id") or trial["trial_id"],
            "description_condition": trial.get("description_condition", ""),
            "metadata_condition": trial.get("metadata_condition", ""),
            "description": trial["description"],
            "image_file": trial["image_file"],
            "original_filename": trial.get("original_filename", ""),
            "original_style_folder": trial.get("original_style_folder", ""),
            "original_relative_path": trial.get("original_relative_path", ""),
            "trial_start_session_s": round(trial_start_session, 6),
            "trial_start_unix_s": round(trial_start_unix, 6),
            "description_onset_session_s": round(desc_onset_session, 6),
            "description_offset_session_s": round(desc_offset_session, 6),
            "description_onset_unix_s": round(desc_onset_unix, 6),
            "description_offset_unix_s": round(desc_offset_unix, 6),
            "description_time_s": round(description_time, 4),
            "image_onset_session_s": round(image_onset_session, 6),
            "image_offset_session_s": round(image_offset_session, 6),
            "image_onset_unix_s": round(image_onset_unix, 6),
            "image_offset_unix_s": round(image_offset_unix, 6),
            "image_view_time_s": round(image_view_time, 4),
            "rating_onset_session_s": round(rating_onset_session, 6),
            "rating_offset_session_s": round(rating_offset_session, 6),
            "rating_onset_unix_s": round(rating_onset_unix, 6),
            "rating_offset_unix_s": round(rating_offset_unix, 6),
            "rating": round(rating, 4),
            "rating_rt_s": round(rating_rt, 4),
            "mismatch_check_triggered": mismatch_triggered,
            "mismatch_response": mismatch_response,
            "mismatch_rt_s": mismatch_rt,
            "mismatch_onset_session_s": mismatch_onset_session,
            "mismatch_offset_session_s": mismatch_offset_session,
            "mismatch_onset_unix_s": mismatch_onset_unix,
            "mismatch_offset_unix_s": mismatch_offset_unix,
        }
        append_rows(output_csv, [trial_row], trial_fieldnames)

        if enable_breaks and idx in BREAK_AFTER_TRIALS and idx < len(trials):
            break_counter += 1
            run_break(
                win,
                break_index=break_counter,
                total_breaks=total_breaks,
                session_clock=session_clock,
                quit_state=quit_state,
                quit_hint=quit_hint,
                allow_fast_skip=allow_fast_skip,
            )


def main() -> None:
    mode, participant, session, n_main_trials_override = get_run_info()

    project_root = Path(__file__).resolve().parents[1]
    practice_root = project_root / "input" / "practice"
    practice_trial_file = practice_root / "stimuli.csv"
    practice_image_dir = practice_root / "images"
    prepared_stimuli_root = project_root / "input" / "main"
    prepared_trial_file = prepared_stimuli_root / "stimuli.csv"

    practice_trials_raw = load_trials(practice_trial_file)
    if len(practice_trials_raw) < N_PRACTICE_TRIALS:
        raise ValueError(
            f"Need at least {N_PRACTICE_TRIALS} practice rows in {practice_trial_file}."
        )

    n_main_trials = (
        n_main_trials_override
        if n_main_trials_override is not None
        else (N_TRIALS_TEST if mode == "test" else (N_TRIALS_RG if mode == "rg" else N_TRIALS_REAL))
    )
    if n_main_trials > MAX_MAIN_STIMULI:
        raise ValueError(
            f"Requested {n_main_trials} main trials, but the prepared image bank is capped at {MAX_MAIN_STIMULI}."
        )
    practice_trials = build_trial_sequence(practice_trials_raw, N_PRACTICE_TRIALS)
    if not prepared_trial_file.exists():
        raise FileNotFoundError(f"Missing prepared stimulus file: {prepared_trial_file}")
    main_image_dir = prepared_stimuli_root / "images"
    main_trials = build_prepared_trial_sequence(load_prepared_trials(prepared_trial_file), n_main_trials)
    main_trial_source = prepared_trial_file
    practice_trial_source = practice_trial_file
    main_check_trials = choose_check_trials(n_main_trials, CHECK_RATE)

    run_timestamp = build_run_timestamp()
    participant_slug = sanitize_path_component(participant)
    run_id = f"{mode}_{participant_slug}_s{session}_{run_timestamp}"
    output_layout = build_run_output_layout(project_root, participant, run_timestamp)
    output_root = Path(output_layout["output_root"])
    participant_dir = Path(output_layout["participant_dir"])
    run_dir = Path(output_layout["run_dir"])
    main_output_csv = Path(output_layout["main_trials_csv"])
    main_gaze_csv = Path(output_layout["main_gaze_csv"])
    practice_output_csv = Path(output_layout["practice_trials_csv"])
    practice_gaze_csv = Path(output_layout["practice_gaze_csv"])
    main_sequence_csv = Path(output_layout["main_sequence_csv"])
    qc_png = Path(output_layout["qc_png"])
    participant_index_csv = Path(output_layout["participant_index_csv"])
    participant_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_started_wall = datetime.now()
    run_started_at = run_started_wall.isoformat(timespec="seconds")
    run_finished_at = ""
    run_duration_s = 0.0
    run_status = "in_progress"
    run_error = ""

    session_clock = core.MonotonicClock()
    win = visual.Window(
        size=cfg.WINDOW_SIZE,
        fullscr=cfg.FULLSCREEN,
        color=cfg.WINDOW_COLOR,
        units=cfg.WINDOW_UNITS,
    )
    mouse = event.Mouse(win=win)
    mouse.setVisible(False)
    key_state = event.pyglet.window.key.KeyStateHandler()
    win.winHandle.push_handlers(key_state)
    quit_state = QuitState()
    record_cursor_samples = mode in {"real", "test"}
    qc_path: Path | None = None
    run_finished_wall: datetime | None = None
    quit_hint = visual.TextStim(
        win,
        text="",
        color="#ffd166",
        height=0.03,
        pos=(0, 0.45),
    )

    practice_cache = build_image_cache(win, practice_trials, practice_image_dir)
    main_cache = build_image_cache(win, main_trials, main_image_dir)

    trial_fieldnames = [
        "run_id",
        "phase",
        "participant",
        "session",
        "trial_index",
        "trial_id",
        "stimulus_id",
        "description_condition",
        "metadata_condition",
        "description",
        "image_file",
        "original_filename",
        "original_style_folder",
        "original_relative_path",
        "trial_start_session_s",
        "trial_start_unix_s",
        "description_onset_session_s",
        "description_offset_session_s",
        "description_onset_unix_s",
        "description_offset_unix_s",
        "description_time_s",
        "image_onset_session_s",
        "image_offset_session_s",
        "image_onset_unix_s",
        "image_offset_unix_s",
        "image_view_time_s",
        "rating_onset_session_s",
        "rating_offset_session_s",
        "rating_onset_unix_s",
        "rating_offset_unix_s",
        "rating",
        "rating_rt_s",
        "mismatch_check_triggered",
        "mismatch_response",
        "mismatch_rt_s",
        "mismatch_onset_session_s",
        "mismatch_offset_session_s",
        "mismatch_onset_unix_s",
        "mismatch_offset_unix_s",
    ]

    sequence_fieldnames = [
        "run_id",
        "phase",
        "participant",
        "session",
        "trial_index",
        "stimulus_id",
        "description_condition",
        "metadata_condition",
        "description",
        "image_file",
        "original_filename",
        "original_style_folder",
        "original_relative_path",
    ]

    gaze_fieldnames = [
        "run_id",
        "participant",
        "session",
        "trial_index",
        "trial_id",
        "image_file",
        "sample_index",
        "sample_session_s",
        "sample_unix_s",
        "sample_from_image_onset_s",
        "gaze_x_pix",
        "gaze_y_pix",
        "units",
    ]

    if mode == "test":
        break_text = "Break screens are disabled in this short test block."
        instruction_text = (
            "Welcome.\n\n"
            "Test mode is active.\n\n"
            "You will first complete 2 practice trials.\n"
            f"Then the main block starts ({n_main_trials} trials).\n\n"
            "Image viewing has a minimum duration of 10 seconds.\n"
            "Some trials include a short text check question.\n"
            f"{break_text}\n\n"
            "Press Q twice quickly to quit at any time.\n\n"
            "Press SPACE to start."
        )
        practice_intro_text = (
            "Practice block starts now (2 trials).\n\n"
            "These images and outputs are stored separately from main data.\n\n"
            "Press SPACE to continue."
        )
    elif mode == "rg":
        break_text = "You will get break screens during the main block."
        instruction_text = (
            "Welcome.\n\n"
            "RG mode is active (research-grade tracker integration mode).\n\n"
            "You will first complete 2 practice trials.\n"
            f"Then the main block starts ({n_main_trials} trials).\n\n"
            "Some trials include a short text check question.\n"
            f"{break_text}\n\n"
            "Press Q twice quickly to quit at any time.\n\n"
            "Press SPACE to start."
        )
        practice_intro_text = (
            "Practice block starts now (2 trials).\n\n"
            "Press SPACE to continue."
        )
    else:
        break_text = "You will get break screens during the main block."
        instruction_text = (
            "Welcome.\n\n"
            "You will first complete 2 practice trials.\n"
            f"Then the main block starts ({n_main_trials} trials).\n\n"
            "Some trials include a short text check question.\n"
            f"{break_text}\n\n"
            "Press Q twice quickly to quit at any time.\n\n"
            "Press SPACE to start."
        )
        practice_intro_text = (
            "Practice block starts now (2 trials).\n\n"
            "Press SPACE to continue."
        )

    instruction = visual.TextStim(
        win,
        text=instruction_text,
        color="white",
        height=0.04,
        wrapWidth=1.45,
    )

    practice_intro = visual.TextStim(
        win,
        text=practice_intro_text,
        color="white",
        height=0.042,
        wrapWidth=1.45,
    )

    main_intro = visual.TextStim(
        win,
        text=(
            f"Main block starts now ({n_main_trials} trials).\n\n"
            "Please stay focused and keep your posture stable.\n"
            "Use the full range of the rating scale when appropriate.\n\n"
            "Press SPACE to continue."
        ),
        color="white",
        height=0.042,
        wrapWidth=1.45,
    )

    try:
        wait_for_continue(win, instruction, session_clock, quit_state, quit_hint)
        wait_for_continue(win, practice_intro, session_clock, quit_state, quit_hint)

        run_phase(
            phase_name="practice",
            win=win,
            mouse=mouse,
            key_state=key_state,
            session_clock=session_clock,
            quit_state=quit_state,
            quit_hint=quit_hint,
            run_id=run_id,
            participant=participant,
            session=session,
            trials=practice_trials,
            image_dir=practice_image_dir,
            image_cache=practice_cache,
            output_csv=practice_output_csv,
            gaze_csv=practice_gaze_csv,
            trial_fieldnames=trial_fieldnames,
            gaze_fieldnames=gaze_fieldnames,
            record_cursor_samples=record_cursor_samples,
            sequence_csv=None,
            sequence_fieldnames=None,
            check_trials=None,
            enable_breaks=False,
            show_min_view_countdown=(mode == "test"),
            allow_fast_skip=(mode == "test"),
        )

        wait_for_continue(win, main_intro, session_clock, quit_state, quit_hint)

        run_phase(
            phase_name="main",
            win=win,
            mouse=mouse,
            key_state=key_state,
            session_clock=session_clock,
            quit_state=quit_state,
            quit_hint=quit_hint,
            run_id=run_id,
            participant=participant,
            session=session,
            trials=main_trials,
            image_dir=main_image_dir,
            image_cache=main_cache,
            output_csv=main_output_csv,
            gaze_csv=main_gaze_csv,
            trial_fieldnames=trial_fieldnames,
            gaze_fieldnames=gaze_fieldnames,
            record_cursor_samples=record_cursor_samples,
            sequence_csv=main_sequence_csv,
            sequence_fieldnames=sequence_fieldnames,
            check_trials=main_check_trials,
            enable_breaks=(mode != "test"),
            show_min_view_countdown=(mode == "test"),
            allow_fast_skip=(mode == "test"),
        )

        finished_screen = visual.TextStim(
            win,
            text=(
                "Experiment finished.\n\n"
                "You can now leave.\n\n"
                "Press SPACE to continue to the final researcher screen."
            ),
            color="white",
            height=0.05,
            wrapWidth=1.45,
        )
        wait_for_space_screen(win, session_clock, quit_state, quit_hint, finished_screen)

        loading_screen = visual.TextStim(
            win,
            text="Preparing run summary and quality check...",
            color="white",
            height=0.04,
            wrapWidth=1.45,
        )
        loading_screen.draw()
        win.flip()

        main_rows = read_rows_for_run(main_output_csv, run_id, phase="main")
        gaze_rows = read_rows_for_run(main_gaze_csv, run_id) if record_cursor_samples else []
        qc_path, qc_summary = create_qc_plot(
            run_id=run_id,
            mode=mode,
            main_rows=main_rows,
            gaze_rows=gaze_rows,
            output_dir=run_dir,
            expected_main_trials=n_main_trials,
            expect_cursor_samples=record_cursor_samples,
        )
        checklist = build_validity_checklist(
            main_rows=main_rows,
            gaze_rows=gaze_rows,
            mode=mode,
            expected_main_trials=n_main_trials,
            expect_cursor_samples=record_cursor_samples,
        )

        qc_image_stim = None
        if qc_path is not None and qc_path.exists():
            qc_image_stim = visual.ImageStim(
                win,
                image=str(qc_path),
                pos=(0, 0.18),
                size=(1.28, 0.60),
                units="height",
            )

        qc_text = (
            "\n".join(qc_summary[: cfg.QC_SUMMARY_LINES]) if qc_summary else "No QC summary available."
        )
        checklist_text = "\n".join(checklist)
        done_title = visual.TextStim(
            win,
            text=(
                "Run Quality Check"
            ),
            color="white",
            height=0.038,
            pos=(0, 0.49),
            wrapWidth=1.2,
        )
        panel = visual.Rect(
            win,
            width=1.36,
            height=0.42,
            pos=(0, -0.30),
            fillColor=(-0.85, -0.85, -0.85),
            lineColor="#666666",
            lineWidth=1.5,
        )
        summary_heading = visual.TextStim(
            win,
            text="Summary",
            color="#ffffff",
            height=0.026,
            pos=(-0.62, -0.13),
            wrapWidth=0.6,
            alignText="left",
            anchorHoriz="left",
        )
        summary_body = visual.TextStim(
            win,
            text=(
                f"{qc_text}\n\n"
                f"Output root: {output_root.relative_to(project_root)}\n"
                f"Run folder: {run_dir.relative_to(project_root)}\n"
                f"QC figure: {qc_path.relative_to(project_root) if qc_path is not None else 'not available'}\n"
                f"Sequence manifest: {main_sequence_csv.relative_to(project_root)}\n"
                f"Latest participant index: {participant_index_csv.relative_to(project_root)}\n"
                f"Stimulus source: {main_trial_source.relative_to(project_root)}\n"
                f"Main trials: {main_output_csv.relative_to(project_root)}\n"
                f"Practice trials: {practice_output_csv.relative_to(project_root)}"
            ),
            color="#f4f4f4",
            height=0.018,
            pos=(-0.62, -0.32),
            wrapWidth=0.60,
            alignText="left",
            anchorHoriz="left",
        )
        checklist_heading = visual.TextStim(
            win,
            text="Validity Checklist",
            color="#ffffff",
            height=0.026,
            pos=(0.03, -0.13),
            wrapWidth=0.62,
            alignText="left",
            anchorHoriz="left",
        )
        checklist_body = visual.TextStim(
            win,
            text=checklist_text,
            color="#f4f4f4",
            height=0.017,
            pos=(0.03, -0.32),
            wrapWidth=0.62,
            alignText="left",
            anchorHoriz="left",
        )
        done_footer = visual.TextStim(
            win,
            text="Hold SPACE for 1 second to close.",
            color="#ffffff",
            height=0.022,
            pos=(0, -0.49),
            wrapWidth=1.2,
        )

        event.clearEvents(eventType="keyboard")
        hold_close_s = 0.0
        last_close_t = session_clock.getTime()
        while True:
            now_session = session_clock.getTime()
            dt = min(max(now_session - last_close_t, 0.0), 0.05)
            last_close_t = now_session
            quit_hint.text = quit_state.active_message(now_session)
            if qc_image_stim is not None:
                qc_image_stim.draw()
            done_title.draw()
            panel.draw()
            summary_heading.draw()
            summary_body.draw()
            checklist_heading.draw()
            checklist_body.draw()
            done_footer.draw()
            if quit_hint.text:
                quit_hint.draw()
            win.flip()
            keys = event.getKeys(keyList=["q"])
            now_session = session_clock.getTime()
            quit_state.process_keys(keys, now_session)
            if bool(key_state[event.pyglet.window.key.SPACE]):
                hold_close_s += dt
            else:
                hold_close_s = 0.0
            if hold_close_s >= 1.0:
                run_status = "completed"
                run_finished_wall = datetime.now()
                run_finished_at = run_finished_wall.isoformat(timespec="seconds")
                run_duration_s = (run_finished_wall - run_started_wall).total_seconds()
                break

    except UserAbort:
        run_status = "aborted"
        run_finished_wall = datetime.now()
        run_finished_at = run_finished_wall.isoformat(timespec="seconds")
        run_duration_s = (run_finished_wall - run_started_wall).total_seconds()
        run_error = ""
        abort_msg = visual.TextStim(
            win,
            text="Session ended early. Partial data has been saved.",
            color="white",
            height=0.05,
        )
        abort_msg.draw()
        win.flip()
        core.wait(1.5)

    except Exception as exc:
        run_status = "error"
        run_error = f"{type(exc).__name__}: {exc}"
        run_finished_wall = datetime.now()
        run_finished_at = run_finished_wall.isoformat(timespec="seconds")
        run_duration_s = (run_finished_wall - run_started_wall).total_seconds()
        error_msg = visual.TextStim(
            win,
            text=(
                "Run failed.\n\n"
                f"{run_error}\n\n"
                "Partial data, if any, has been saved.\n"
                "Press SPACE to close."
            ),
            color="white",
            height=0.045,
            wrapWidth=1.45,
        )
        error_msg.draw()
        win.flip()
        event.waitKeys(keyList=["space"])

    finally:
        if run_status == "in_progress":
            run_status = "completed"
        if not run_finished_at:
            run_finished_wall = datetime.now()
            run_finished_at = run_finished_wall.isoformat(timespec="seconds")
            run_duration_s = (run_finished_wall - run_started_wall).total_seconds()
        latest_participant_row = build_latest_participant_row(
            project_root=project_root,
            participant=participant,
            participant_dir=participant_dir,
            run_dir=run_dir,
            mode=mode,
            session=session,
            run_id=run_id,
            run_started_at=run_started_at,
            run_finished_at=run_finished_at,
            run_duration_s=run_duration_s,
            run_status=run_status,
            run_error=run_error,
            main_output_csv=main_output_csv,
            main_gaze_csv=main_gaze_csv,
            practice_output_csv=practice_output_csv,
            practice_gaze_csv=practice_gaze_csv,
            main_sequence_csv=main_sequence_csv,
            qc_png=qc_path if qc_path is not None else qc_png,
            n_main_trials_expected=n_main_trials,
            n_practice_trials_expected=N_PRACTICE_TRIALS,
            record_cursor_samples=record_cursor_samples,
            main_trial_source=main_trial_source,
            practice_trial_source=practice_trial_source,
        )
        upsert_latest_row(
            participant_index_csv,
            latest_participant_row,
            [
                "participant",
                "participant_folder",
                "run_folder",
                "run_id",
                "mode",
                "session",
                "status",
                "error_message",
                "run_started_at",
                "run_finished_at",
                "run_duration_s",
                "n_main_trials_expected",
                "n_main_trials_recorded",
                "n_practice_trials_expected",
                "n_practice_trials_recorded",
                "n_main_gaze_rows",
                "n_practice_gaze_rows",
                "record_cursor_samples",
                "main_trials_csv",
                "main_gaze_csv",
                "practice_trials_csv",
                "practice_gaze_csv",
                "main_sequence_csv",
                "qc_png",
                "main_stimulus_source",
                "practice_stimulus_source",
            ],
            "participant",
        )
        win.close()
        core.quit()


if __name__ == "__main__":
    main()
