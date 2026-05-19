from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "irii_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_IMAGE_DIR = PROJECT_ROOT / "input" / "main" / "images"
PRACTICE_IMAGE_DIR = PROJECT_ROOT / "input" / "practice" / "images"
REALRUN1_OUTPUT_ROOT = PROJECT_ROOT / "output" / "Realrun 1 output"
MAX_IMAGE_WIDTH_HEIGHT_UNITS = 1.55
MAX_IMAGE_HEIGHT_HEIGHT_UNITS = 0.82
COMMON_SCREEN_HEIGHTS = (720, 768, 900, 1080, 1200, 1440, 2160)


def first_numeric(series: pd.Series, default: float | None = None) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.iloc[0])


def resolve_image_path(image_dir: Path, image_file: str) -> Path | None:
    requested = image_dir / image_file
    if requested.exists():
        return requested
    if not image_dir.exists():
        return None
    lowered = image_file.lower()
    for candidate in image_dir.iterdir():
        if candidate.is_file() and candidate.name.lower() == lowered:
            return candidate
    stem = Path(image_file).stem
    matches = [candidate for candidate in image_dir.iterdir() if candidate.is_file() and candidate.stem == stem]
    return sorted(matches)[0] if matches else None


def infer_screen_height(gaze: pd.DataFrame) -> int:
    max_abs = float(np.nanmax(np.abs(gaze[["gaze_x_pix", "gaze_y_pix"]].to_numpy())))
    needed = max_abs * 2.0
    for height in COMMON_SCREEN_HEIGHTS:
        if height >= needed:
            return height
    return int(math.ceil(needed / 100.0) * 100)


def fit_image_size_height_units(image: Image.Image) -> tuple[float, float]:
    width_px, height_px = image.size
    if width_px <= 0 or height_px <= 0:
        return MAX_IMAGE_WIDTH_HEIGHT_UNITS, MAX_IMAGE_HEIGHT_HEIGHT_UNITS
    aspect = width_px / height_px
    width = MAX_IMAGE_HEIGHT_HEIGHT_UNITS * aspect
    height = MAX_IMAGE_HEIGHT_HEIGHT_UNITS
    if width > MAX_IMAGE_WIDTH_HEIGHT_UNITS:
        width = MAX_IMAGE_WIDTH_HEIGHT_UNITS
        height = MAX_IMAGE_WIDTH_HEIGHT_UNITS / aspect
    return width, height


def map_gaze_to_image(
    gaze: pd.DataFrame,
    image: Image.Image,
    screen_height_px: int,
) -> pd.DataFrame:
    if {"image_stim_width_units", "image_stim_height_units"}.issubset(gaze.columns):
        stim_w_units = first_numeric(gaze["image_stim_width_units"])
        stim_h_units = first_numeric(gaze["image_stim_height_units"])
    else:
        stim_w_units = None
        stim_h_units = None
    if stim_w_units is None or stim_h_units is None:
        stim_w_units, stim_h_units = fit_image_size_height_units(image)
    center_x_units = first_numeric(gaze["image_stim_center_x_units"], 0.0) if "image_stim_center_x_units" in gaze.columns else 0.0
    center_y_units = first_numeric(gaze["image_stim_center_y_units"], 0.0) if "image_stim_center_y_units" in gaze.columns else 0.0
    stim_w_px = stim_w_units * screen_height_px
    stim_h_px = stim_h_units * screen_height_px
    center_x_px = center_x_units * screen_height_px
    center_y_px = center_y_units * screen_height_px
    mapped = gaze.copy()
    mapped["image_x"] = (mapped["gaze_x_pix"] - (center_x_px - stim_w_px / 2.0)) / stim_w_px * image.width
    mapped["image_y"] = ((center_y_px + stim_h_px / 2.0) - mapped["gaze_y_pix"]) / stim_h_px * image.height
    mapped["inside_image"] = (
        mapped["image_x"].between(0, image.width)
        & mapped["image_y"].between(0, image.height)
    )
    return mapped


def make_trial_overlay(
    trial_gaze: pd.DataFrame,
    image_path: Path,
    output_path: Path,
    screen_height_px: int,
    title: str,
) -> dict[str, float | int | str]:
    with Image.open(image_path) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert("RGB")
    mapped = map_gaze_to_image(trial_gaze, image, screen_height_px)
    inside = mapped[mapped["inside_image"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    axes[0].imshow(image)
    if len(mapped):
        image_x = mapped["image_x"].to_numpy(dtype=float)
        image_y = mapped["image_y"].to_numpy(dtype=float)
        colors = mapped["sample_from_image_onset_s"].to_numpy(dtype=float)
        axes[0].scatter(
            image_x,
            image_y,
            c=colors,
            cmap="viridis",
            s=16,
            alpha=0.62,
            edgecolors="none",
        )
        axes[0].plot(image_x, image_y, color="white", alpha=0.22, linewidth=0.8)
    axes[0].set_title("Gaze over stimulus\ncolor = time since image onset")
    axes[0].set_xlim(0, image.width)
    axes[0].set_ylim(image.height, 0)
    axes[0].axis("off")

    axes[1].scatter(
        mapped["gaze_x_pix"].to_numpy(dtype=float),
        mapped["gaze_y_pix"].to_numpy(dtype=float),
        c=mapped["sample_from_image_onset_s"].to_numpy(dtype=float),
        cmap="viridis",
        s=14,
        alpha=0.72,
        edgecolors="none",
    )
    if {"image_stim_width_units", "image_stim_height_units"}.issubset(trial_gaze.columns):
        stim_w_units = first_numeric(trial_gaze["image_stim_width_units"])
        stim_h_units = first_numeric(trial_gaze["image_stim_height_units"])
    else:
        stim_w_units = None
        stim_h_units = None
    if stim_w_units is None or stim_h_units is None:
        stim_w_units, stim_h_units = fit_image_size_height_units(image)
    center_x_units = first_numeric(trial_gaze["image_stim_center_x_units"], 0.0) if "image_stim_center_x_units" in trial_gaze.columns else 0.0
    center_y_units = first_numeric(trial_gaze["image_stim_center_y_units"], 0.0) if "image_stim_center_y_units" in trial_gaze.columns else 0.0
    stim_w_px = stim_w_units * screen_height_px
    stim_h_px = stim_h_units * screen_height_px
    center_x_px = center_x_units * screen_height_px
    center_y_px = center_y_units * screen_height_px
    rect = plt.Rectangle(
        (center_x_px - stim_w_px / 2.0, center_y_px - stim_h_px / 2.0),
        stim_w_px,
        stim_h_px,
        fill=False,
        color="red",
        linewidth=2,
        label="displayed image bounds",
    )
    axes[1].add_patch(rect)
    axes[1].axhline(0, color="black", alpha=0.25, linewidth=1)
    axes[1].axvline(0, color="black", alpha=0.25, linewidth=1)
    axes[1].set_title("Screen-centered gaze coordinates")
    axes[1].set_xlabel("x pixels from screen center")
    axes[1].set_ylabel("y pixels from screen center")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].legend(loc="best")

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    duration = float(mapped["sample_from_image_onset_s"].max()) if len(mapped) else 0.0
    return {
        "trial_index": int(trial_gaze["trial_index"].iloc[0]),
        "image_file": str(trial_gaze["image_file"].iloc[0]),
        "samples": int(len(mapped)),
        "duration_s": round(duration, 3),
        "inside_image_fraction": round(float(mapped["inside_image"].mean()), 4) if len(mapped) else 0.0,
        "median_x": round(float(mapped["gaze_x_pix"].median()), 2) if len(mapped) else 0.0,
        "median_y": round(float(mapped["gaze_y_pix"].median()), 2) if len(mapped) else 0.0,
        "image_path_found": str(image_path),
    }


def make_run_summary(summary: pd.DataFrame, output_path: Path, phase: str, run_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    axes[0, 0].bar(summary["trial_index"].astype(str), summary["samples"])
    axes[0, 0].set_title("Samples per trial")
    axes[0, 0].set_xlabel("trial")
    axes[0, 0].set_ylabel("samples")

    axes[0, 1].bar(summary["trial_index"].astype(str), summary["inside_image_fraction"])
    axes[0, 1].set_title("Fraction of samples inside displayed image")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_xlabel("trial")

    axes[1, 0].bar(summary["trial_index"].astype(str), summary["duration_s"])
    axes[1, 0].set_title("Image viewing duration from gaze samples")
    axes[1, 0].set_xlabel("trial")
    axes[1, 0].set_ylabel("seconds")

    axes[1, 1].scatter(summary["median_x"], summary["median_y"], s=80)
    for _, row in summary.iterrows():
        axes[1, 1].annotate(str(row["trial_index"]), (row["median_x"], row["median_y"]))
    axes[1, 1].axhline(0, color="black", alpha=0.25)
    axes[1, 1].axvline(0, color="black", alpha=0.25)
    axes[1, 1].set_title("Median gaze by trial")
    axes[1, 1].set_xlabel("x pixels from center")
    axes[1, 1].set_ylabel("y pixels from center")

    fig.suptitle(f"{phase} gaze QC: {run_dir}")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def latest_run_dir(base: Path, phase: str) -> Path:
    run_dirs = sorted(
        path
        for path in base.glob("participant_*/*")
        if path.is_dir() and (path / f"{phase}_gaze.csv").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run folders with {phase}_gaze.csv found under {base}")
    return run_dirs[-1]


def run_dirs_with_phase(base: Path, phase: str) -> list[Path]:
    return sorted(
        path
        for path in base.glob("participant_*/*")
        if path.is_dir() and (path / f"{phase}_gaze.csv").exists()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize study gaze CSV files over the presented images.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run folder containing main_gaze.csv and main_trials.csv")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output", help="Root containing real participant_* run folders")
    parser.add_argument("--testrun-root", type=Path, default=PROJECT_ROOT / "output" / "Testrun Output", help="Root containing old Testrun Output folders")
    parser.add_argument("--testrun", action="store_true", help="Use output/Testrun Output instead of real output/participant_* runs")
    parser.add_argument("--realrun1", action="store_true", help="Use output/Realrun 1 output as the run root")
    parser.add_argument("--all-runs", action="store_true", help="Visualize every run under the selected root that has the requested phase CSV")
    parser.add_argument("--phase", choices=["main", "practice"], default="main")
    parser.add_argument("--screen-height", type=int, default=None, help="Fullscreen height used during recording. Auto-inferred if omitted.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def visualize_run(args: argparse.Namespace, run_dir: Path, image_dir: Path) -> pd.DataFrame:
    if args.output_dir is not None and args.all_runs:
        output_dir = args.output_dir / run_dir.parent.name / run_dir.name
    else:
        output_dir = args.output_dir or run_dir / "gaze_visualization"

    gaze_csv = run_dir / f"{args.phase}_gaze.csv"
    if not gaze_csv.exists():
        raise FileNotFoundError(f"Missing gaze CSV: {gaze_csv}")

    gaze = pd.read_csv(gaze_csv)
    if gaze.empty:
        raise ValueError(f"No gaze rows in {gaze_csv}")
    recorded_screen_height = first_numeric(gaze["screen_height_px"]) if "screen_height_px" in gaze.columns else None
    screen_height_px = (
        args.screen_height
        or (int(recorded_screen_height) if recorded_screen_height is not None else infer_screen_height(gaze))
    )

    rows = []
    for trial_index, trial_gaze in gaze.groupby("trial_index", sort=True):
        image_file = str(trial_gaze["image_file"].iloc[0])
        image_path = resolve_image_path(image_dir, image_file)
        if image_path is None:
            rows.append(
                {
                    "trial_index": int(trial_index),
                    "image_file": image_file,
                    "samples": int(len(trial_gaze)),
                    "duration_s": round(float(trial_gaze["sample_from_image_onset_s"].max()), 3),
                    "inside_image_fraction": 0.0,
                    "median_x": round(float(trial_gaze["gaze_x_pix"].median()), 2),
                    "median_y": round(float(trial_gaze["gaze_y_pix"].median()), 2),
                    "image_path_found": "",
                }
            )
            continue
        title = f"{args.phase} trial {trial_index}: {image_file} | samples={len(trial_gaze)} | screen height={screen_height_px}px"
        rows.append(
            make_trial_overlay(
                trial_gaze=trial_gaze,
                image_path=image_path,
                output_path=output_dir / f"{args.phase}_trial_{int(trial_index):03d}_{Path(image_file).stem}.png",
                screen_height_px=screen_height_px,
                title=title,
            )
        )

    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"{args.phase}_gaze_summary.csv", index=False)
    make_run_summary(summary, output_dir / f"{args.phase}_run_summary.png", args.phase, run_dir)

    print(f"Run: {run_dir}")
    print(f"Phase: {args.phase}")
    print(f"Screen height used for mapping: {screen_height_px}px")
    print(f"Output: {output_dir}")
    print(summary.to_string(index=False))
    return summary


def selected_root(args: argparse.Namespace) -> Path:
    if args.realrun1:
        return REALRUN1_OUTPUT_ROOT
    if args.testrun:
        return args.testrun_root
    return args.output_root


def main() -> None:
    args = parse_args()
    default_root = selected_root(args)
    image_dir = MAIN_IMAGE_DIR if args.phase == "main" else PRACTICE_IMAGE_DIR

    if args.all_runs:
        run_dirs = run_dirs_with_phase(default_root, args.phase)
        if not run_dirs:
            raise FileNotFoundError(f"No runs with {args.phase}_gaze.csv found under {default_root}")
        for run_dir in run_dirs:
            visualize_run(args, run_dir, image_dir)
        print(f"Visualized {len(run_dirs)} run(s) under {default_root}")
        return

    run_dir = args.run_dir or latest_run_dir(default_root, args.phase)
    visualize_run(args, run_dir, image_dir)


if __name__ == "__main__":
    main()
