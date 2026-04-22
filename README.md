# TrackingScript (PsychoPy)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python scripts/study_task.py
```

Choose mode at startup:
- `real`
- `test`
- `rg` (research graded eye tracker mode)

You can also set `main_trials` in the same start dialog.
- leave blank to use mode defaults (`real=60`, `test` from config, `rg=60`)
- or enter any positive integer for a custom run length

## Current Flow

- Practice block: `2` trials
- Main block:
  - `real`: `60` trials
  - `test`: from `N_TRIALS_TEST` in config
  - `rg`: `60` trials (default)
- Main trials randomized
- Manipulation check on ~15% of main trials
- Breaks after trial 20 and 40 in `real` mode only (minimum 20s)
- Image phase minimum duration: 10s (normal continue with `SPACE`)
- Rating input: hold `LEFT/RIGHT` for smooth slider movement, `SPACE` confirms
- Hidden quit: press `Q` twice within 2s

## Hidden Test Shortcuts

Only in `test` mode:
- Double-`F` skips image phase immediately (bypasses 10s minimum)
- Double-`F` skips break screens

No on-screen indicators are shown for these shortcuts.

## Cursor Proxy Logging

- `test` mode: cursor-proxy samples are recorded during image phase
- `real` mode: cursor-proxy samples are recorded during image phase
- `rg` mode: cursor-proxy sampling is disabled by design (placeholder for external tracker input)

Cursor is hidden during the task.
Saved cursor rows are resampled to a fixed rate (`CURSOR_TARGET_HZ`, default `60 Hz`) from
timestamped raw points, so the stored sample count follows duration instead of render-loop FPS.

## End-of-Run Screens

At successful completion, two `SPACE`-gated end screens are shown:
1. `Experiment finished` screen (participant can leave)
2. QC/results screen with:
   - automatic run-quality summary
   - automatic validity checklist (pass/warn/fail checks)
   - generated QC figure path
   - output file paths

QC loading starts only after the first screen is acknowledged with `SPACE`.

The program does not auto-close these screens; it waits for `SPACE`.

## Automatic QC Visualization

After each completed run, a QC plot is generated automatically:
- location: `<main output folder>/qc/<run_id>_qc.png`
- includes:
  - image-view time by trial (+ min-time reference line)
  - rating by trial
  - cursor sample counts by trial (when available)

## Files

Trial configs:
- main: `stimuli/trials.csv`
- practice: `stimuli/practice/trials.csv`

Required columns:
- `trial_id`
- `description`
- `image_file`

Image folders:
- main: `images/`
- practice: `images/practice/`

Outputs:
- real main trials: `runs/all_runs.csv`
- real practice trials: `runs/practice/all_runs_practice.csv`
- test main trials: `runs/test/all_runs_test.csv`
- test practice trials: `runs/test/practice/all_runs_practice_test.csv`
- test main gaze samples: `runs/test/gaze_samples_test.csv`
- test practice gaze samples: `runs/test/practice/gaze_samples_practice_test.csv`

## Tunable Parameters

Use `scripts/study_config.py` as the central place for quick parameter changes.

This includes:
- trial counts (`N_TRIALS_REAL`, `N_TRIALS_TEST`, `N_PRACTICE_TRIALS`)
- timing and checks (`MIN_IMAGE_VIEW_S`, `CHECK_RATE`, breaks, key windows)
- sampling (`GAZE_SAMPLE_INTERVAL_S`, `CURSOR_TARGET_HZ`)
- slider behavior (`RATING_*`)
- window/display settings
- QC summary layout (`QC_SUMMARY_LINES`)
# IRII-Project
