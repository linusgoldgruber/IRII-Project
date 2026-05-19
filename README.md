# TrackingScript (PsychoPy)

## Setup For The Study Task

Use conda for `scripts/study_task.py`, especially on Windows. This avoids
building PsychoPy media dependencies such as `ffpyplayer` through pip.

Install Miniforge from:

```text
https://conda-forge.org/download/
```

Then open "Miniforge Prompt" or "Anaconda Prompt" in the project folder and run:

```bash
conda env create -f environment-study.yml
conda activate irii-study
python scripts/study_task.py
```

If the environment already exists and you changed dependencies:

```bash
conda env update -f environment-study.yml --prune
conda activate irii-study
```

## Lightweight AOI-Only Setup

The AOI editor does not need PsychoPy. If you only want to edit AOIs, this is
faster:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pillow
```

## Run

```bash
conda activate irii-study
python scripts/study_task.py
```

Choose the mode in the startup dialog:
- `real`
- `test`
- `rg`
- `check_images`

Choose the participant's task language in the same dialog:
- `de`
- `en`

You can also set `main_trials` in the same dialog.
- leave it blank to use the mode default
- or enter any positive integer up to the 80-image bank for a custom run length

`real` and `rg` require a participant and session value. Blank participant or
session values are allowed only for local test/check workflows.

Mode behavior:

- `real`: full study task, 60 main trials by default, cursor-proxy sampling on,
  main-block breaks on.
- `test`: short study task, 3 main trials by default, cursor-proxy sampling on,
  main-block breaks off.
- `rg`: research-grade tracker integration mode, 60 main trials by default,
  cursor-proxy sampling off, main-block breaks on.
- `check_images`: researcher-only image browser for `input/main/images`.
  Click, `SPACE`, or right arrow advances; left arrow goes back; Escape exits.

## Input

All study inputs live under `input/`:

- `input/main/stimuli.csv`
- `input/main/images/`
- `input/practice/stimuli.csv`
- `input/practice/images/`

The main stimulus CSV stores the prepared image-description mappings. It uses
one row per stimulus with separate prepared descriptions for `congruent`,
`ambiguous`, and `incongruent` conditions. German descriptions are stored in
the base condition columns; parallel English versions are stored in
`condition_congruent_en`, `condition_semi_congruent_en`, and
`condition_incongruent_en`.

To review and edit those prompt texts next to each artwork image, run:

```bash
python scripts/prompt_editor.py
```

The prompt editor opens `input/main/stimuli.csv`, shows the images in stimulus
order, and lets you edit the German and English congruent, semi-congruent, and
incongruent prompts. `Save CSV` writes directly back to
`input/main/stimuli.csv`. The editor creates a timestamped backup in
`input/main/stimuli_backups/` on launch.

To run a model-assisted prompt audit, set an OpenAI API key and run:

```bash
export OPENAI_API_KEY=...
python scripts/audit_stimulus_prompts.py
```

The audit writes a timestamped folder under `output/prompt_audit/` with:

- `review.html`: image-by-image review packet
- `suggestions.csv`: current prompts, model inventory, and suggested prompts
- `audit.jsonl`: raw row-by-row audit data

The audit does not modify `input/main/stimuli.csv`. To accept suggestions,
open `suggestions.csv`, set `approved` to `yes` only for rows you want to use,
then run:

```bash
python scripts/audit_stimulus_prompts.py --apply-approved output/prompt_audit/<timestamp>/suggestions.csv
```

Applying approved rows creates a backup in `input/main/stimuli_backups/` before
writing to `input/main/stimuli.csv`.

The prepared main bank is capped at 80 images. The task samples from that bank
without replacement for the requested run length.

The practice CSV stays separate but uses the same simple flat layout. It stores
German practice descriptions in `description` and English practice descriptions
in `description_en`. The current practice images are WikiArt practice-only JPEGs
in:

- `input/practice/images/wikiart_practice/`
- `input/practice/images/wikiart_practice/manifest.csv`

The older `.ppm` practice placeholders were moved out of the active image
folder to avoid AOI/setup confusion:

- `input/practice/old_ppm_test_images/`

The loader resolves common raster formats through Pillow, so `jpg`, `png`,
`ppm`, `bmp`, `tif`, and similar files are all displayable as long as they
exist in the input folder.

## Study Flow

Every run starts with 2 practice trials, then the main block.

The chosen language is applied consistently to practice and main descriptions
for the whole run. It is recorded in trial rows, `main_sequence.csv`,
`run_metadata.json`, and `participants_latest.csv`.

Each trial follows this order:

1. Description screen. Participant presses `SPACE`.
2. Fixation cross for `1.25s`.
3. Artwork display. No progress label, debug filename, `.ppm` label, or
   `Press SPACE` instruction is shown over the artwork. The image remains until
   the participant presses `SPACE` after the minimum viewing time.
4. Rating screen. Participant moves the slider with left/right arrows and
   confirms with `SPACE`.
5. Optional text check question on selected main trials.

Timing and flow defaults live in `scripts/study_config.py`:

- `MIN_IMAGE_VIEW_S = 8.0`
- `FIXATION_DURATION_S = 1.25`
- `CONTINUE_KEY_BUFFER_S = 0.5`
- `BREAK_AFTER_TRIALS = (20, 40)`
- `BREAK_MIN_S = 20.0`
- `CHECK_RATE = 0.15`

The `SPACE` key is gated on continue/confirm screens. One press advances only
one screen. To advance again, the participant must release `SPACE`, wait through
the `0.5s` buffer, and press it again. This prevents a held key from skipping
the next screen.

Hidden controls:

- Press `Q` twice within `1.0s` to abort a run.
- Double-press `F` within `1.0s` to fast-skip image viewing when the hidden
  fast-skip path is enabled.

Break screens appear after main trials 20 and 40 in `real` and `rg` mode.
They are disabled in `test` mode.

## Trial Selection

Main trials are randomized per run. The task records a random seed in
`run_metadata.json` and in `participants_latest.csv`.

Description conditions are counterbalanced by participant ID:

- Numeric participant IDs use the last number in the participant code.
- Non-numeric participant IDs use a stable hash.
- Trial rows store both `counterbalance_offset` and
  `preferred_description_condition`.
- The exact selected order and condition assignment is written to
  `main_sequence.csv`.

Manipulation/text check trials are selected only from non-congruent trials. The
target count is approximately `CHECK_RATE` times the main-trial count.

## Output

Each run is written to:

```text
output/participant_<participant>/<YYYY-MM-DD_HH-MM-SS-ms>/
```

That run folder contains:
- `main_trials.csv`
- `main_gaze.csv`
- `practice_trials.csv`
- `practice_gaze.csv`
- `main_sequence.csv`
- `run_metadata.json`
- `qc.png`

The top-level summary file is:
- `output/participants_latest.csv`

It contains one row per participant, always keeping only the latest run for that participant. If the same participant runs again, their previous row is replaced.

`main_trials.csv` and `practice_trials.csv` include:

- trial identity, stimulus identity, image file, and description condition
- selected task language
- description, fixation, image-viewing, rating, and optional text-check timing
- screen size and displayed-image size/position metadata
- rating value and reaction time

`main_gaze.csv` and `practice_gaze.csv` are written when cursor-proxy sampling
is enabled (`real` and `test`). Samples are resampled to `60Hz` by default and
include:

- screen-centered gaze/cursor coordinates in pixels: `gaze_x_pix`, `gaze_y_pix`
- per-sample movement since the previous sample in the same trial:
  `gaze_movement_px`
- cumulative within-trial movement: `gaze_movement_cumulative_px`
- screen size and displayed-image geometry metadata for image-space mapping

The first gaze sample in a trial has `gaze_movement_px = 0.0`.

`run_metadata.json` includes the run mode, selected language, participant,
session, trial counts, random seed, counterbalance offset, selected
manipulation-check trials, and fixation duration.

## End-Of-Run QC

At the end of a completed run, the researcher sees a compact QC screen:

- QC plot
- main trial count
- gaze sample count
- mean image-viewing time
- mean rating
- condition counts as `C/A/I`
- run folder
- warnings/failures only

Long file lists, metadata paths, random seed details, and full checklist text
are intentionally not shown on screen. They remain saved in the run folder and
CSV outputs.

## Gaze Visualization

Use `scripts/visualize_gaze.py` to generate per-trial gaze overlays and a run
summary from a saved run folder.

```bash
python scripts/visualize_gaze.py --run-dir output/participant_001/<run-folder> --phase main
python scripts/visualize_gaze.py --run-dir output/participant_001/<run-folder> --phase practice
```

Outputs are written to `<run-folder>/gaze_visualization/` by default:

- per-trial overlay PNGs
- `<phase>_gaze_summary.csv`
- `<phase>_run_summary.png`

The visualizer maps screen-centered gaze coordinates back onto the displayed
image using the image geometry saved in the gaze CSV.

## Candidate Image Downloader

`scripts/download_wikiart_candidates.py` is a helper for sourcing candidate
WikiArt images from the `asahi417/wikiart-all` mirror. It scores and downloads
candidates into `input/candidate_images/`, writes a `manifest.csv`, and creates
a contact sheet for manual selection. It is not required for running the study.

## AOI Setup

AOIs are initialized as editable, rule-based predefined regions with a margin.
This is the preferred workflow for the current artwork stimuli: the tool
proposes a small number of pre-gaze operational regions around separated
salient image areas, but it keeps all pixels outside those regions as
`non-AOI / background` instead of forcing the whole image into semantic
categories.

This means the defaults are not full-screen Voronoi regions and not pure
freehand semantic boundaries. They are operational analysis regions that can be
edited before analysis. For ambiguous or abstract artworks, treat AOIs as
transparent analysis regions rather than objective object boundaries.

```bash
source .venv/bin/activate
python scripts/aoi_tool.py init
```

`--k` is optional. It only controls the automatic starting layout; individual
images can have different AOI counts after editing. Without `--k`, the tool
starts with up to 4 total categories per image: one non-AOI/background category
and up to three foreground operational regions. Some images receive fewer
foreground regions when the algorithm does not find enough separated candidates.

This writes:

- `input/main/aois/aoi_shapes.json`: normalized AOI shapes per image
- `input/main/aois/aois.csv`: one row per image AOI, including shape type,
  method, source note, area, bounding box, and polygon points when applicable
- `input/main/aois/previews/`: overlay images for visual checking
- `input/main/aois/label_maps/`: grayscale AOI label maps, where pixel values
  `1..k` identify AOI membership. `1` is the background/rest AOI.

Operational-region export rule:

- Pixels outside all foreground AOIs are assigned to `non-AOI / background`
- If automatically proposed regions overlap, the pixel is assigned to the
  nearest AOI center
- Manually drawn polygon/rectangle AOIs still use the visible layer order,
  because those are explicitly edited regions

To manually adjust AOI shapes:

```bash
source .venv/bin/activate
python scripts/aoi_tool.py edit
```

Editor controls:

- Click/drag shape: move it
- Click/drag corner handle: resize it
- `E`: draw a new ellipse
- `R`: draw a new rectangle
- `L`: draw a freehand lasso/polygon AOI. Hold the mouse button, trace the
  object boundary, then release.
- `]`: bring the selected AOI one layer forward
- `[`: send the selected AOI one layer backward
- `T`: move the selected AOI to the top layer
- `B`: move the selected AOI to the bottom foreground layer
- `Delete` / `Backspace`: delete the selected foreground AOI
- `Tab`: select the next foreground AOI
- `2` through `9`: reset the current image to that many total AOIs,
  including background/rest
- `A`: reset the current image to the default image-oriented AOIs
- `+` / `-`: grow or shrink the selected AOI
- Arrow keys: nudge the selected AOI
- `S`: save current image
- `N` / `P`: save and move to next / previous image
- `Q`: save and quit

After editing, regenerate the CSV, previews, and label maps without changing
saved shapes:

```bash
python scripts/aoi_tool.py batch
```

AOI records are preserved when the image bank changes. If you delete images
from `input/main/images/`, their saved AOIs remain in `aoi_shapes.json`. If you
add new images, `init`, `edit`, and `batch` initialize only missing records.
If an image file is replaced while keeping the same filename, the older AOI
record is moved into `archived_images` before a fresh AOI is created for the new
file.
