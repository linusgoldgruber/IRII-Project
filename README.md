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

You can also set `main_trials` in the same dialog.
- leave it blank to use the mode default
- or enter any positive integer up to the 80-image bank for a custom run length

## Input

All study inputs live under `input/`:

- `input/main/stimuli.csv`
- `input/main/images/`
- `input/practice/stimuli.csv`
- `input/practice/images/`

The main stimulus CSV stores the prepared image-description mappings.
The practice CSV stays separate but uses the same simple flat layout.
The prepared main bank is capped at 80 images.
The loader resolves common raster formats through Pillow, so `jpg`, `png`,
`ppm`, `bmp`, `tif`, and similar files are all displayable as long as they
exist in the input folder.

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
- `qc.png`

The top-level summary file is:
- `output/participants_latest.csv`

It contains one row per participant, always keeping only the latest run for that participant. If the same participant runs again, their previous row is replaced.

## Behavior

- Practice block: 2 trials
- Main block: mode-dependent trial count
- Main trials are randomized from the prepared stimulus bank
- Descriptions are balanced across `congruent`, `ambiguous`, and `incongruent`
- The exact trial order is saved in `main_sequence.csv`
- QC is generated automatically at the end of each run

## Notes

- Hidden quit: press `Q` twice within the quit window
- In `test` mode, the hidden fast-skip shortcuts are enabled
- Cursor-proxy sampling is enabled for `real` and `test`, and disabled for `rg`

## AOI Setup

Coarse AOIs for the main image bank can be generated with editable shape masks.
This is better than fixed grid or Voronoi AOIs for this stimulus set because a
single `background / rest` AOI can cover all pixels that are not part of a
foreground shape. For example, a circular central object can be marked with an
ellipse while the surrounding background remains one AOI instead of being split
into top-left/top-right regions.

```bash
source .venv/bin/activate
python scripts/aoi_tool.py init
```

`--k` is optional. It only controls the automatic starting layout; individual
images can have different AOI counts after editing.

This writes:

- `input/main/aois/aoi_shapes.json`: normalized AOI shapes per image
- `input/main/aois/aois.csv`: one row per image AOI, including shape type,
  area, bounding box, and polygon points when applicable
- `input/main/aois/previews/`: overlay images for visual checking
- `input/main/aois/label_maps/`: grayscale AOI label maps, where pixel values
  `1..k` identify AOI membership. `1` is the background/rest AOI.

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
