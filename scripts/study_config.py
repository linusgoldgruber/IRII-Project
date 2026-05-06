"""Central configuration for the eye-tracking study task.

Edit values here to quickly tune experiment behavior without touching task logic.
"""
github_key = "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"

# Trial counts
N_TRIALS_REAL = 60
N_TRIALS_RG = 60
N_TRIALS_TEST = 3
N_PRACTICE_TRIALS = 2
MAX_MAIN_STIMULI = 80

# Trial flow and timing
CHECK_RATE = 0.15
MIN_IMAGE_VIEW_S = 10.0
BREAK_AFTER_TRIALS = (20, 40)
BREAK_MIN_S = 20.0

# Alpha/prototype behavior
ALPHA_SHOW_IMAGE_PROMPTS = True
ALPHA_SHOW_STIMULUS_DEBUG = True

# Hidden key windows (seconds)
QUIT_DOUBLE_PRESS_WINDOW_S = 1.0
FAST_SKIP_DOUBLE_PRESS_WINDOW_S = 1.0

# Sampling
GAZE_SAMPLE_INTERVAL_S = 1.0 / 60.0
CURSOR_TARGET_HZ = 60.0

# Rating slider behavior
RATING_MIN = 1.0
RATING_MAX = 5.0
RATING_DEFAULT = 3.0
RATING_SPEED_BASE = 1.0
RATING_SPEED_ACCEL = 4.8
RATING_SPEED_MAX = 5.0

# Window/display
WINDOW_SIZE = (1280, 720)
FULLSCREEN = True
SCREEN_INDEX = 1
WINDOW_COLOR = "black"
WINDOW_UNITS = "height"
IMAGE_MAX_WIDTH_HEIGHT_UNITS = 1.55
IMAGE_MAX_HEIGHT_HEIGHT_UNITS = 0.82

# End-of-run QC display
QC_SUMMARY_LINES = 5
