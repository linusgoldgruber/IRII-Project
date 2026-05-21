from __future__ import annotations

import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, DISABLED, END, LEFT, NORMAL, RIGHT, Button, Frame, Label, StringVar, Text, Tk, messagebox
from tkinter import ttk

from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STIMULI_CSV = PROJECT_ROOT / "input" / "main" / "stimuli.csv"
IMAGE_DIR = PROJECT_ROOT / "input" / "main" / "images"
BACKUP_DIR = PROJECT_ROOT / "input" / "main" / "stimuli_backups"

TEXT_COLUMNS = [
    ("condition_congruent", "DE congruent"),
    ("condition_semi_congruent", "DE semi-congruent"),
    ("condition_incongruent", "DE incongruent"),
    ("condition_congruent_en", "EN congruent"),
    ("condition_semi_congruent_en", "EN semi-congruent"),
    ("condition_incongruent_en", "EN incongruent"),
]

RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def load_rows() -> tuple[list[dict[str, str]], list[str]]:
    if not STIMULI_CSV.exists():
        raise FileNotFoundError(f"Missing stimulus file: {STIMULI_CSV}")
    with STIMULI_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        missing = [column for column, _label in TEXT_COLUMNS if column not in fieldnames]
        missing.extend(column for column in ("stimulus_id", "image_file") if column not in fieldnames)
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")
        return list(reader), fieldnames


def write_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with STIMULI_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def backup_stimuli_csv(reason: str = "prompt_editor_launch") -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_path = BACKUP_DIR / f"stimuli_{reason}_{timestamp}.csv"
    shutil.copy2(STIMULI_CSV, backup_path)
    return backup_path


class PromptEditor:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Stimulus Prompt Editor")
        self.rows, self.fieldnames = load_rows()
        self.index = 0
        self.dirty = False
        self.photo: ImageTk.PhotoImage | None = None
        self.loading_row = False
        self.updating_slider = False
        self.status_var = StringVar()
        self.image_meta_var = StringVar()
        self.text_widgets: dict[str, Text] = {}

        self.backup_path = backup_stimuli_csv()
        self.build_ui()
        self.load_row(0)

    def build_ui(self) -> None:
        outer = Frame(self.root, padx=12, pady=10)
        outer.pack(fill=BOTH, expand=True)

        top = Frame(outer)
        top.pack(fill=BOTH, expand=True)

        image_panel = Frame(top)
        image_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 12))

        self.image_label = Label(image_panel, bg="#222222", width=760, height=620)
        self.image_label.pack(fill=BOTH, expand=True)

        self.image_meta_label = Label(image_panel, textvariable=self.image_meta_var, anchor="w", justify=LEFT)
        self.image_meta_label.pack(fill="x", pady=(8, 0))

        prompt_panel = Frame(top)
        prompt_panel.pack(side=RIGHT, fill=BOTH, expand=True)

        for column, label in TEXT_COLUMNS:
            Label(prompt_panel, text=label, anchor="w").pack(fill="x")
            text = Text(prompt_panel, height=4, wrap="word", undo=True)
            text.pack(fill=BOTH, expand=True, pady=(0, 8))
            text.bind("<<Modified>>", self.on_text_modified)
            self.text_widgets[column] = text

        controls = Frame(outer)
        controls.pack(fill="x", pady=(8, 0))

        self.prev_button = Button(controls, text="Previous", command=self.previous_row, takefocus=False)
        self.prev_button.pack(side=LEFT)

        self.next_button = Button(controls, text="Next", command=self.next_row, takefocus=False)
        self.next_button.pack(side=LEFT, padx=(8, 0))

        self.save_button = Button(controls, text="Save CSV", command=self.save_current_to_csv, takefocus=False)
        self.save_button.pack(side=LEFT, padx=(18, 0))

        Button(controls, text="Reload Current", command=self.reload_current, takefocus=False).pack(side=LEFT, padx=(8, 0))
        Button(controls, text="Quit", command=self.on_close, takefocus=False).pack(side=RIGHT)

        self.progress = ttk.Scale(
            controls,
            from_=0,
            to=max(len(self.rows) - 1, 0),
            orient="horizontal",
            command=self.on_slider_move,
        )
        self.progress.pack(side=LEFT, fill="x", expand=True, padx=18)

        Label(outer, textvariable=self.status_var, anchor="w").pack(fill="x", pady=(8, 0))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_text_modified(self, _event: object) -> None:
        if self.loading_row:
            for text in self.text_widgets.values():
                text.edit_modified(False)
            return
        for text in self.text_widgets.values():
            if text.edit_modified():
                text.edit_modified(False)
                self.dirty = True
        self.update_status()

    def update_status(self) -> None:
        row = self.rows[self.index]
        marker = "unsaved edits" if self.dirty else "saved"
        self.status_var.set(
            f"{self.index + 1}/{len(self.rows)} | stimulus {row.get('stimulus_id', '')} | "
            f"{row.get('image_file', '')} | {marker} | backup: {self.backup_path.relative_to(PROJECT_ROOT)}"
        )
        self.prev_button.config(state=NORMAL if self.index > 0 else DISABLED)
        self.next_button.config(state=NORMAL if self.index < len(self.rows) - 1 else DISABLED)

    def load_row(self, index: int) -> None:
        if not self.confirm_discard_or_save():
            return
        self.index = max(0, min(index, len(self.rows) - 1))
        row = self.rows[self.index]
        self.loading_row = True
        try:
            for column, _label in TEXT_COLUMNS:
                text = self.text_widgets[column]
                text.delete("1.0", END)
                text.insert("1.0", row.get(column, ""))
                text.edit_modified(False)
        finally:
            self.loading_row = False
        self.dirty = False
        self.load_image(row)
        self.updating_slider = True
        try:
            self.progress.set(self.index)
        finally:
            self.updating_slider = False
        self.root.after_idle(self.clear_modified_flags_after_load)
        self.update_status()

    def clear_modified_flags_after_load(self) -> None:
        if self.dirty:
            return
        for text in self.text_widgets.values():
            text.edit_modified(False)
        self.update_status()

    def load_image(self, row: dict[str, str]) -> None:
        image_file = row.get("image_file", "")
        image_path = IMAGE_DIR / image_file
        if not image_path.exists():
            self.image_label.config(image="", text=f"Missing image:\n{image_path}", fg="white")
            self.image_meta_var.set(f"Missing image: {image_path.relative_to(PROJECT_ROOT)}")
            self.photo = None
            return

        with Image.open(image_path) as image:
            original_size = image.size
            display = image.copy()
        display.thumbnail((760, 620), RESAMPLE_LANCZOS)
        self.photo = ImageTk.PhotoImage(display)
        self.image_label.config(image=self.photo, text="")
        self.image_meta_var.set(
            f"{image_file} | {original_size[0]}x{original_size[1]} px | "
            f"{row.get('original_filename', '')} | {row.get('original_style_folder', '')}"
        )

    def collect_text_edits(self) -> None:
        row = self.rows[self.index]
        for column, _label in TEXT_COLUMNS:
            row[column] = self.text_widgets[column].get("1.0", "end-1c").strip()

    def save_current_to_csv(self) -> None:
        self.collect_text_edits()
        write_rows(self.rows, self.fieldnames)
        self.dirty = False
        for text in self.text_widgets.values():
            text.edit_modified(False)
        self.update_status()

    def confirm_discard_or_save(self) -> bool:
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel(
            "Unsaved edits",
            "Save the current prompt edits before changing image?",
        )
        if answer is None:
            return False
        if answer:
            self.save_current_to_csv()
        else:
            self.dirty = False
        return True

    def previous_row(self) -> None:
        self.load_row(self.index - 1)

    def next_row(self) -> None:
        self.load_row(self.index + 1)

    def reload_current(self) -> None:
        if not self.confirm_discard_or_save():
            return
        self.rows, self.fieldnames = load_rows()
        self.load_row(self.index)

    def on_slider_move(self, value: str) -> None:
        if self.updating_slider:
            return
        target = int(float(value))
        if target != self.index:
            self.load_row(target)

    def on_close(self) -> None:
        if not self.confirm_discard_or_save():
            return
        self.root.destroy()


def main() -> int:
    try:
        root = Tk()
        PromptEditor(root)
        root.mainloop()
        return 0
    except Exception as exc:
        print(f"Prompt editor failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
