BUTTONS = [
    ("attack_btn_home", "Attack button  (village, bottom right)"),
    ("find_btn", "Find Match  (green button)"),
    ("nxt_btn", "Next base  (on the search screen)"),
    ("attack_btn", "Attack  (confirm the battle)"),
    ("s_goblin", "Goblin troop  (bottom troop bar)"),
    ("surr", "Surrender button  (in battle, bottom-left)"),
    ("surr_conf", "Surrender confirm  (red, on the quit dialog)"),
    ("return_home", "Return Home  (post-battle button)"),
    ("retry_btn_1", "Retry spot 1  (tap 1 of 3 when a scan fails)"),
    ("retry_btn_2", "Retry spot 2  (tap 2 of 3 when a scan fails)"),
    ("retry_btn_3", "Retry spot 3  (tap 3 of 3 when a scan fails)"),
    ("troop_slot_1", "Troop slot 1  (first troop icon in the bottom army bar)"),
    ("troop_slot_2", "Troop slot 2  (second troop icon)"),
    ("troop_slot_3", "Troop slot 3  (third troop icon)"),
    ("troop_slot_4", "Troop slot 4  (fourth troop icon)"),
    ("troop_slot_5", "Troop slot 5  (fifth troop icon)"),
]

REQUIRED_BUTTONS = 11

BUTTON_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#00e5ff",
    "#f032e6",
    "#ffff33",
    "#46f0f0",
    "#d4a017",
    "#7fbf7f",
    "#c77dff",
    "#80ed99",
    "#ff9f1c",
    "#8ecae6",
    "#ffb3c1",
]

import os
import subprocess
import sys
import time
import tkinter as tk
from tkinter import messagebox

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
_VENV_PY = os.path.join(_ROOT_DIR, ".venv", "Scripts", "python.exe")
if (
    not getattr(sys, "frozen", False)
    and sys.argv
    and os.path.isfile(sys.argv[0])
    and os.path.isfile(_VENV_PY)
    and os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(_VENV_PY)
):
    raise SystemExit(subprocess.call([_VENV_PY, *sys.argv]))

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.config import AppConfig
from device.adb import EmulatorAdb
from core.settings import load_settings, save_settings
from core.utils import user_path


class Calibrator:
    def __init__(self, root, adb, settings):
        self.root = root
        self.adb = adb
        self.settings = settings
        self.controls = dict(settings.get("controls") or {})

        self.index = self._first_uncalibrated()
        self.fix_mode = False
        self.row_widgets = []
        self._photo = None
        self._img = None
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._undo_stack = []
        self._render_job = None

        root.title("COC Auto-Farm - Button Calibration")
        root.minsize(720, 560)
        self._size_to_screen()

        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        self.instruction = tk.StringVar()
        tk.Label(
            root,
            textvariable=self.instruction,
            font=("Segoe UI", 12, "bold"),
            fg="#f8f9fa",
            bg="#202124",
            pady=8,
        ).grid(row=0, column=0, sticky="we")

        sub = tk.Label(
            root,
            text="Klik 1x titik tombol ATAU Klik & Geser (drag kotak) untuk crop area tombol secara utuh.",
            font=("Segoe UI", 9),
            fg="#bdc1c6",
            bg="#202124",
        )
        sub.grid(row=0, column=0, sticky="we")

        self.canvas = tk.Canvas(
            root, bg="#15161a", highlightthickness=0,
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 8))
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Configure>", self._on_resize)
        self._drag_start = None
        self._drag_rect_id = None

        list_wrap = tk.Frame(root, bg="#202124")
        list_wrap.grid(row=2, column=0, sticky="we")
        self.status_labels = {}
        for i, (name, desc) in enumerate(BUTTONS):
            row = tk.Frame(list_wrap, bg="#202124")
            row.pack(fill="x", padx=12, pady=1)
            dot = tk.Label(
                row, text="\u25cf", fg=BUTTON_COLORS[i], bg="#202124", font=(None, 11)
            )
            dot.pack(side="left")
            label = tk.Label(
                row, text=desc, fg="#e8eaed", bg="#202124", anchor="w",
                font=("Segoe UI", 10),
            )
            label.pack(side="left")
            value = tk.Label(
                row, text=self._fmt(self.controls.get(name)), fg="#9aa0a6",
                bg="#202124", width=26, anchor="e", font=("Consolas", 9),
            )
            value.pack(side="right")
            self.status_labels[name] = value
            self.row_widgets.append(row)
            self._bind_row_click(row, name)
        self._paint_rows()

        btn_bar = tk.Frame(root, bg="#202124")
        btn_bar.grid(row=3, column=0, sticky="we", pady=(4, 10))
        self.status_var = tk.StringVar()
        tk.Label(
            btn_bar, textvariable=self.status_var, fg="#9aa0a6", bg="#202124",
            font=("Segoe UI", 9),
        ).pack(side="right", padx=10)
        for text, cmd, color in [
            ("Undo", self.undo, "#5f6368"),
            ("Clear selected", self.clear_selected, "#a86418"),
            ("Clear all", self.clear_all, "#8b0000"),
            ("Recapture", self.recapture, "#1a73e8"),
            ("Save to settings", self.save, "#188038"),
        ]:
            tk.Button(
                btn_bar, text=text, command=cmd, bg=color, fg="white",
                relief="flat", padx=16, pady=4, font=("Segoe UI", 10, "bold"),
            ).pack(side="left", padx=6)

        self._capture()
        root.after(50, self._render)
        self._update_instruction()

    def _size_to_screen(self):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = max(720, int(sw * 0.92))
        h = max(560, int(sh * 0.92))
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _fmt(self, norm):
        if not norm or "x" not in norm:
            return "not set"
        return f"({norm['x']:.3f}, {norm['y']:.3f})"

    def _first_uncalibrated(self):
        for i, (name, _) in enumerate(BUTTONS):
            if name not in self.controls:
                return i
        return 0

    def _paint_rows(self):
        base = "#202124"
        highlight = "#3d4957"
        for row in self.row_widgets:
            row.configure(bg=base)
            for child in row.winfo_children():
                child.configure(bg=base)
        if self.index < len(self.row_widgets):
            row = self.row_widgets[self.index]
            row.configure(bg=highlight)
            for child in row.winfo_children():
                child.configure(bg=highlight)

    def _bind_row_click(self, widget, name):
        widget.bind("<Button-1>", lambda _e, n=name: self._select(n))
        for child in widget.winfo_children():
            self._bind_row_click(child, name)

    def _capture(self):
        frame = self.adb.screencap()
        if frame is None:
            messagebox.showerror(
                "Capture failed",
                f"Could not capture the emulator screen.\n{self.adb.last_error}",
            )
            sys.exit(1)
        self._img = frame

    def recapture(self):
        frame = self.adb.screencap()
        if frame is None:
            messagebox.showerror(
                "Recapture failed",
                f"Could not capture the emulator screen.\n{self.adb.last_error}",
            )
            return
        self._img = frame
        self._render()
        self.status_var.set("Recaptured " + time.strftime("%H:%M:%S"))

    def _on_resize(self, _event):
        if self._render_job is not None:
            try:
                self.root.after_cancel(self._render_job)
            except ValueError:
                pass
        self._render_job = self.root.after(60, self._render)

    def _render(self):
        self._render_job = None
        if self._img is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 4 or ch < 4:
            return

        im_h, im_w = self._img.shape[:2]
        scale = min(cw / im_w, ch / im_h)
        dw, dh = int(im_w * scale), int(im_h * scale)
        self._scale = scale
        self._offset_x = (cw - dw) // 2
        self._offset_y = (ch - dh) // 2

        rgb = cv2.cvtColor(self._img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((dw, dh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)

        for i, (name, _) in enumerate(BUTTONS):
            norm = self.controls.get(name)
            if norm and "x" in norm:
                x = self._offset_x + norm["x"] * im_w * scale
                y = self._offset_y + norm["y"] * im_h * scale
                is_current = i == self.index
                r = 14 if is_current else 12
                self.canvas.create_oval(
                    x - r, y - r, x + r, y + r,
                    outline=BUTTON_COLORS[i], width=(6 if is_current else 3),
                )

    def _update_instruction(self):
        if self.index >= len(BUTTONS):
            self.instruction.set("All buttons set. Press Save, then run the farm.")
            return
        if self.fix_mode:
            name, _ = BUTTONS[self.index]
            self.instruction.set(
                f"Editing {name} — click its new spot on the screenshot, "
                "then press Save. Other buttons keep their saved positions."
            )
            return
        _, desc = BUTTONS[self.index]
        self.instruction.set(
            f"({self.index + 1}/{len(BUTTONS)}) Click the button:  {desc}"
        )

    def _to_orig(self, cx, cy):
        ox = (cx - self._offset_x) / self._scale
        oy = (cy - self._offset_y) / self._scale
        im_w, im_h = self._img.shape[1], self._img.shape[0]
        ox = max(0, min(im_w - 1, int(ox)))
        oy = max(0, min(im_h - 1, int(oy)))
        return ox, oy

    def _select(self, name):
        self.index = next(i for i, (n, _) in enumerate(BUTTONS) if n == name)
        self.fix_mode = True
        self._paint_rows()
        self._render()
        self._update_instruction()

    def on_canvas_click(self, event):
        if self.index >= len(BUTTONS) or self._img is None:
            return
        self._drag_start = (event.x, event.y)
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

    def on_canvas_drag(self, event):
        if not self._drag_start or self._img is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        if self._drag_rect_id is None:
            self._drag_rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="#00ffcc", width=2, dash=(4, 4)
            )
        else:
            self.canvas.coords(self._drag_rect_id, x0, y0, x1, y1)

    def on_canvas_release(self, event):
        if not self._drag_start or self.index >= len(BUTTONS) or self._img is None:
            self._drag_start = None
            return

        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self._drag_start = None

        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

        name, _ = BUTTONS[self.index]
        im_w, im_h = self._img.shape[1], self._img.shape[0]

        # Check if user dragged a box (> 12px) or just single-clicked
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        tpl_dir = user_path("templates")
        os.makedirs(tpl_dir, exist_ok=True)

        if dist >= 12:
            # User dragged a manual crop bounding box!
            ox0, oy0 = self._to_orig(min(x0, x1), min(y0, y1))
            ox1, oy1 = self._to_orig(max(x0, x1), max(y0, y1))
            cx = (ox0 + ox1) // 2
            cy = (oy0 + oy1) // 2
            crop = self._img[oy0:oy1, ox0:ox1]
            if crop.size > 0:
                tpl_path = os.path.join(tpl_dir, f"{name}.png")
                cv2.imwrite(tpl_path, crop)
                self.status_var.set(f"Saved custom crop for {name} ({crop.shape[1]}x{crop.shape[0]})")
        else:
            # Single click: calculate center and use generous bounding box
            cx, cy = self._to_orig(x0, y0)
            crop_rx = max(38, int(im_w * 0.055))
            crop_ry = max(30, int(im_h * 0.055))
            y_start = max(0, cy - crop_ry)
            y_end = min(im_h, cy + crop_ry)
            x_start = max(0, cx - crop_rx)
            x_end = min(im_w, cx + crop_rx)
            crop = self._img[y_start:y_end, x_start:x_end]
            if crop.size > 0:
                tpl_path = os.path.join(tpl_dir, f"{name}.png")
                cv2.imwrite(tpl_path, crop)

        norm = {"x": round(cx / im_w, 5), "y": round(cy / im_h, 5)}
        self._undo_stack.append((name, self.controls.get(name)))
        self.controls[name] = norm
        self.status_labels[name].config(text=self._fmt(norm))

        if not self.fix_mode:
            self.index = min(len(BUTTONS), self.index + 1)
        self._paint_rows()
        self._render()
        self._update_instruction()

    def undo(self):
        if not self._undo_stack:
            return
        name, old = self._undo_stack.pop()
        if old is None:
            self.controls.pop(name, None)
        else:
            self.controls[name] = old
        self.status_labels[name].config(text=self._fmt(self.controls.get(name)))
        self.index = next(i for i, (n, _) in enumerate(BUTTONS) if n == name)
        self.fix_mode = True
        self._paint_rows()
        self._render()
        self._update_instruction()

    def clear_selected(self):
        if self.index >= len(BUTTONS):
            return
        name, _ = BUTTONS[self.index]
        self.controls.pop(name, None)
        self.status_labels[name].config(text=self._fmt(None))
        self.fix_mode = True
        self._paint_rows()
        self._render()
        self._update_instruction()

    def clear_all(self):
        self.controls = {}
        for name, _ in BUTTONS:
            self.status_labels[name].config(text="not set")
        self.index = 0
        self.fix_mode = False
        self._paint_rows()
        self._render()
        self._update_instruction()

    def save(self):
        missing = [
            name for name, _ in BUTTONS[:REQUIRED_BUTTONS]
            if name not in self.controls
        ]
        if missing:
            messagebox.showwarning(
                "Missing buttons",
                "Still missing:\n" + "\n".join(f" - {name}" for name in missing)
                + "\n\nSet every button before saving.",
            )
            return
        self.settings["controls"] = self.controls
        ok = save_settings(self.settings)
        if ok:
            messagebox.showinfo(
                "Saved",
                "Calibration saved to settings.json.\n\nThe farm bot can now start.",
            )
        else:
            messagebox.showerror("Save failed", "Could not write settings.json.")


def main():
    config = AppConfig()
    adb = EmulatorAdb(host=config.adb_host, port=config.adb_port, adb_path=config.adb_path)
    if not adb.connect():
        tk.Tk().withdraw()
        messagebox.showerror(
            "ADB Error",
            "Could not connect to the emulator via ADB.\n\n"
            f"Details: {adb.last_error}",
        )
        sys.exit(1)

    settings = load_settings()
    root = tk.Tk()
    Calibrator(root, adb, settings)
    root.mainloop()


if __name__ == "__main__":
    main()