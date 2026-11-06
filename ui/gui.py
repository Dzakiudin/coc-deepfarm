import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from ai.brain import train_commander_from_data
from ai.policy_net import train_policy_from_dataset
from ai.video_learner import VideoAttackExtractor
from core.settings import ensure_default_controls, save_settings
from core.utils import user_path


class BotGUI:

    BG = "#080d1a"
    PANEL = "#0f172a"
    PANEL2 = "#162038"
    PANEL_ALT = "#131c31"
    BORDER = "#233252"
    BORDER_GLOW = "#365082"
    TEXT = "#f1f5f9"
    MUTED = "#8b9bb4"
    CYAN = "#38bdf8"
    BLUE = "#3b82f6"
    INDIGO = "#5865f2"
    GREEN = "#22c55e"
    GREEN_BG = "#123528"
    RED = "#ef4444"
    RED_BG = "#3b1822"
    GOLD = "#fbbf24"
    ELIXIR = "#e879f9"
    PURPLE = "#a855f7"

    def __init__(self, root: tk.Tk, bot, msg_queue: queue.Queue, settings=None):
        self.root = root
        self.bot = bot
        self.msg_queue = msg_queue
        self.settings = dict(settings or {})
        self.root.title("Coc-DeepFarm | Dual-Brain CoC AI Dashboard | By jakijekiiii")
        self.root.geometry("1480x880")
        self.root.minsize(1120, 740)
        self.root.configure(bg=self.BG)

        self.bot_status = "stopped"
        self.game_screen = "unknown"
        self.stage = "Waiting for bot events"
        self.started_at = None
        self.last_elapsed = 0.0
        self.latest_frame = None
        self.latest_detections = {}
        self.latest_loot = {}
        self.latest_ai_overlay = {}
        self.show_ai_radar = True
        self.chart_mode = "ai"  # 'ai' (RL Learning Curve) or 'perf' (System Latency ms)
        self.rl_history = []
        self.latest_metrics = {"capture_ms": 0.0, "yolo_ms": 0.0, "ocr_ms": 0.0}
        self.performance_history = []
        self.raids = []
        self.session_totals = {"gold": 0, "elixir": 0, "delixir": 0}
        self.session_raid_count = 0
        self._photo = None
        self._banner_photo = None
        self._icons = {}
        self._last_metric_sample = 0.0
        self._last_telemetry_at = None
        self._preview_frame_lock = threading.Lock()
        self._pending_preview_frame = None
        self._preview_generation = 0
        self._rendered_preview_generation = 0
        self._preview_needs_render = False
        self._build_icons()
        self._build_styles()
        self._build_ui()
        self._load_recent_raids()
        self._load_rl_history()
        self._refresh_all()
        self._poll_queue()
        self._tick()
        self._preview_stop = threading.Event()
        self.root.after(33, self._refresh_preview)
        self._preview_thread = threading.Thread(target=self._idle_preview_loop, daemon=True)
        self._preview_thread.start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_icons(self):
        self._icons["shield"] = self._make_icon("shield", 36)
        self._icons["gold_lg"] = self._make_icon("gold", 38)
        self._icons["elixir_lg"] = self._make_icon("elixir", 38)
        self._icons["delixir_lg"] = self._make_icon("delixir", 38)
        self._icons["raids_lg"] = self._make_icon("raids", 38)
        self._icons["progress_lg"] = self._make_icon("trophy", 38)
        self._icons["runtime_lg"] = self._make_icon("clock", 38)
        self._icons["gold_sm"] = self._make_icon("gold", 20)
        self._icons["elixir_sm"] = self._make_icon("elixir", 20)
        self._icons["delixir_sm"] = self._make_icon("delixir", 20)

    def _make_icon(self, kind: str, size: int) -> ImageTk.PhotoImage:
        s = 80
        img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if kind == "shield":
            pts = [
                (40, 6),
                (72, 18),
                (68, 52),
                (40, 74),
                (12, 52),
                (8, 18),
            ]
            draw.polygon(pts, fill="#2a1f14", outline="#f5b82e", width=4)
            inner = [
                (40, 14),
                (64, 23),
                (60, 49),
                (40, 66),
                (20, 49),
                (16, 23),
            ]
            draw.polygon(inner, fill="#f59e0b", outline="#fde68a", width=2)
        elif kind == "gold":
            draw.ellipse((4, 4, 76, 76), fill="#2a2112", outline="#785616", width=3)
            draw.ellipse((14, 14, 66, 66), fill="#f59e0b", outline="#fde047", width=3)
            draw.ellipse((24, 24, 56, 56), fill="#fbbf24", outline="#fef08a", width=2)
        elif kind == "elixir":
            draw.ellipse((4, 4, 76, 76), fill="#291436", outline="#6b21a8", width=3)
            cx, cy, r = 40, 46, 18
            draw.polygon([(cx, 16), (cx - r, cy), (cx + r, cy)], fill="#e879f9")
            draw.ellipse((cx - r, cy - r + 4, cx + r, cy + r), fill="#d946ef", outline="#f5d0fe", width=2)
            draw.ellipse((cx - 9, cy - 6, cx - 3, cy + 4), fill="#fae8ff")
        elif kind == "delixir":
            draw.ellipse((4, 4, 76, 76), fill="#1b152e", outline="#4c1d95", width=3)
            cx, cy, r = 40, 46, 18
            draw.polygon([(cx, 16), (cx - r, cy), (cx + r, cy)], fill="#7e22ce")
            draw.ellipse((cx - r, cy - r + 4, cx + r, cy + r), fill="#581c87", outline="#c084fc", width=2)
            draw.ellipse((cx - 9, cy - 6, cx - 3, cy + 4), fill="#d8b4fe")
        elif kind == "raids":
            draw.ellipse((4, 4, 76, 76), fill="#15233b", outline="#2e4a7d", width=3)
            m = 20
            draw.line((m, s - m, s - m, m), fill="#e2e8f0", width=5)
            draw.line((m, m, s - m, s - m), fill="#94a3b8", width=5)
            draw.line((15, 65, 29, 51), fill="#fbbf24", width=6)
            draw.line((65, 65, 51, 51), fill="#fbbf24", width=6)
        elif kind == "trophy":
            draw.ellipse((4, 4, 76, 76), fill="#292012", outline="#785616", width=3)
            cx = 40
            draw.rectangle((cx - 15, 20, cx + 15, 42), fill="#fbbf24", outline="#fef08a", width=2)
            draw.pieslice((cx - 15, 32, cx + 15, 52), 0, 180, fill="#f59e0b")
            draw.rectangle((cx - 4, 49, cx + 4, 59), fill="#d97706")
            draw.rectangle((cx - 13, 58, cx + 13, 64), fill="#fbbf24")
        elif kind == "clock":
            draw.ellipse((4, 4, 76, 76), fill="#12243d", outline="#2563eb", width=3)
            draw.ellipse((15, 15, 65, 65), outline="#38bdf8", width=5)
            draw.line((40, 40, 40, 25), fill="#7dd3fc", width=4)
            draw.line((40, 40, 53, 46), fill="#38bdf8", width=4)

        img = img.resize((size, size), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _build_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Dark.TEntry",
            fieldbackground="#0b1324",
            foreground=self.TEXT,
            insertcolor=self.TEXT,
            bordercolor=self.BORDER,
            lightcolor=self.BORDER,
            darkcolor=self.BORDER,
            padding=6,
        )
        style.configure(
            "Dark.Treeview",
            background=self.PANEL,
            fieldbackground=self.PANEL,
            foreground=self.TEXT,
            rowheight=26,
            bordercolor=self.BORDER,
            lightcolor=self.PANEL,
            darkcolor=self.PANEL,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Dark.Treeview.Heading",
            background=self.PANEL2,
            foreground=self.MUTED,
            relief="flat",
            bordercolor=self.BORDER,
            font=("Segoe UI", 8, "bold"),
            padding=(8, 6),
        )
        style.map(
            "Dark.Treeview",
            background=[("selected", "#1e3a6d")],
            foreground=[("selected", "#ffffff")],
        )
        style.map(
            "Dark.Treeview.Heading",
            background=[("active", "#1e2d4d")],
            foreground=[("active", self.TEXT)],
        )
        style.configure(
            "Dark.Vertical.TScrollbar",
            background=self.PANEL2,
            troughcolor=self.PANEL,
            bordercolor=self.PANEL,
            arrowcolor=self.MUTED,
            relief="flat",
        )

    def _panel(self, parent, *, padx=12, pady=10, bg=None, border=None):
        return tk.Frame(
            parent,
            bg=bg or self.PANEL,
            padx=padx,
            pady=pady,
            highlightbackground=border or self.BORDER,
            highlightcolor=border or self.BORDER,
            highlightthickness=1,
        )

    def _label(self, parent, text, *, size=10, color=None, bold=False, **kwargs):
        return tk.Label(
            parent,
            text=text,
            bg=parent.cget("bg"),
            fg=color or self.TEXT,
            font=("Segoe UI", size, "bold" if bold else "normal"),
            **kwargs,
        )

    def _build_ui(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

        header = tk.Frame(
            self.root,
            bg="#0b1222",
            padx=18,
            pady=8,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(2, weight=1)

        tk.Label(
            header,
            image=self._icons["shield"],
            bg="#0b1222",
        ).grid(row=0, column=0, rowspan=2, padx=(0, 10))

        title_row = tk.Frame(header, bg="#0b1222")
        title_row.grid(row=0, column=1, sticky="sw")
        tk.Label(
            title_row,
            text="Coc-DeepFarm",
            bg="#0b1222",
            fg=self.TEXT,
            font=("Segoe UI", 16, "bold"),
        ).pack(side="left")
        tk.Label(
            title_row,
            text="v2.0 DeepRL",
            bg="#143829",
            fg=self.GREEN,
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=1,
        ).pack(side="left", padx=(8, 0), pady=(2, 0))

        tk.Label(
            header,
            text="Dual-Brain Clash of Clans Neural Farming Bot  ·  By jakijekiiii",
            bg="#0b1222",
            fg=self.MUTED,
            font=("Segoe UI", 8),
        ).grid(row=1, column=1, sticky="nw")

        # Interactive Quick-Action Toolbar (replaces old static dummy tags)
        nav_pill = tk.Frame(
            header,
            bg="#121d33",
            padx=6,
            pady=4,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        nav_pill.grid(row=0, column=2, rowspan=2)
        self.brain_device_badge = tk.Label(
            nav_pill,
            text="🧠 CoC Neural Brain: Ready",
            bg="#1e1b4b",
            fg="#c4b5fd",
            font=("Segoe UI", 8, "bold"),
            padx=10,
            pady=4,
        )
        self.brain_device_badge.pack(side="left", padx=(0, 6))

        adb_obj = getattr(getattr(self.bot, "window", None), "adb", None)
        adb_ok = bool(getattr(adb_obj, "serial", None))
        self.adb_connect_btn = tk.Button(
            nav_pill,
            text="🔌 ADB Ready" if adb_ok else "🔌 Connect ADB",
            command=self._on_connect_adb,
            bg="#143829" if adb_ok else "#3b2314",
            activebackground="#166534",
            fg=self.GREEN if adb_ok else self.GOLD,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=10,
            pady=3,
            font=("Segoe UI", 8, "bold"),
            cursor="hand2",
        )
        self.adb_connect_btn.pack(side="left", padx=3)

        self.quick_retrain_btn = tk.Button(
            nav_pill,
            text="⚡ Retrain AI",
            command=self._on_quick_retrain_ai,
            bg="#172554",
            activebackground="#1e3a8a",
            fg="#93c5fd",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=10,
            pady=3,
            font=("Segoe UI", 8, "bold"),
            cursor="hand2",
        )
        self.quick_retrain_btn.pack(side="left", padx=3)

        tk.Button(
            nav_pill,
            text="🎯 Calibrate",
            command=self._on_open_calibrator,
            bg="#162038",
            activebackground="#1e293b",
            fg=self.TEXT,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=10,
            pady=3,
            font=("Segoe UI", 8, "bold"),
            cursor="hand2",
        ).pack(side="left", padx=3)

        tk.Button(
            nav_pill,
            text="📸 Snapshot",
            command=self._on_save_snapshot,
            bg="#162038",
            activebackground="#1e293b",
            fg=self.CYAN,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=10,
            pady=3,
            font=("Segoe UI", 8, "bold"),
            cursor="hand2",
        ).pack(side="left", padx=(3, 0))

        status_card = tk.Frame(
            header,
            bg="#101b30",
            padx=12,
            pady=5,
            highlightbackground="#26453e",
            highlightthickness=1,
        )
        status_card.grid(row=0, column=3, rowspan=2, sticky="e")
        self.connection_badge = tk.Label(
            status_card,
            text="● BOT STOPPED",
            bg="#101b30",
            fg=self.CYAN,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        )
        self.connection_badge.pack(anchor="w")
        tk.Label(
            status_card,
            text="ADB Stream  ·  CUDA Accelerated",
            bg="#101b30",
            fg=self.MUTED,
            font=("Segoe UI", 7),
            anchor="w",
        ).pack(anchor="w")

        hero_wrap = tk.Frame(
            self.root,
            bg=self.PANEL,
            highlightbackground=self.BORDER_GLOW,
            highlightthickness=1,
        )
        hero_wrap.grid(row=1, column=0, sticky="ew", padx=14, pady=(10, 10))
        self.hero_canvas = tk.Canvas(
            hero_wrap,
            bg="#0e192d",
            height=90,
            highlightthickness=0,
            bd=0,
        )
        self.hero_canvas.pack(fill="x", expand=True)
        self.hero_canvas.bind("<Configure>", self._draw_hero_banner)

        body = tk.Frame(self.root, bg=self.BG, padx=14, pady=0)
        body.grid(row=2, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=5, uniform="top")
        body.grid_columnconfigure(1, weight=3, uniform="top")
        body.grid_columnconfigure(2, weight=3, uniform="top")
        body.grid_rowconfigure(1, weight=5)
        body.grid_rowconfigure(2, weight=4)

        metrics = tk.Frame(body, bg=self.BG)
        metrics.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        for col in range(6):
            metrics.grid_columnconfigure(col, weight=1, uniform="kpi")
        self.metric_vars = {}
        self.metric_hint_vars = {}
        self._metric_card(metrics, 0, "Total Gold", "gold", self.GOLD, "0 / hr", "gold_lg")
        self._metric_card(metrics, 1, "Total Elixir", "elixir", self.ELIXIR, "0 / hr", "elixir_lg")
        self._metric_card(metrics, 2, "Total Dark Elixir", "delixir", self.PURPLE, "0 / hr", "delixir_lg")
        self._metric_card(metrics, 3, "Total Raids", "raids", self.TEXT, "Avg drain —", "raids_lg")
        self._metric_card(metrics, 4, "Session Progress", "progress", self.GOLD, "Target goals", "progress_lg")
        self._metric_card(metrics, 5, "Runtime", "runtime", self.CYAN, "Current run", "runtime_lg")

        live = self._panel(body, padx=12, pady=10)
        live.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        live.grid_columnconfigure(0, weight=1)
        live.grid_rowconfigure(1, weight=1)
        live_header = tk.Frame(live, bg=self.PANEL)
        live_header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(
            live_header,
            text="Live Screen — Emulator",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Segoe UI", 12, "bold"),
        ).pack(side="left")
        self.live_badge = tk.Label(
            live_header,
            text="● WAITING",
            bg="#263215",
            fg=self.GOLD,
            font=("Segoe UI", 8, "bold"),
            padx=9,
            pady=3,
        )
        self.live_badge.pack(side="left", padx=(10, 6))

        self.radar_toggle_btn = tk.Button(
            live_header,
            text="👁 AI Radar: ON",
            command=self._on_toggle_ai_radar,
            bg="#143829",
            activebackground="#166534",
            fg=self.GREEN,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=8,
            pady=2,
            font=("Segoe UI", 8, "bold"),
            cursor="hand2",
        )
        self.radar_toggle_btn.pack(side="left", padx=(2, 8))

        self.preview_info = tk.StringVar(value="No emulator frame received yet")
        tk.Label(
            live_header,
            textvariable=self.preview_info,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 8),
            anchor="e",
        ).pack(side="right")

        preview_frame = tk.Frame(
            live,
            bg="#050912",
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        preview_frame.grid(row=1, column=0, sticky="nsew")
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)

        self.preview_label = tk.Label(
            preview_frame,
            text="Waiting for a frame from the emulator…\n\n"
            "Start the bot or wait for its next ADB capture.",
            bg="#050912",
            fg=self.MUTED,
            justify="center",
            font=("Segoe UI", 10),
            padx=10,
            pady=10,
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        self.preview_label.bind("<Configure>", lambda _event: self._render_preview())

        state_panel = self._panel(body, padx=12, pady=10)
        state_panel.grid(row=1, column=1, sticky="nsew", padx=(0, 8), pady=(0, 8))
        state_panel.grid_columnconfigure(0, weight=1)

        state_top = tk.Frame(state_panel, bg=self.PANEL)
        state_top.grid(row=0, column=0, sticky="ew")
        self._label(state_top, "Current State", size=12, bold=True).pack(side="left")
        self.game_state_badge = tk.Label(
            state_top,
            text="UNKNOWN",
            bg="#2e3678",
            fg="#e0e7ff",
            font=("Segoe UI", 8, "bold"),
            padx=12,
            pady=4,
        )
        self.game_state_badge.pack(side="right")

        self.stepper_canvas = tk.Canvas(
            state_panel,
            bg=self.PANEL,
            height=58,
            highlightthickness=0,
            bd=0,
        )
        self.stepper_canvas.grid(row=1, column=0, sticky="ew", pady=(6, 2))
        self.stepper_canvas.bind("<Configure>", lambda _e: self._draw_state_stepper())

        self.stage_var = tk.StringVar(value="Waiting for bot telemetry")
        tk.Label(
            state_panel,
            textvariable=self.stage_var,
            bg=self.PANEL,
            fg=self.MUTED,
            wraplength=280,
            justify="left",
            font=("Segoe UI", 8),
        ).grid(row=2, column=0, sticky="ew", pady=(0, 6))

        tk.Frame(state_panel, bg=self.BORDER, height=1).grid(
            row=3, column=0, sticky="ew", pady=(0, 6)
        )

        det_hdr = tk.Frame(state_panel, bg=self.PANEL)
        det_hdr.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        self._label(det_hdr, "Detected Objects (YOLO)", size=10, bold=True).pack(side="left")
        self.det_count_badge = tk.Label(
            det_hdr,
            text="0",
            bg=self.PANEL2,
            fg=self.TEXT,
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=2,
        )
        self.det_count_badge.pack(side="right")

        det_table = tk.Frame(
            state_panel,
            bg=self.PANEL,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        det_table.grid(row=5, column=0, sticky="nsew")
        det_table.grid_columnconfigure(0, weight=1)
        det_table.grid_rowconfigure(0, weight=1)
        self.det_tree = ttk.Treeview(
            det_table,
            columns=("class", "count", "conf"),
            show="headings",
            style="Dark.Treeview",
            height=6,
        )
        for col, label, width in [("class", "Object Class", 135), ("count", "#", 40), ("conf", "Conf.", 55)]:
            self.det_tree.heading(col, text=label)
            self.det_tree.column(
                col,
                width=width,
                anchor="w" if col == "class" else "center",
                stretch=col == "class",
            )
        self.det_tree.tag_configure("odd", background="#131e35")
        self.det_tree.tag_configure("even", background=self.PANEL)
        det_scroll = ttk.Scrollbar(
            det_table,
            orient="vertical",
            command=self.det_tree.yview,
            style="Dark.Vertical.TScrollbar",
        )
        self.det_tree.configure(yscrollcommand=det_scroll.set)
        self.det_tree.grid(row=0, column=0, sticky="nsew")
        det_scroll.grid(row=0, column=1, sticky="ns")
        state_panel.grid_rowconfigure(5, weight=1)

        self.loot_var = tk.StringVar(value="Last scanned loot: —")
        tk.Label(
            state_panel,
            textvariable=self.loot_var,
            bg=self.PANEL2,
            fg=self.GOLD,
            wraplength=270,
            justify="left",
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=5,
            anchor="w",
        ).grid(row=6, column=0, sticky="ew", pady=(7, 0))

        control = self._panel(body, padx=12, pady=10)
        control.grid(row=1, column=2, sticky="nsew", pady=(0, 8))
        control.grid_columnconfigure(0, weight=1)
        control.grid_rowconfigure(1, weight=1)
        self._label(control, "Farming & Neural Brain Control", size=12, bold=True).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        control_canvas = tk.Canvas(
            control,
            bg=self.PANEL,
            highlightthickness=0,
            bd=0,
            yscrollincrement=24,
        )
        control_canvas.grid(row=1, column=0, sticky="nsew")
        control_scroll = ttk.Scrollbar(
            control,
            orient="vertical",
            command=control_canvas.yview,
            style="Dark.Vertical.TScrollbar",
        )
        control_scroll.grid(row=1, column=1, sticky="ns")
        control_canvas.configure(yscrollcommand=control_scroll.set)
        control_inner = tk.Frame(control_canvas, bg=self.PANEL)
        self._control_inner_id = control_canvas.create_window(
            (0, 0), window=control_inner, anchor="nw"
        )
        control_inner.bind(
            "<Configure>",
            lambda _event: control_canvas.configure(
                scrollregion=control_canvas.bbox("all")
            ),
        )
        control_canvas.bind(
            "<Configure>",
            lambda event: control_canvas.itemconfigure(
                self._control_inner_id, width=event.width
            ),
        )
        control_canvas.bind(
            "<Enter>",
            lambda _event: control_canvas.bind_all(
                "<MouseWheel>",
                lambda event: control_canvas.yview_scroll(
                    int(-event.delta / 120), "units"
                ),
            ),
        )
        control_canvas.bind(
            "<Leave>",
            lambda _event: control_canvas.unbind_all("<MouseWheel>"),
        )

        control_inner.grid_columnconfigure(0, weight=1)

        self.min_gold = tk.IntVar(value=700000)
        self.min_elixir = tk.IntVar(value=700000)
        self.min_dark = tk.IntVar(value=0)
        self.goal_gold = tk.IntVar(value=20000000)
        self.goal_elixir = tk.IntVar(value=20000000)
        self.goal_dark = tk.IntVar(value=0)

        minimum_box = tk.Frame(control_inner, bg=self.PANEL)
        minimum_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        minimum_box.grid_columnconfigure(1, weight=1)
        min_hdr = tk.Frame(minimum_box, bg=self.PANEL)
        min_hdr.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self._label(
            min_hdr,
            "Target per Raid (Minimum)",
            size=9,
            color=self.MUTED,
            bold=True,
        ).pack(side="left")
        # Quick Loot Presets
        preset_bar = tk.Frame(min_hdr, bg=self.PANEL)
        preset_bar.pack(side="right")
        for lbl, val in [("400k", 400_000), ("700k", 700_000), ("1M+", 1_000_000)]:
            tk.Button(
                preset_bar,
                text=lbl,
                command=lambda v=val: self._apply_loot_preset(v),
                bg="#162038",
                activebackground="#1e3a8a",
                fg=self.CYAN,
                activeforeground="#ffffff",
                relief="flat",
                bd=0,
                padx=5,
                pady=1,
                font=("Segoe UI", 7, "bold"),
                cursor="hand2",
            ).pack(side="left", padx=1)

        self._target_row(minimum_box, 1, "Gold", self.min_gold, self.GOLD, "gold_sm")
        self._target_row(minimum_box, 2, "Elixir", self.min_elixir, self.ELIXIR, "elixir_sm")
        self._target_row(minimum_box, 3, "Dark", self.min_dark, self.PURPLE, "delixir_sm")

        goals_box = tk.Frame(control_inner, bg=self.PANEL)
        goals_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        goals_box.grid_columnconfigure(1, weight=1)
        self._label(
            goals_box,
            "Target Total (Session)",
            size=9,
            color=self.MUTED,
            bold=True,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._target_row(goals_box, 1, "Gold", self.goal_gold, self.GOLD, "gold_sm")
        self._target_row(goals_box, 2, "Elixir", self.goal_elixir, self.ELIXIR, "elixir_sm")
        self._target_row(goals_box, 3, "Dark", self.goal_dark, self.PURPLE, "delixir_sm")

        strategy_box = tk.Frame(control_inner, bg=self.PANEL)
        strategy_box.grid(row=2, column=0, sticky="ew", pady=(2, 8))
        strategy_box.grid_columnconfigure(1, weight=1)
        self._label(
            strategy_box,
            "Brain Mode",
            size=9,
            color=self.MUTED,
            bold=True,
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.ai_mode_var = tk.BooleanVar(
            value=bool(getattr(self.bot, "ai_enabled", True))
        )
        self.ai_mode_text = tk.StringVar()
        self.ai_mode_btn = tk.Checkbutton(
            strategy_box,
            textvariable=self.ai_mode_text,
            variable=self.ai_mode_var,
            command=self._on_ai_mode_toggle,
            indicatoron=False,
            bg="#0b1324",
            activebackground=self.PANEL2,
            selectcolor="#1d2f52",
            fg=self.TEXT,
            activeforeground=self.TEXT,
            relief="flat",
            bd=1,
            highlightbackground=self.BORDER,
            highlightthickness=1,
            padx=8,
            pady=5,
            font=("Segoe UI", 8, "bold"),
            cursor="hand2",
        )
        self.ai_mode_btn.grid(row=0, column=1, sticky="ew")

        self._label(
            strategy_box,
            "Loot Filter",
            size=9,
            color=self.MUTED,
            bold=True,
        ).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        initial_tmode = str(
            getattr(self.bot, "attack", {}).get("target_mode", "smart")
        ).lower()
        if initial_tmode not in ("and", "or", "smart"):
            initial_tmode = "smart"
        self.target_mode_var = tk.StringVar(value=initial_tmode)
        self.target_mode_text = tk.StringVar()
        self.target_mode_btn = tk.Button(
            strategy_box,
            textvariable=self.target_mode_text,
            command=self._on_target_mode_cycle,
            bg="#0b1324",
            activebackground=self.PANEL2,
            fg=self.TEXT,
            activeforeground=self.TEXT,
            relief="flat",
            bd=1,
            highlightbackground=self.BORDER,
            highlightthickness=1,
            padx=8,
            pady=5,
            font=("Segoe UI", 8, "bold"),
            cursor="hand2",
        )
        self.target_mode_btn.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        self._update_target_mode_control()

        self._label(
            strategy_box,
            "Video Trainer",
            size=9,
            color=self.MUTED,
            bold=True,
        ).grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        self.learn_video_btn = tk.Button(
            strategy_box,
            text="🎬 Learn Attack (.mp4)",
            command=self._on_learn_from_video,
            bg="#1e1b4b",
            activebackground="#312e81",
            fg="#c4b5fd",
            activeforeground="#ffffff",
            relief="flat",
            bd=1,
            highlightbackground="#4c1d95",
            highlightthickness=1,
            padx=8,
            pady=5,
            font=("Segoe UI", 8, "bold"),
            cursor="hand2",
        )
        self.learn_video_btn.grid(row=2, column=1, sticky="ew", pady=(6, 0))

        # Live CoC Neural Brain Telemetry Card
        ai_card = tk.Frame(
            control_inner,
            bg="#0b1324",
            padx=8,
            pady=6,
            highlightbackground="#312e81",
            highlightthickness=1,
        )
        ai_card.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ai_card.grid_columnconfigure(0, weight=1)
        self.ai_telemetry_title = tk.StringVar(value="🧠 Neural Memory: Active")
        self.ai_telemetry_detail = tk.StringVar(value="Loading policy & commander stats...")
        tk.Label(
            ai_card,
            textvariable=self.ai_telemetry_title,
            bg="#0b1324",
            fg="#a5b4fc",
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        tk.Label(
            ai_card,
            textvariable=self.ai_telemetry_detail,
            bg="#0b1324",
            fg=self.MUTED,
            font=("Consolas", 7),
            justify="left",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", pady=(2, 0))

        self._update_ai_mode_control()

        controls = tk.Frame(control_inner, bg=self.PANEL)
        controls.grid(row=4, column=0, sticky="ew", pady=(2, 0))
        controls.grid_columnconfigure(0, weight=1)
        self.start_btn = tk.Button(
            controls,
            text="▶   Start Farming",
            command=self._on_start,
            bg="#1fa855",
            activebackground="#22c55e",
            fg="#ffffff",
            activeforeground="#ffffff",
            disabledforeground="#94a3b8",
            relief="flat",
            bd=0,
            padx=10,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        self.start_btn.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.stop_btn = tk.Button(
            controls,
            text="■   Stop (F12)",
            command=self._on_stop,
            bg="#5c212b",
            activebackground="#991b1b",
            fg="#fca5a5",
            activeforeground="#ffffff",
            disabledforeground="#64748b",
            relief="flat",
            bd=0,
            padx=10,
            pady=7,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            state="disabled",
        )
        self.stop_btn.grid(row=1, column=0, sticky="ew")

        self.status_detail = tk.StringVar(
            value="F12 emergency stop · Online RL active"
        )
        tk.Label(
            control_inner,
            textvariable=self.status_detail,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 7),
            wraplength=290,
            justify="left",
        ).grid(row=5, column=0, sticky="w", pady=(6, 2))

        recent = self._panel(body, padx=12, pady=10)
        recent.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        recent.grid_columnconfigure(0, weight=1)
        recent.grid_rowconfigure(1, weight=1)
        recent_hdr = tk.Frame(recent, bg=self.PANEL)
        recent_hdr.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._label(recent_hdr, "Recent Raids", size=11, bold=True).pack(side="left")
        tk.Label(
            recent_hdr,
            text="Session History",
            bg=self.PANEL,
            fg=self.BLUE,
            font=("Segoe UI", 8, "bold"),
        ).pack(side="right")

        raid_table = tk.Frame(
            recent,
            bg=self.PANEL,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        raid_table.grid(row=1, column=0, sticky="nsew")
        raid_table.grid_columnconfigure(0, weight=1)
        raid_table.grid_rowconfigure(0, weight=1)
        self.raid_tree = ttk.Treeview(
            raid_table,
            columns=("time", "status", "gold", "elixir", "drain"),
            show="headings",
            style="Dark.Treeview",
            height=5,
        )
        for col, label, width in [
            ("time", "Time", 75),
            ("status", "Result", 80),
            ("gold", "Gold", 95),
            ("elixir", "Elixir", 95),
            ("drain", "Drain", 60),
        ]:
            self.raid_tree.heading(col, text=label)
            self.raid_tree.column(
                col,
                width=width,
                anchor="w" if col == "time" else "center",
                stretch=col in ("gold", "elixir"),
            )
        raid_scroll = ttk.Scrollbar(
            raid_table,
            orient="vertical",
            command=self.raid_tree.yview,
            style="Dark.Vertical.TScrollbar",
        )
        self.raid_tree.configure(yscrollcommand=raid_scroll.set)
        self.raid_tree.grid(row=0, column=0, sticky="nsew")
        raid_scroll.grid(row=0, column=1, sticky="ns")
        self.raid_tree.tag_configure("attacked", foreground=self.GREEN)
        self.raid_tree.tag_configure("skipped", foreground=self.GOLD)
        self.raid_tree.tag_configure("error", foreground=self.RED)
        self.raid_tree.tag_configure("odd", background="#131e35")

        logs = self._panel(body, padx=12, pady=10)
        logs.grid(row=2, column=1, sticky="nsew", padx=(0, 8))
        logs.grid_columnconfigure(0, weight=1)
        logs.grid_rowconfigure(1, weight=1)
        log_header = tk.Frame(logs, bg=self.PANEL)
        log_header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._label(log_header, "Live Log & Self-Learning Feed", size=11, bold=True).pack(side="left")
        tk.Button(
            log_header,
            text="Clear",
            command=self._clear_log,
            bg=self.PANEL2,
            fg=self.TEXT,
            activebackground=self.BORDER,
            activeforeground=self.TEXT,
            relief="flat",
            highlightbackground=self.BORDER,
            highlightthickness=1,
            padx=10,
            pady=2,
            font=("Segoe UI", 8, "bold"),
            cursor="hand2",
        ).pack(side="right")

        log_frame = tk.Frame(
            logs,
            bg="#080f1e",
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_frame,
            height=7,
            wrap=tk.WORD,
            state="disabled",
            bg="#080f1e",
            fg="#dbeafe",
            insertbackground=self.TEXT,
            selectbackground="#1e3a8a",
            relief="flat",
            bd=0,
            font=("Consolas", 8),
            padx=8,
            pady=7,
        )
        log_scroll = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log_text.yview,
            style="Dark.Vertical.TScrollbar",
        )
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.tag_configure("time", foreground=self.MUTED)
        self.log_text.tag_configure(
            "info", background="#0284c7", foreground="#ffffff", font=("Consolas", 7, "bold")
        )
        self.log_text.tag_configure(
            "success", background="#15803d", foreground="#ffffff", font=("Consolas", 7, "bold")
        )
        self.log_text.tag_configure(
            "ai", background="#6d28d9", foreground="#ffffff", font=("Consolas", 7, "bold")
        )
        self.log_text.tag_configure(
            "error", background="#b91c1c", foreground="#ffffff", font=("Consolas", 7, "bold")
        )
        self.log_text.tag_configure("msg_success", foreground=self.GREEN)
        self.log_text.tag_configure("msg_ai", foreground="#c4b5fd")
        self.log_text.tag_configure("msg_error", foreground=self.RED)

        perf = self._panel(body, padx=12, pady=10)
        perf.grid(row=2, column=2, sticky="nsew")
        perf.grid_columnconfigure(0, weight=1)
        perf.grid_rowconfigure(2, weight=1)

        perf_hdr = tk.Frame(perf, bg=self.PANEL)
        perf_hdr.grid(row=0, column=0, sticky="ew")
        self.perf_title_var = tk.StringVar(value="AI Self-Learning & Telemetry")
        tk.Label(
            perf_hdr,
            textvariable=self.perf_title_var,
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")
        self.chart_toggle_btn = tk.Button(
            perf_hdr,
            text="🧠 AI Curve  ⇄  Latency",
            command=self._on_toggle_chart_mode,
            bg="#3b49df",
            activebackground="#4f46e5",
            fg="#ffffff",
            activeforeground="#ffffff",
            font=("Segoe UI", 7, "bold"),
            relief="flat",
            bd=0,
            padx=8,
            pady=2,
            cursor="hand2",
        )
        self.chart_toggle_btn.pack(side="right")

        perf_cards = tk.Frame(perf, bg=self.PANEL)
        perf_cards.grid(row=1, column=0, sticky="ew", pady=(7, 6))
        for i in range(3):
            perf_cards.grid_columnconfigure(i, weight=1, uniform="perf")
        self.perf_vars = {
            "capture_ms": tk.StringVar(value="—"),
            "yolo_ms": tk.StringVar(value="—"),
            "ocr_ms": tk.StringVar(value="—"),
        }
        for i, (key, title, color, icon_char) in enumerate(
            [
                ("capture_ms", "Capture Delay", self.CYAN, "📷"),
                ("yolo_ms", "YOLO Inference", self.ELIXIR, "◈"),
                ("ocr_ms", "OCR Processing", self.GREEN, "▤"),
            ]
        ):
            card = tk.Frame(
                perf_cards,
                bg="#121d33",
                padx=8,
                pady=6,
                highlightbackground=self.BORDER,
                highlightthickness=1,
            )
            card.grid(
                row=0,
                column=i,
                sticky="ew",
                padx=(0 if i == 0 else 3, 3 if i < 2 else 0),
            )
            top_line = tk.Frame(card, bg="#121d33")
            top_line.pack(fill="x")
            tk.Label(
                top_line,
                text=icon_char,
                bg="#121d33",
                fg=color,
                font=("Segoe UI Symbol", 9, "bold"),
            ).pack(side="left", padx=(0, 4))
            tk.Label(
                top_line,
                text=title,
                bg="#121d33",
                fg=self.MUTED,
                font=("Segoe UI", 7, "bold"),
            ).pack(side="left")
            tk.Label(
                card,
                textvariable=self.perf_vars[key],
                bg="#121d33",
                fg=self.TEXT,
                font=("Segoe UI", 11, "bold"),
            ).pack(anchor="w", pady=(2, 0))

        self.perf_canvas = tk.Canvas(
            perf,
            bg="#091120",
            highlightbackground=self.BORDER,
            highlightthickness=1,
            height=110,
        )
        self.perf_canvas.grid(row=2, column=0, sticky="nsew")
        self.perf_canvas.bind("<Configure>", lambda _event: self._draw_performance())
        self.perf_note = tk.StringVar(value="Online RL Self-Learning Curve · Green=Reinforce · Gold=Mistake Fix")
        tk.Label(
            perf,
            textvariable=self.perf_note,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 7),
            anchor="w",
        ).grid(row=3, column=0, sticky="ew", pady=(4, 0))

        footer = tk.Frame(self.root, bg=self.BG, padx=16, pady=4)
        footer.grid(row=3, column=0, sticky="ew")
        self.footer_var = tk.StringVar(value="Coc-DeepFarm · CoCCommanderNet + AttackPolicyNet · Online RL Active")
        tk.Label(
            footer,
            textvariable=self.footer_var,
            bg=self.BG,
            fg="#64748b",
            font=("Segoe UI", 8),
        ).pack(anchor="w")

    def _apply_loot_preset(self, amount: int):
        self.min_gold.set(int(amount))
        self.min_elixir.set(int(amount))
        self._log(f"[Preset] Minimum Raid Loot set to Gold {amount:,} / Elixir {amount:,}")

    def _on_toggle_ai_radar(self):
        self.show_ai_radar = not self.show_ai_radar
        if self.show_ai_radar:
            self.radar_toggle_btn.config(text="👁 AI Radar: ON", bg="#143829", fg=self.GREEN)
        else:
            self.radar_toggle_btn.config(text="👁 AI Radar: OFF", bg="#1e293b", fg=self.MUTED)
        self._preview_needs_render = True
        self._render_preview()

    def _on_toggle_chart_mode(self):
        self.chart_mode = "perf" if self.chart_mode == "ai" else "ai"
        if self.chart_mode == "ai":
            self._load_rl_history()
            self.perf_note.set("Online RL Self-Learning Curve · Green=Reinforce · Gold=Mistake Fix")
        else:
            self.perf_note.set("Measured milliseconds · Capture / YOLO / OCR latency")
        self._draw_performance()

    def _on_connect_adb(self):
        adb_obj = getattr(getattr(self.bot, "window", None), "adb", None)
        if adb_obj is None:
            return
        self.adb_connect_btn.config(state="disabled", text="⏳ Connecting...")
        self._log("[ADB] Mencari dan menghubungkan ke emulator Android...")

        def _worker():
            ok = False
            err = ""
            try:
                ok = bool(adb_obj.connect())
                if ok:
                    self.bot.window.start_video_capture()
                else:
                    err = str(getattr(adb_obj, "last_error", "") or "Emulator tidak ditemukan")
            except Exception as exc:
                err = str(exc)

            def _done():
                if ok:
                    self.adb_connect_btn.config(
                        state="normal",
                        text="🔌 ADB Ready",
                        bg="#143829",
                        fg=self.GREEN,
                    )
                    self._log(
                        f"[ADB] Berhasil terhubung ke emulator ({getattr(adb_obj, 'serial', 'device')})!"
                    )
                else:
                    self.adb_connect_btn.config(
                        state="normal",
                        text="🔌 Connect ADB",
                        bg="#3b2314",
                        fg=self.GOLD,
                    )
                    self._log(
                        f"[ADB] Belum terhubung ({err}). Pastikan emulator (LDPlayer/MuMu/BlueStacks) sudah menyala."
                    )

            self.root.after(0, _done)

        threading.Thread(target=_worker, daemon=True, name="AdbConnectThread").start()

    def _on_open_calibrator(self):
        if self.bot.is_running():
            messagebox.showwarning(
                "Calibrate Controls",
                "Stop the bot first before opening the button calibrator.",
            )
            return
        try:
            cal_script = user_path("ui/calibrate.py")
            subprocess.Popen([sys.executable, cal_script])
            self._log("[Calibrator] Launched ui/calibrate.py in a separate window.")
        except Exception as exc:
            messagebox.showerror("Calibrate Controls", f"Could not launch calibrator: {exc}")

    def _on_save_snapshot(self):
        if self.latest_frame is None:
            messagebox.showinfo("Snapshot", "No emulator frame available yet.")
            return
        try:
            log_dir = user_path("logs")
            os.makedirs(log_dir, exist_ok=True)
            out_path = os.path.join(
                log_dir, f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            )
            cv2.imwrite(out_path, self.latest_frame)
            self._log(f"[Snapshot] Saved current emulator screen to logs/{os.path.basename(out_path)}")
        except Exception as exc:
            self._log(f"[Snapshot] Failed to save screenshot: {exc}")

    def _on_quick_retrain_ai(self):
        if self.bot.is_running():
            messagebox.showwarning(
                "Retrain AI",
                "Stop the bot first before retraining the neural networks.",
            )
            return
        self.quick_retrain_btn.config(state="disabled", text="⏳ Training...")
        self._log("[Deep Learning] Retraining CoCCommanderNet + AttackPolicyNet from dataset & raid history...")

        def _worker():
            try:
                p_metrics = train_policy_from_dataset(epochs=80)
                c_metrics = train_commander_from_data(epochs=90)
                if getattr(self.bot, "policy_planner", None) is not None:
                    self.bot.policy_planner.reload()
                if getattr(self.bot, "brain_alt", None) is not None and hasattr(self.bot.brain_alt, "reload"):
                    self.bot.brain_alt.reload()
                self.msg_queue.put(
                    {
                        "type": "log",
                        "text": (
                            f"[Deep Learning] Retrain complete on {p_metrics['device']}! "
                            f"PolicySamples={p_metrics['samples']} (loss={p_metrics['final_loss']:.4f}), "
                            f"CommanderSamples={c_metrics['samples']} (loss={c_metrics['final_loss']:.4f})"
                        ),
                    }
                )
            except Exception as exc:
                self.msg_queue.put(
                    {
                        "type": "log",
                        "text": f"[Deep Learning] Retrain error: {exc}",
                    }
                )
            finally:
                self.root.after(
                    0,
                    lambda: (
                        self.quick_retrain_btn.config(state="normal", text="⚡ Retrain AI"),
                        self._update_ai_mode_control(),
                        self._load_rl_history(),
                        self._draw_performance(),
                    ),
                )

        threading.Thread(target=_worker, daemon=True, name="QuickRetrainThread").start()

    def _load_rl_history(self):
        path = user_path("dataset/self_learned_raids.jsonl")
        entries = []
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            continue
            except Exception:
                pass
        self.rl_history = entries[-40:]

    def _draw_hero_banner(self, _event=None):
        canvas = self.hero_canvas
        width = max(600, canvas.winfo_width())
        height = max(84, canvas.winfo_height())
        canvas.delete("all")

        grad = Image.new("RGB", (width, height), "#0d1629")
        draw = ImageDraw.Draw(grad)
        for x in range(width):
            t = x / max(1, width - 1)
            if t < 0.55:
                r = int(12 + 16 * (t / 0.55))
                g = int(20 + 26 * (t / 0.55))
                b = int(38 + 48 * (t / 0.55))
            else:
                u = (t - 0.55) / 0.45
                r = int(28 + 85 * u)
                g = int(46 + 38 * u)
                b = int(86 - 25 * u)
            draw.line([(x, 0), (x, height)], fill=(r, g, b))

        for offset in range(0, 320, 42):
            x0 = width - 340 + offset
            draw.line(
                [(x0, 0), (x0 + 70, height)],
                fill=(251, 191, 36) if offset % 84 == 0 else (56, 189, 248),
                width=1,
            )

        self._banner_photo = ImageTk.PhotoImage(grad)
        canvas.create_image(0, 0, image=self._banner_photo, anchor="nw")

        canvas.create_text(
            22,
            10,
            text="Coc-",
            anchor="nw",
            fill="#ffffff",
            font=("Segoe UI", 23, "bold"),
        )
        canvas.create_text(
            96,
            10,
            text="DeepFarm",
            anchor="nw",
            fill="#818cf8",
            font=("Segoe UI", 23, "bold"),
        )
        canvas.create_text(
            24,
            46,
            text="Autonomous Dual-Brain Clash of Clans Deep Learning Bot",
            anchor="nw",
            fill="#e2e8f0",
            font=("Segoe UI", 10, "bold"),
        )
        canvas.create_text(
            24,
            68,
            text="YOLO Vision   •   High-Contrast OCR   •   CoCCommanderNet   •   AttackPolicyNet   •   Online Self-Learning RL",
            anchor="nw",
            fill="#94a3b8",
            font=("Segoe UI", 8),
        )

    def _draw_state_stepper(self):
        canvas = self.stepper_canvas
        canvas.delete("all")
        width = max(240, canvas.winfo_width())
        steps = [
            ("home", "⌂", "Home"),
            ("find_match", "⌕", "Find Match"),
            ("scouting", "◎", "Scouting"),
            ("battle", "⚔", "Battle"),
            ("post_battle", "🏆", "Post Battle"),
        ]
        scr = (self.game_screen or "").lower()
        active_idx = 2
        if "home" in scr or "village" in scr:
            active_idx = 0
        elif "find" in scr or "cloud" in scr or "search" in scr:
            active_idx = 1
        elif "scout" in scr or "next" in scr:
            active_idx = 2
        elif "battle" in scr and "post" not in scr:
            active_idx = 3
        elif "post" in scr or "end" in scr or "return" in scr:
            active_idx = 4

        n = len(steps)
        margin = 26
        span = max(1, width - 2 * margin)
        cy = 20

        for i in range(n - 1):
            x1 = margin + span * i / (n - 1)
            x2 = margin + span * (i + 1) / (n - 1)
            line_col = "#6366f1" if i < active_idx else "#233252"
            canvas.create_line(x1, cy, x2, cy, fill=line_col, width=2)

        for i, (_key, icon_txt, label_txt) in enumerate(steps):
            cx = margin + span * i / (n - 1)
            is_active = i == active_idx
            r = 14 if is_active else 12
            fill_col = "#5865f2" if is_active else "#17233b"
            out_col = "#818cf8" if is_active else "#2a3c61"
            canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r, fill=fill_col, outline=out_col, width=2
            )
            canvas.create_text(
                cx,
                cy,
                text=icon_txt,
                fill="#ffffff" if is_active else self.MUTED,
                font=("Segoe UI Symbol", 9, "bold"),
            )
            canvas.create_text(
                cx,
                cy + 25,
                text=label_txt,
                fill="#ffffff" if is_active else self.MUTED,
                font=("Segoe UI", 7, "bold" if is_active else "normal"),
            )

    def _metric_card(self, parent, col, title, key, accent, hint, icon_key):
        card = self._panel(parent, padx=10, pady=8, bg=self.PANEL)
        card.grid(
            row=0,
            column=col,
            sticky="ew",
            padx=(0 if col == 0 else 4, 4 if col < 5 else 0),
        )
        card.grid_columnconfigure(1, weight=1)

        tk.Label(
            card,
            image=self._icons[icon_key],
            bg=self.PANEL,
        ).grid(row=0, column=0, rowspan=3, padx=(0, 10), sticky="n")

        self._label(card, title, size=8, color=self.MUTED).grid(
            row=0, column=1, sticky="w"
        )
        var = tk.StringVar(value="0")
        self.metric_vars[key] = var
        tk.Label(
            card,
            textvariable=var,
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Segoe UI", 14, "bold"),
        ).grid(row=1, column=1, sticky="w", pady=(1, 1))

        bottom_row = tk.Frame(card, bg=self.PANEL)
        bottom_row.grid(row=2, column=1, sticky="ew")
        hint_var = tk.StringVar(value=f"↑ {hint}")
        self.metric_hint_vars[key] = hint_var
        tk.Label(
            bottom_row,
            textvariable=hint_var,
            bg=self.PANEL,
            fg=self.GREEN if key in ("gold", "elixir", "delixir", "progress") else self.MUTED,
            font=("Segoe UI", 7, "bold"),
        ).pack(side="left")

        mini = tk.Canvas(
            bottom_row, bg=self.PANEL, width=38, height=12, highlightthickness=0, bd=0
        )
        mini.pack(side="right")
        heights = [4, 7, 5, 9, 11]
        for idx, h in enumerate(heights):
            bx = idx * 7 + 2
            mini.create_rectangle(
                bx, 12 - h, bx + 4, 12, fill=accent, width=0
            )

    def _target_row(self, parent, row, label, variable, accent, icon_key=None):
        row_frame = tk.Frame(
            parent,
            bg="#0b1324",
            padx=6,
            pady=2,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        row_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)
        row_frame.grid_columnconfigure(2, weight=1)

        if icon_key and icon_key in self._icons:
            tk.Label(
                row_frame,
                image=self._icons[icon_key],
                bg="#0b1324",
            ).grid(row=0, column=0, padx=(0, 6))
        else:
            tk.Label(
                row_frame,
                text="●",
                bg="#0b1324",
                fg=accent,
                font=("Segoe UI", 9),
            ).grid(row=0, column=0, padx=(0, 6))

        tk.Label(
            row_frame,
            text=label,
            bg="#0b1324",
            fg=self.MUTED,
            width=5,
            anchor="w",
            font=("Segoe UI", 8, "bold"),
        ).grid(row=0, column=1, sticky="w", padx=(0, 4))

        ttk.Entry(
            row_frame,
            textvariable=variable,
            width=11,
            style="Dark.TEntry",
        ).grid(row=0, column=2, sticky="ew")

    def _on_start(self):
        try:
            targets = (self.min_gold.get(), self.min_elixir.get(), self.min_dark.get())
            goals = (self.goal_gold.get(), self.goal_elixir.get(), self.goal_dark.get())
            if any(value < 0 for value in targets + goals):
                raise ValueError("Loot targets and goals must be non-negative integers.")
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid target", str(exc) or "Enter non-negative integers.")
            return

        # Pre-Flight Check 1: Ensure all 16 Clash of Clans button coordinates exist (Universal Preset fallback)
        merged_controls = ensure_default_controls(getattr(self.bot, "controls", None))
        if getattr(self.bot, "controls", None) != merged_controls:
            self.bot.controls = merged_controls
            self.settings["controls"] = merged_controls
            save_settings(self.settings)
            self._log("[Pre-Flight] Koordinat tombol CoC otomatis dilengkapi dari Universal Preset.")

        # Pre-Flight Check 2: Auto-connect ADB if emulator wasn't connected when GUI opened
        adb_obj = getattr(getattr(self.bot, "window", None), "adb", None)
        if adb_obj is not None and not getattr(adb_obj, "serial", None):
            self._log("[Pre-Flight] Menghubungkan otomatis ke emulator via ADB...")
            if adb_obj.connect():
                self.bot.window.start_video_capture()
                if hasattr(self, "adb_connect_btn"):
                    self.adb_connect_btn.config(
                        text="🔌 ADB Ready", bg="#143829", fg=self.GREEN
                    )
                self._log(f"[Pre-Flight] Emulator terhubung ({adb_obj.serial})!")
            else:
                messagebox.showwarning(
                    "Emulator Belum Terdeteksi",
                    "Bot belum bisa menemukan emulator Android yang aktif.\n\n"
                    "Langkah mudah:\n"
                    "1. Buka emulator (LDPlayer / MuMu / BlueStacks / MEmu).\n"
                    "2. Buka game Clash of Clans.\n"
                    "3. Klik tombol '▶ Start Farming' lagi!",
                )
                return

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.ai_mode_btn.config(state="disabled")
        self._set_bot_status("starting")
        self.bot.start(targets, goals)
        if not self.bot.is_running():
            self._set_bot_status("stopped")

    def _current_strategy_mode(self) -> str:
        if not bool(getattr(self.bot, "ai_enabled", False)):
            return "rule"
        return "coc_ai"

    def _on_ai_mode_toggle(self):
        order = ("coc_ai", "rule")
        current = self._current_strategy_mode()
        candidate_mode = order[(order.index(current) + 1) % len(order)] if current in order else "coc_ai"
        try:
            if hasattr(self.bot, "set_strategy_mode"):
                self.bot.set_strategy_mode(candidate_mode)
            else:
                self.bot.set_ai_enabled(candidate_mode != "rule")
            candidate = dict(self.settings)
            brain_settings = candidate.get("brain")
            brain_settings = (
                dict(brain_settings) if isinstance(brain_settings, dict) else {}
            )
            brain_settings["name"] = candidate_mode
            brain_settings["use_ai"] = candidate_mode != "rule"
            candidate["brain"] = brain_settings
            save_settings(candidate)
            self.settings = candidate
        except (OSError, RuntimeError) as exc:
            messagebox.showerror("Strategy mode", str(exc))
        self.ai_mode_var.set(bool(getattr(self.bot, "ai_enabled", False)))
        self._update_ai_mode_control()

    def _update_ai_mode_control(self):
        mode = self._current_strategy_mode()
        planner = getattr(self.bot, "policy_planner", None)
        commander = getattr(self.bot, "brain_alt", None)
        policy_ready = bool(getattr(planner, "loaded", False))
        suffix = " + PolicyNet" if policy_ready else ""
        labels = {
            "coc_ai": f"CoC Deep AI (Commander{suffix})  ▾",
            "rule": f"Rule-based{suffix}  ▾",
        }
        self.ai_mode_text.set(labels.get(mode, labels["coc_ai"]))
        running = self.bot_status in (
            "running", "farming", "loading", "starting", "stopping"
        )
        self.ai_mode_btn.config(
            state="disabled" if running else "normal",
            fg=self.TEXT,
        )

        # Refresh Neural Memory Card & Header Badge
        prof = getattr(planner, "strategy_profile", {}) if planner else {}
        c_stats = getattr(commander, "stats", {}) if commander else {}
        mem_bank_obj = getattr(commander, "memory_bank", None) if commander else None
        mem_count = len(getattr(mem_bank_obj, "entries", [])) if mem_bank_obj else int(c_stats.get("memory_bank_size", 0))
        mem_recalls = int(getattr(mem_bank_obj, "total_recalls", c_stats.get("memory_recalls", 0)))
        dev = getattr(planner, "device", None) or getattr(commander, "device", "cuda:0")
        raids_self = int(prof.get("raids_self_learned", 0))
        cmd_learned = int(c_stats.get("raids_learned", 0))
        avg_rew = float(prof.get("ema_reward", c_stats.get("avg_reward", 0.70)))
        min_r = float(prof.get("min_deploy_radius", 0.26))
        sigma = float(prof.get("explore_sigma", 0.025))
        order = prof.get("deploy_order") or c_stats.get("preferred_order") or []
        short_order = "→".join(s.replace("troop_slot_", "S").replace("s_goblin", "Gob") for s in order[:4])

        if hasattr(self, "brain_device_badge"):
            self.brain_device_badge.config(
                text=f"🧠 CoC Neural ({str(dev).upper()}) · {cmd_learned + raids_self} Ep · {mem_count} Mem"
            )
        if hasattr(self, "ai_telemetry_title"):
            self.ai_telemetry_title.set(
                f"🧠 Neural: Reward {avg_rew * 100:.0f}% · Memory: {mem_count} Base ({mem_recalls} Recall)"
            )
        if hasattr(self, "ai_telemetry_detail"):
            self.ai_telemetry_detail.set(
                f"Radius={min_r:.3f}  Sigma={sigma:.3f}  Order=[{short_order or 'default'}]"
            )

    def _on_learn_from_video(self):
        if self.bot.is_running():
            messagebox.showwarning(
                "Learn from Video",
                "Stop the bot first before training from an attack video.",
            )
            return
        video_path = filedialog.askopenfilename(
            title="Select Clash of Clans Attack Gameplay Video",
            filetypes=[
                ("Video files", "*.mp4 *.mkv *.avi *.mov"),
                ("All files", "*.*"),
            ],
        )
        if not video_path:
            return

        self.learn_video_btn.config(state="disabled", text="⏳ Training CoC AI...")
        self._log(f"[Deep Learning] Extracting attack demonstrations from: {os.path.basename(video_path)}")

        def _worker():
            try:
                self.bot.vision.load()
                extractor = VideoAttackExtractor(
                    vision=self.bot.vision,
                    slot_names=list(self.bot.troops.get("slots") or ["s_goblin"]),
                    controls=self.bot.controls,
                )
                demo = extractor.extract_from_video(video_path, sample_fps=4.0)
                self.msg_queue.put(
                    {
                        "type": "log",
                        "text": (
                            f"[Deep Learning] Extracted {len(demo.get('taps', []))} "
                            f"human tap samples and {len(demo.get('storages', []))} storages."
                        ),
                    }
                )
                metrics = train_policy_from_dataset(epochs=60)
                cmd_metrics = train_commander_from_data(epochs=80)
                if getattr(self.bot, "policy_planner", None) is not None:
                    self.bot.policy_planner.reload()
                if getattr(self.bot, "brain_alt", None) is not None and hasattr(self.bot.brain_alt, "reload"):
                    self.bot.brain_alt.reload()
                self.msg_queue.put(
                    {
                        "type": "log",
                        "text": (
                            f"[Deep Learning] Trained CoCCommanderNet + PolicyNet on {metrics['device']}! "
                            f"Samples={metrics['samples']}, PolicyLoss={metrics['final_loss']:.4f}, "
                            f"CommanderLoss={cmd_metrics['final_loss']:.4f}"
                        ),
                    }
                )
            except Exception as exc:
                self.msg_queue.put(
                    {
                        "type": "log",
                        "text": f"[Deep Learning] Training error: {exc}",
                    }
                )
            finally:
                self.root.after(
                    0,
                    lambda: (
                        self.learn_video_btn.config(
                            state="normal", text="🎬 Learn Attack (.mp4)"
                        ),
                        self._update_ai_mode_control(),
                    ),
                )

        threading.Thread(target=_worker, daemon=True, name="VideoLearnerThread").start()

    def _on_target_mode_cycle(self):
        order = ("smart", "and", "or")
        current = str(self.target_mode_var.get() or "smart").lower()
        next_mode = order[(order.index(current) + 1) % len(order)] if current in order else "smart"
        self.target_mode_var.set(next_mode)
        if hasattr(self.bot, "set_target_mode"):
            self.bot.set_target_mode(next_mode)
        candidate = dict(self.settings)
        attack_cfg = candidate.get("attack")
        attack_cfg = dict(attack_cfg) if isinstance(attack_cfg, dict) else {}
        attack_cfg["target_mode"] = next_mode
        candidate["attack"] = attack_cfg
        if save_settings(candidate):
            self.settings = candidate
        self._update_target_mode_control()

    def _update_target_mode_control(self):
        labels = {
            "smart": "Smart (Combo + Dead Base)  ▾",
            "and": "Strict (All Targets / AND)  ▾",
            "or": "Flexible (Any Target / OR)  ▾",
        }
        mode = str(self.target_mode_var.get() or "smart").lower()
        self.target_mode_text.set(labels.get(mode, labels["smart"]))

    def _on_stop(self):
        self.stop_btn.config(state="disabled")
        self.bot.stop()

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state="disabled")

    def _log(self, text):
        self.log_text.config(state="normal")
        lower = str(text).lower()
        is_err = any(word in lower for word in ("error", "failed", "exception", "cannot"))
        is_ai = any(word in lower for word in ("self-learning", "deep learning", "brain ("))
        is_ok = any(word in lower for word in ("completed", "success", "stopped cleanly", "meets target", "reinforce"))
        badge_tag = "error" if is_err else ("ai" if is_ai else ("success" if is_ok else "info"))
        badge_txt = " ERR  " if is_err else (" AI   " if is_ai else (" OK   " if is_ok else " INFO "))
        msg_tag = "msg_error" if is_err else ("msg_ai" if is_ai else ("msg_success" if is_ok else ""))

        now_str = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"{now_str} ", "time")
        self.log_text.insert(tk.END, badge_txt, badge_tag)
        self.log_text.insert(tk.END, f"  {text}\n", msg_tag if msg_tag else ())
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        self.footer_var.set(str(text)[-150:])

    def _set_bot_status(self, status):
        self.bot_status = str(status or "unknown").lower()
        label = self.bot_status.replace("_", " ").upper()
        color = self.GREEN if self.bot_status in ("running", "farming") else (
            self.RED if self.bot_status in ("error", "stopping") else self.CYAN
        )
        self.connection_badge.config(text=f"● BOT {label}", fg=color)
        self.status_detail.set(f"Bot status: {label}  ·  F12 emergency stop")
        if self.bot_status in ("running", "farming", "loading", "starting") and self.started_at is None:
            self.started_at = time.time()
        if self.bot_status in ("stopped", "error"):
            if self.started_at is not None:
                self.last_elapsed = max(0, time.time() - self.started_at)
                self.started_at = None
            self.start_btn.config(state="normal", bg="#1fa855")
            self.stop_btn.config(state="disabled", bg="#3b1822")
            self._update_ai_mode_control()
        elif self.bot_status in (
            "running", "farming", "loading", "starting", "stopping"
        ):
            self.start_btn.config(state="disabled", bg="#14532d")
            self.stop_btn.config(state="normal", bg="#991b1b")
            self._update_ai_mode_control()

    def _load_recent_raids(self):
        replay_cfg = getattr(self.bot, "replay_config", {}) or {}
        if replay_cfg.get("enabled", True) is False:
            return
        directory = str(replay_cfg.get("dir") or "logs")
        if not os.path.isabs(directory):
            directory = user_path(directory)
        raids = []
        for path in glob.glob(os.path.join(directory, "raid_*.json")):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if not isinstance(data, dict):
                    continue
                outcome = data.get("outcome") or {}
                state = data.get("state") or {}
                base_loot = state.get("loot") or {}
                stats = {
                    "timestamp": os.path.getmtime(path),
                    "status": outcome.get("status", "pending"),
                    "base_loot": {
                        "gold": int(base_loot.get("gold", 0) or 0),
                        "elixir": int(base_loot.get("elixir", 0) or 0),
                        "delixir": int(base_loot.get("delixir", 0) or 0),
                    },
                    "loot_gain": outcome.get("loot_gain") or {"gold": 0, "elixir": 0, "delixir": 0},
                    "drained": outcome.get("drained"),
                }
                raids.append(stats)
            except (OSError, ValueError, TypeError):
                continue
        raids.sort(key=lambda item: item.get("timestamp", 0))
        self.raids = raids[-100:]
        self.session_raid_count = len(self.raids)

    def _format_num(self, value):
        try:
            return f"{int(float(value or 0)):,}"
        except (ValueError, TypeError):
            return "0"

    def _format_rate(self, amount: float, elapsed_s: float) -> str:
        if elapsed_s < 15 or amount <= 0:
            return "↑ Session gains"
        per_hr = amount * (3600.0 / elapsed_s)
        if per_hr >= 1_000_000:
            return f"↑ {per_hr / 1_000_000:.1f}M / hr"
        if per_hr >= 1_000:
            return f"↑ {per_hr / 1_000:.0f}K / hr"
        return f"↑ {per_hr:.0f} / hr"

    def _record_raid(self, raid):
        if not isinstance(raid, dict):
            return
        row = dict(raid)
        row.setdefault("timestamp", time.time())
        self.raids.append(row)
        self.raids = self.raids[-100:]
        if row.get("status") in ("attacked", "skipped"):
            self.session_raid_count += 1
        self._load_rl_history()
        self._update_ai_mode_control()
        self._render_raids()
        self._refresh_metrics()
        if self.chart_mode == "ai":
            self._draw_performance()

    def _refresh_metrics(self):
        elapsed = time.time() - self.started_at if self.started_at is not None else self.last_elapsed
        seconds = int(max(0, elapsed))

        for key in ("gold", "elixir", "delixir"):
            val = self.session_totals.get(key, 0)
            self.metric_vars[key].set(self._format_num(val))
            if hasattr(self, "metric_hint_vars") and key in self.metric_hint_vars:
                self.metric_hint_vars[key].set(self._format_rate(val, elapsed))

        self.metric_vars["raids"].set(str(self.session_raid_count))
        if hasattr(self, "metric_hint_vars") and "raids" in self.metric_hint_vars:
            drains = [
                float(r["drained"])
                for r in self.raids[-30:]
                if isinstance(r.get("drained"), (int, float))
            ]
            if drains:
                self.metric_hint_vars["raids"].set(f"↑ Avg drain {sum(drains) / len(drains) * 100:.0f}%")

        try:
            goals = {
                "gold": max(0, int(self.goal_gold.get())),
                "elixir": max(0, int(self.goal_elixir.get())),
                "delixir": max(0, int(self.goal_dark.get())),
            }
            positive = [
                min(1.0, self.session_totals[key] / goal)
                for key, goal in goals.items() if goal > 0
            ]
            progress = sum(positive) / len(positive) * 100 if positive else None
            self.metric_vars["progress"].set(f"{progress:.0f}%" if progress is not None else "—")
        except (tk.TclError, ValueError):
            self.metric_vars["progress"].set("—")

        self.metric_vars["runtime"].set(f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}")

    def _render_raids(self):
        for child in self.raid_tree.get_children():
            self.raid_tree.delete(child)
        for index, raid in enumerate(reversed(self.raids[-8:]), 1):
            timestamp = raid.get("timestamp")
            try:
                time_text = time.strftime("%H:%M:%S", time.localtime(float(timestamp)))
            except (TypeError, ValueError, OSError):
                time_text = "—"
            gain = raid.get("loot_gain") or {}
            status = str(raid.get("status") or "pending")
            drained = raid.get("drained")
            drain_text = f"{float(drained) * 100:.0f}%" if isinstance(drained, (int, float)) else "—"
            tag = status if status in ("attacked", "skipped", "error") else ""
            row_tags = (tag, "odd") if index % 2 == 1 else (tag,)
            self.raid_tree.insert(
                "",
                "end",
                values=(
                    time_text,
                    status.upper(),
                    self._format_num(gain.get("gold", 0)),
                    self._format_num(gain.get("elixir", 0)),
                    drain_text,
                ),
                tags=row_tags,
            )

    def _render_detections(self):
        for child in self.det_tree.get_children():
            self.det_tree.delete(child)
        detections = self.latest_detections or {}
        rows = sorted(
            detections.items(),
            key=lambda item: (-len(item[1]), item[0])
        )
        total_count = sum(len(items) for _, items in rows)
        if hasattr(self, "det_count_badge"):
            self.det_count_badge.config(text=str(total_count))
        for idx, (name, items) in enumerate(rows):
            confidences = [float(item.get("confidence", 0) or 0) for item in items]
            average = sum(confidences) / len(confidences) if confidences else 0
            row_tag = "odd" if idx % 2 == 1 else "even"
            self.det_tree.insert(
                "", "end", values=(name, len(items), f"{average:.2f}"), tags=(row_tag,)
            )
        if not rows:
            self.det_tree.insert("", "end", values=("Waiting for scan", "—", "—"))

    def _render_preview(self):
        if self.latest_frame is None:
            return
        try:
            frame = self.latest_frame
            fh, fw = frame.shape[:2]
            image = Image.fromarray(frame[:, :, ::-1].copy())
            draw = ImageDraw.Draw(image)

            # 1. Draw YOLO bounding boxes
            for name, items in (self.latest_detections or {}).items():
                color = "#38bdf8" if "gold" not in name.lower() else "#fbbf24"
                if "elixir" in name.lower():
                    color = "#e879f9"
                for item in items:
                    bbox = item.get("bbox") or []
                    if len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
                    draw.text((x1 + 3, max(0, y1 - 14)), f"{name} {float(item.get('confidence', 0) or 0):.2f}", fill=color)

            # 2. Draw AI Tactical Radar Overlay (Perimeter Ring + Predicted Tap Targets + HUD)
            if getattr(self, "show_ai_radar", True):
                overlay = getattr(self, "latest_ai_overlay", None) or {}
                planner = getattr(self.bot, "policy_planner", None)
                prof = getattr(planner, "strategy_profile", {}) if planner else {}
                min_r = float(overlay.get("min_radius") or prof.get("min_deploy_radius", 0.26))

                # Draw Perimeter Safe-Ring around base center (0.50, 0.45)
                cx, cy = int(fw * 0.50), int(fh * 0.45)
                rx, ry = int(fw * min_r), int(fh * min_r)
                draw.ellipse(
                    (cx - rx, cy - ry, cx + rx, cy + ry),
                    outline="#22d3ee",
                    width=2,
                )

                # Draw AI Planned Tap Crosshairs
                tap_pts = overlay.get("tap_points") or []
                for px, py in tap_pts[:32]:
                    ix, iy = int(px), int(py)
                    draw.ellipse((ix - 7, iy - 7, ix + 7, iy + 7), outline="#22c55e", width=2)
                    draw.line((ix - 10, iy, ix + 10, iy), fill="#22c55e", width=1)
                    draw.line((ix, iy - 10, ix, iy + 10), fill="#22c55e", width=1)

                # Draw AI Tactical HUD Banner on top-left if overlay telemetry exists
                if overlay:
                    pred_p = float(overlay.get("predicted_profit", 0.0) or 0.0)
                    flank = str(overlay.get("side") or "all").upper()
                    dead_tag = " · DEAD" if overlay.get("dead_base") else ""
                    if overlay.get("memory_match"):
                        m_sim = float(overlay.get("memory_similarity", 0.0) or 0.0) * 100.0
                        m_act = "FIX" if overlay.get("memory_action") == "AVOID_MISTAKE" else "WIN"
                        mem_tag = f"  |  MEM: {m_sim:.0f}% ({m_act})"
                    else:
                        mem_tag = f"  |  MEM: NEW ({int(overlay.get('memory_bank_size', 0))})"
                    hud_txt = (
                        f"AI BRAIN  |  PROFIT: {pred_p * 100:.0f}%  |  FLANK: {flank}"
                        f"{mem_tag}  |  R={min_r:.3f}{dead_tag}"
                    )
                    draw.rectangle((14, 14, 595, 40), fill="#0f172a", outline="#38bdf8", width=1)
                    draw.text((22, 20), hud_txt, fill="#38bdf8")

            width = max(320, self.preview_label.winfo_width() - 12)
            height = max(190, self.preview_label.winfo_height() - 12)
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            self._photo = ImageTk.PhotoImage(image)
            self.preview_label.config(image=self._photo, text="")
            self.preview_info.set(
                f"{fw} × {fh}   ·   {self.game_screen.upper()}   ·   "
                f"Frame {time.strftime('%H:%M:%S', time.localtime(self._last_telemetry_at or time.time()))}"
            )
            self.live_badge.config(text="● LIVE", bg=self.GREEN_BG, fg=self.GREEN)
        except Exception as exc:
            self.preview_info.set(f"Preview render error: {exc}")

    def _queue_preview_frame(self, frame):
        with self._preview_frame_lock:
            self._pending_preview_frame = frame
            self._preview_generation += 1

    def _refresh_preview(self):
        with self._preview_frame_lock:
            frame = self._pending_preview_frame
            generation = self._preview_generation
            self._pending_preview_frame = None

        if frame is not None:
            self.latest_frame = frame
            self._last_telemetry_at = time.time()
        if generation != self._rendered_preview_generation or self._preview_needs_render:
            if self.latest_frame is not None:
                self._render_preview()
            self._rendered_preview_generation = generation
            self._preview_needs_render = False

        if not self._preview_stop.is_set():
            self.root.after(33, self._refresh_preview)

    def _draw_performance(self):
        canvas = self.perf_canvas
        canvas.delete("all")
        width = max(250, canvas.winfo_width())
        height = max(90, canvas.winfo_height())
        left, right, top, bottom = 38, width - 12, 22, height - 18

        if getattr(self, "chart_mode", "ai") == "ai" and getattr(self, "rl_history", None):
            # Plot AI Online RL Self-Learning Curve (Reward % & Drain %)
            legend_items = [
                ("Reward %", "#a855f7"),
                ("Drain %", self.CYAN),
                ("Reinforce", self.GREEN),
                ("Mistake Fix", self.GOLD),
            ]
            lx = left
            for label, color in legend_items:
                canvas.create_oval(lx, 6, lx + 7, 13, fill=color, outline=color)
                canvas.create_text(
                    lx + 10, 9, text=label, anchor="w", fill=self.MUTED, font=("Segoe UI", 7)
                )
                lx += 62

            for i in range(4):
                y = top + (bottom - top) * i / 3
                canvas.create_line(left, y, right, y, fill="#192844")
                val_pct = int(round(100 * (3 - i) / 3))
                canvas.create_text(
                    left - 5, y, text=f"{val_pct}%", anchor="e", fill=self.MUTED, font=("Segoe UI", 6)
                )

            samples = self.rl_history[-30:]
            if len(samples) == 1:
                samples = samples * 2
            reward_pts = []
            drain_pts = []
            for idx, item in enumerate(samples):
                x = left + (right - left) * idx / max(1, len(samples) - 1)
                rew = max(0.0, min(1.0, float(item.get("reward", 0.0))))
                drn = max(0.0, min(1.0, float(item.get("drain_ratio", 0.0))))
                ry = bottom - rew * (bottom - top)
                dy = bottom - drn * (bottom - top)
                reward_pts.extend((x, ry))
                drain_pts.extend((x, dy))

            if len(drain_pts) >= 4:
                canvas.create_line(*drain_pts, fill=self.CYAN, width=1)
            if len(reward_pts) >= 4:
                canvas.create_line(*reward_pts, fill="#a855f7", width=2)
                for idx, item in enumerate(samples):
                    px, py = reward_pts[idx * 2], reward_pts[idx * 2 + 1]
                    dot_col = self.GOLD if item.get("is_mistake_correction") else self.GREEN
                    canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill=dot_col, outline=dot_col)
            return

        # System Latency (ms) Graph
        legend_items = [
            ("Capture", self.CYAN),
            ("YOLO", self.ELIXIR),
            ("OCR", self.GREEN),
        ]
        lx = left
        for label, color in legend_items:
            canvas.create_oval(lx, 6, lx + 7, 13, fill=color, outline=color)
            canvas.create_text(
                lx + 11, 9, text=label, anchor="w", fill=self.MUTED, font=("Segoe UI", 7)
            )
            lx += 75

        for i in range(4):
            y = top + (bottom - top) * i / 3
            canvas.create_line(left, y, right, y, fill="#192844")
        values = self.performance_history[-40:]
        maximum = max(
            [100.0] + [
                float(sample.get(key, 0) or 0)
                for sample in values for key in ("capture_ms", "yolo_ms", "ocr_ms")
            ]
        )
        for i in range(4):
            value = maximum * (3 - i) / 3
            y = top + (bottom - top) * i / 3
            canvas.create_text(
                left - 5, y, text=f"{value:.0f} ms", anchor="e", fill=self.MUTED, font=("Segoe UI", 6)
            )
        if len(values) < 2:
            canvas.create_text(
                width / 2,
                (top + bottom) / 2,
                text="Waiting for real timing samples…",
                fill=self.MUTED,
                font=("Segoe UI", 8),
            )
            return
        for key, color in [
            ("capture_ms", self.CYAN),
            ("yolo_ms", self.ELIXIR),
            ("ocr_ms", self.GREEN),
        ]:
            points = []
            for i, sample in enumerate(values):
                x = left + (right - left) * i / max(1, len(values) - 1)
                y = bottom - min(maximum, float(sample.get(key, 0) or 0)) / maximum * (bottom - top)
                points.extend((x, y))
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=2, smooth=False)
                for idx in range(0, len(points), 2):
                    px, py = points[idx], points[idx + 1]
                    canvas.create_oval(px - 2, py - 2, px + 2, py + 2, fill=color, outline=color)

    def _apply_dashboard_message(self, msg):
        if msg.get("screen"):
            self.game_screen = str(msg["screen"])
            state = self.game_screen.replace("_", " ").title()
            bg_col = "#166534" if self.game_screen in ("battle", "post_battle") else "#3730a3"
            fg_col = "#dcfce7" if self.game_screen in ("battle", "post_battle") else "#e0e7ff"
            self.game_state_badge.config(text=state, bg=bg_col, fg=fg_col)
            self._draw_state_stepper()
        if msg.get("stage"):
            self.stage = str(msg["stage"])
            self.stage_var.set(self.stage)
        if "ai_overlay" in msg:
            incoming_ov = msg.get("ai_overlay") or {}
            self.latest_ai_overlay.update(incoming_ov)
            self._preview_needs_render = True
        if "detections" in msg:
            self.latest_detections = msg.get("detections") or {}
            self._render_detections()
            self._preview_needs_render = True
        if "loot" in msg:
            incoming = msg.get("loot") or {}
            self.latest_loot = {
                "gold": incoming.get("gold", incoming.get("post_gold")),
                "elixir": incoming.get("elixir", incoming.get("post_elixir")),
                "delixir": incoming.get("delixir", incoming.get("d_elixir", incoming.get("post_d_elixir"))),
            }
            parts = []
            for name, key in [("G", "gold"), ("E", "elixir"), ("D", "delixir")]:
                value = self.latest_loot.get(key)
                if value is not None:
                    parts.append(f"{name}: {self._format_num(value)}")
            if parts:
                self.loot_var.set("Last scanned loot  ·  " + "   ".join(parts))
        if "metrics" in msg:
            metrics = msg.get("metrics") or {}
            self.latest_metrics.update(metrics)
            for key in self.perf_vars:
                self.perf_vars[key].set(f"{float(self.latest_metrics.get(key, 0) or 0):.0f} ms")
            now = time.monotonic()
            if msg.get("perf_sample") and now - self._last_metric_sample >= 0.7:
                self.performance_history.append(dict(self.latest_metrics))
                self.performance_history = self.performance_history[-60:]
                self._last_metric_sample = now
                self._draw_performance()
        if msg.get("frame") is not None:
            self._queue_preview_frame(msg["frame"])
            if "detections" not in msg:
                self.latest_detections = {}
                self._render_detections()
        if msg.get("raid"):
            self._record_raid(msg["raid"])
        self._refresh_metrics()

    def _refresh_all(self):
        self._refresh_metrics()
        self._render_raids()
        self._render_detections()
        self._draw_state_stepper()
        self._draw_performance()

    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                mtype = msg.get("type")
                if mtype == "log":
                    self._log(msg.get("text", ""))
                elif mtype == "status":
                    self._set_bot_status(msg.get("state", "unknown"))
                elif mtype == "alert":
                    messagebox.showinfo("Coc-DeepFarm alert", msg.get("text", ""))
                elif mtype == "totals":
                    data = msg.get("data") or {}
                    self.session_totals = {
                        "gold": int(data.get("gold", 0) or 0),
                        "elixir": int(data.get("elixir", 0) or 0),
                        "delixir": int(data.get("delixir", data.get("d_elixir", 0)) or 0),
                    }
                    self._refresh_metrics()
                elif mtype == "dashboard":
                    self._apply_dashboard_message(msg)
        except queue.Empty:
            pass
        finally:
            self.root.after(150, self._poll_queue)

    def _idle_preview_loop(self):
        while not self._preview_stop.is_set():
            try:
                stream = getattr(self.bot.window, "video_stream", None)
                if stream is not None and stream.is_active:
                    frame, _age_ms = stream.capture()
                    if frame is not None:
                        self._queue_preview_frame(frame)
                    self._preview_stop.wait(1.0 / 20.0)
                    continue
                adb_obj = getattr(getattr(self.bot, "window", None), "adb", None)
                if not self.bot.is_running() and getattr(adb_obj, "serial", None):
                    frame = self.bot.window.capture_window()
                    self._queue_preview_frame(frame)
            except Exception:
                pass
            self._preview_stop.wait(0.1 if self.bot.is_running() else 1.0)

    def _on_close(self):
        self._preview_stop.set()
        try:
            if self.bot.is_running():
                self.bot.stop()
        except Exception:
            pass
        self.root.destroy()

    def _tick(self):
        self._refresh_metrics()
        self.root.after(1000, self._tick)
