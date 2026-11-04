"""Command manager UI + modding project helper.

Run with: python main_ui.py

Commands, notes, order, and working directories are saved automatically to
commands.json in the same folder as this file.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "commands.json"
SETTINGS_PATH = APP_DIR / "settings.json"

# Modding project layout (all next to this .py file)
LEVELS = [f"level_{i:02d}" for i in range(9)]  # level_00 .. level_08
PROJECT_DIR = APP_DIR / "Modding_project"
ORG_DIR = PROJECT_DIR / "Org_pfguns" / "levels"                      # original .db / .wdd copied from game
ANIM_DIR = PROJECT_DIR / "org_pfguns_anim_extracted" / "levels"      # extracted anims, one folder per db
MOD_DIR = PROJECT_DIR / "pfguns_anim_mod" / "levels"                      # modded anims (one per edited clip) -> packed into the db
FBX_DIR = PROJECT_DIR / "org_pfguns_fbx_extracted" / "levels"        # <level>/<dbname>.fbx
WORK_DIR = PROJECT_DIR / "work"                                      # default place for json / modded files

DEFAULT_SETTINGS = {
    "game_root": "path",
    "level": "level_00",
    "db": "",
    "clip": "",
    "tools_dir": "path",
    # EDIT THIS to your real extractor. {db} = selected db, {out_dir} = extracted folder for that db.
    "extract_template": 'python black_anim_extract.py "{db}" "{out_dir}"',
    # {db} = selected db, {out_fbx} = Modding_project/org_pfguns_fbx_extracted/levels/<level>/<dbname>.fbx
    "fbx_template": 'python black_fps_extract_fbx_maya_v5.py "{db}" "{out_fbx}"',
    # "1" = every level/db/clip gets its own work files under Modding_project/work (recommended for many clips)
    "auto_paths": "1",
    "baked_json": str(WORK_DIR / "reload_baked.json"),
    "edit_json": str(WORK_DIR / "reload_edit.json"),
    "pos_json": str(WORK_DIR / "reload_edit_position_only.json"),
    "out_db": str(WORK_DIR / "modded.db"),
    "out_anim": str(WORK_DIR / "modded.anim"),
}

TOKEN_RE = re.compile(r"\{(\w+)\}")


class CommandEditor(tk.Toplevel):
    """Dialog used to add or edit one saved command."""

    def __init__(self, parent: "CommandManager", command: dict[str, Any] | None = None):
        super().__init__(parent)
        self.parent_app = parent
        self.original = command
        self.result: dict[str, Any] | None = None
        self.title("Edit Command" if command else "Add Command")
        self.geometry("760x650")
        self.minsize(620, 520)
        self.transient(parent)
        self.grab_set()

        source = command or {}
        default_order = source.get("order", parent.next_order())

        form = ttk.Frame(self, padding=12)
        form.pack(fill="both", expand=True)
        form.columnconfigure(1, weight=1)
        form.rowconfigure(7, weight=1)

        ttk.Label(form, text="Command name:").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        self.name_var = tk.StringVar(value=source.get("name", ""))
        ttk.Entry(form, textvariable=self.name_var).grid(row=0, column=1, sticky="ew", pady=5)

        ttk.Label(form, text="Order number:").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=5)
        self.order_var = tk.StringVar(value=str(default_order))
        ttk.Entry(form, textvariable=self.order_var, width=12).grid(row=1, column=1, sticky="w", pady=5)

        ttk.Label(form, text="Working folder (optional):").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=5)
        cwd_row = ttk.Frame(form)
        cwd_row.grid(row=2, column=1, sticky="ew", pady=5)
        cwd_row.columnconfigure(0, weight=1)
        self.cwd_var = tk.StringVar(value=source.get("working_directory", ""))
        ttk.Entry(cwd_row, textvariable=self.cwd_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(cwd_row, text="Browse…", command=self.browse_folder).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(form, text="Comment / notes:").grid(row=3, column=0, sticky="nw", padx=(0, 10), pady=5)
        self.comment_text = tk.Text(form, height=4, wrap="word", undo=True)
        self.comment_text.grid(row=3, column=1, sticky="ew", pady=5)
        self.comment_text.insert("1.0", source.get("comment", ""))

        ttk.Label(form, text="Command to run:").grid(row=4, column=0, sticky="nw", padx=(0, 10), pady=5)
        ttk.Label(
            form,
            text="Examples: python script.py  |  python tool.py input.anim  |  cd /d S:\\project then run a script",
            foreground="#555555",
            wraplength=520,
        ).grid(row=5, column=1, sticky="w", pady=(0, 4))

        self.command_text = tk.Text(form, height=12, wrap="none", undo=True)
        self.command_text.grid(row=7, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        self.command_text.insert("1.0", source.get("command", ""))

        buttons = ttk.Frame(form)
        buttons.grid(row=8, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(buttons, text="Save Command", command=self.save).pack(side="right")

        self.bind("<Escape>", lambda _event: self.destroy())
        self.bind("<Control-Return>", lambda _event: self.save())

    def browse_folder(self) -> None:
        initial = self.cwd_var.get().strip() or str(APP_DIR)
        chosen = filedialog.askdirectory(parent=self, initialdir=initial, title="Choose working folder")
        if chosen:
            self.cwd_var.set(chosen)

    def save(self) -> None:
        name = self.name_var.get().strip()
        command_text = self.command_text.get("1.0", "end-1c").strip()
        comment = self.comment_text.get("1.0", "end-1c").strip()
        cwd = self.cwd_var.get().strip()

        if not name:
            messagebox.showerror("Missing name", "Enter a name for this command.", parent=self)
            return
        if not command_text:
            messagebox.showerror("Missing command", "Enter the command you want to run.", parent=self)
            return
        try:
            order = int(self.order_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid order", "Order number must be a whole number, such as 1 or 2.", parent=self)
            return
        if order < 1:
            messagebox.showerror("Invalid order", "Order number must be 1 or greater.", parent=self)
            return

        self.result = {
            "id": (self.original or {}).get("id", str(uuid.uuid4())),
            "order": order,
            "name": name,
            "command": command_text,
            "comment": comment,
            "working_directory": cwd,
        }
        self.destroy()


class CommandManager(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Python / CMD Command Manager")
        self.geometry("1180x1000")
        self.minsize(950, 700)

        self.commands: list[dict[str, Any]] = []
        self.output_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.current_process: subprocess.Popen[str] | None = None
        self.is_running = False
        self.stop_requested = False
        self.run_thread: threading.Thread | None = None

        self.load_config()
        self.init_settings()
        self.build_ui()
        self.refresh_command_list()
        self.update_selected_details()
        self.refresh_db_list()
        self.after(100, self.process_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=(10, 10, 10, 6))
        toolbar.pack(fill="x")

        ttk.Button(toolbar, text="＋ Add Command", command=self.add_command).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="Edit", command=self.edit_command).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Delete", command=self.delete_command).pack(side="left", padx=5)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(toolbar, text="Move Up", command=lambda: self.move_selected(-1)).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Move Down", command=lambda: self.move_selected(1)).pack(side="left", padx=5)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        self.run_selected_button = ttk.Button(toolbar, text="▶ Run Selected", command=self.run_selected)
        self.run_selected_button.pack(side="left", padx=5)
        self.run_all_button = ttk.Button(toolbar, text="▶ Run All (by order)", command=self.run_all)
        self.run_all_button.pack(side="left", padx=5)
        self.stop_button = ttk.Button(toolbar, text="■ Stop", command=self.stop_running, state="disabled")
        self.stop_button.pack(side="left", padx=5)

        self.build_project_panel()

        splitter = ttk.Panedwindow(self, orient="vertical")
        splitter.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        upper = ttk.Panedwindow(splitter, orient="horizontal")
        splitter.add(upper, weight=3)

        list_frame = ttk.LabelFrame(upper, text="Saved Commands — double-click a row to edit", padding=8)
        upper.add(list_frame, weight=3)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        columns = ("order", "name", "comment")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("order", text="#", anchor="center")
        self.tree.heading("name", text="Command Name")
        self.tree.heading("comment", text="Comment / Notes")
        self.tree.column("order", width=48, minwidth=42, stretch=False, anchor="center")
        self.tree.column("name", width=190, minwidth=125)
        self.tree.column("comment", width=300, minwidth=120)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self.update_selected_details())
        self.tree.bind("<Double-1>", lambda _event: self.edit_command())

        details = ttk.LabelFrame(upper, text="Selected Command Details", padding=10)
        upper.add(details, weight=2)
        details.columnconfigure(0, weight=1)
        details.rowconfigure(6, weight=1)
        self.detail_title = ttk.Label(details, text="Select a command", font=("TkDefaultFont", 11, "bold"))
        self.detail_title.grid(row=0, column=0, sticky="w")
        self.detail_order = ttk.Label(details, text="Order: —")
        self.detail_order.grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.detail_cwd = ttk.Label(details, text="Working folder: (default)", wraplength=360)
        self.detail_cwd.grid(row=2, column=0, sticky="w", pady=(5, 8))
        ttk.Label(details, text="Comment / notes:").grid(row=3, column=0, sticky="w")
        self.detail_comment = tk.Text(details, height=4, wrap="word", state="disabled", relief="flat", background=self.cget("background"))
        self.detail_comment.grid(row=4, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(details, text="Command:").grid(row=5, column=0, sticky="nw")
        self.detail_command = tk.Text(details, height=8, wrap="word", state="disabled")
        self.detail_command.grid(row=6, column=0, sticky="nsew", pady=(2, 0))

        output_frame = ttk.LabelFrame(splitter, text="Run Output", padding=8)
        splitter.add(output_frame, weight=2)
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)
        self.output_text = tk.Text(output_frame, wrap="word", state="disabled", height=12, background="#101418", foreground="#e6edf3", insertbackground="#ffffff")
        self.output_text.grid(row=0, column=0, sticky="nsew")
        output_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        output_scroll.grid(row=0, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=output_scroll.set)
        output_buttons = ttk.Frame(output_frame)
        output_buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(output_buttons, text="Clear Output", command=self.clear_output).pack(side="left")
        ttk.Button(output_buttons, text="Open JSON Folder", command=self.open_config_folder).pack(side="left", padx=6)
        self.status_var = tk.StringVar(value=f"Ready. Settings file: {CONFIG_PATH.name}")
        ttk.Label(self, textvariable=self.status_var, anchor="w", relief="sunken", padding=(8, 4)).pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # Settings, project folders, placeholders
    # ------------------------------------------------------------------
    def init_settings(self) -> None:
        saved: dict[str, Any] = {}
        try:
            if SETTINGS_PATH.exists():
                loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    saved = loaded
        except Exception:
            saved = {}
        self.s: dict[str, tk.StringVar] = {
            key: tk.StringVar(value=str(saved.get(key, default))) for key, default in DEFAULT_SETTINGS.items()
        }
        for level in LEVELS:
            (ORG_DIR / level).mkdir(parents=True, exist_ok=True)
            (ANIM_DIR / level).mkdir(parents=True, exist_ok=True)
            (FBX_DIR / level).mkdir(parents=True, exist_ok=True)
            (MOD_DIR / level).mkdir(parents=True, exist_ok=True)
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        for var in self.s.values():
            var.trace_add("write", lambda *_: self.save_settings())

    def save_settings(self) -> None:
        try:
            data = {key: var.get() for key, var in self.s.items()}
            SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def extract_dir_for(level: str, db_name: str) -> Path:
        return ANIM_DIR / level / Path(db_name).stem

    def token_map(self) -> dict[str, str]:
        """Values for the {placeholders} that can be used in saved commands."""
        level = self.s["level"].get().strip()
        db = self.s["db"].get().strip()
        clip = self.s["clip"].get().strip()
        game = self.s["game_root"].get().strip()
        source_db = str(ORG_DIR / level / db) if level and db else ""
        clip_anim = str(self.extract_dir_for(level, db) / clip) if level and db and clip else ""
        game_dest = str(Path(game) / "levels" / level / "fpguns" / db) if game and level and db else ""
        clip_name = Path(clip).stem if clip else ""
        baked_json, edit_json = self.s["baked_json"].get().strip(), self.s["edit_json"].get().strip()
        pos_json, out_db, out_anim = self.s["pos_json"].get().strip(), self.s["out_db"].get().strip(), self.s["out_anim"].get().strip()
        clip_work, patch_source = "", source_db
        mod_dir = str(MOD_DIR / level / Path(db).stem) if level and db else ""
        if self.s["auto_paths"].get() == "1":
            baked_json = edit_json = pos_json = out_anim = out_db = ""
            if level and db:
                db_work = WORK_DIR / level / Path(db).stem
                out_db = str(db_work / db)                      # ONE modded db per original db, collects every clip edit
                if Path(out_db).is_file():
                    patch_source = out_db                       # keep earlier clip edits
                if clip_name:
                    clip_work = str(db_work / clip_name)
                    baked_json = str(Path(clip_work) / "reload_baked.json")
                    edit_json = str(Path(clip_work) / "reload_edit.json")
                    pos_json = str(Path(clip_work) / "reload_edit_position_only.json")
                    out_anim = str(Path(mod_dir) / Path(clip).name)     # same file name as the original clip, inside pfguns_anim_mod
        mapping = {
            "clip_work_dir": clip_work,
            "mod_anim_dir": mod_dir,
            "scratch_db": str((Path(clip_work) if clip_work else WORK_DIR) / "_scratch_patched.db"),
            "patch_source_db": patch_source,
            "app_dir": str(APP_DIR),
            "project_dir": str(PROJECT_DIR),
            "tools_dir": self.s["tools_dir"].get().strip(),
            "game_root": game,
            "level": level,
            "db_name": db,
            "db_stem": Path(db).stem if db else "",
            "source_db": source_db,
            "extract_dir": str(self.extract_dir_for(level, db)) if level and db else "",
            "fbx_path": str(FBX_DIR / level / (Path(db).stem + ".fbx")) if level and db else "",
            "fbx_dir": str(FBX_DIR / level) if level else "",
            "clip_anim": clip_anim,
            "clip_name": Path(clip).stem if clip else "",
            "baked_json": baked_json,
            "edit_json": edit_json,
            "pos_json": pos_json,
            "out_db": out_db,
            "out_anim": out_anim,
            "game_db_dest": game_dest,
        }
        return {k: os.path.normpath(v) if v and k not in ("level", "db_name", "db_stem", "clip_name") else v for k, v in mapping.items()}

    @staticmethod
    def resolve_text(text: str, mapping: dict[str, str]) -> tuple[str, set[str]]:
        missing: set[str] = set()

        def sub(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in mapping:
                return match.group(0)  # unknown braces are left alone
            if not mapping[key]:
                missing.add(key)
            return mapping[key]

        return TOKEN_RE.sub(sub, text), missing

    def prepare_and_run(self, commands: list[dict[str, Any]]) -> None:
        """Fill {placeholders} from the current selections, then run."""
        base = self.token_map()
        resolved: list[dict[str, Any]] = []
        problems: list[str] = []
        for command in commands:
            mapping = {**base, **command.get("_extra", {})}
            text, miss_a = self.resolve_text(command.get("command", ""), mapping)
            cwd, miss_b = self.resolve_text(command.get("working_directory", ""), mapping)
            missing = miss_a | miss_b
            if missing:
                problems.append(f"• {command.get('name', '?')}: {', '.join(sorted(missing))}")
            item = {k: v for k, v in command.items() if k != "_extra"}
            item["command"] = text
            item["working_directory"] = cwd
            resolved.append(item)
        for key in ("clip_work_dir", "mod_anim_dir"):
            if base.get(key):
                Path(base[key]).mkdir(parents=True, exist_ok=True)
        if problems:
            messagebox.showerror(
                "Missing selections / paths",
                "These values are empty. Pick them in the Modding Project panel first:\n\n" + "\n".join(problems),
                parent=self,
            )
            return
        self.start_run(resolved)

    # ------------------------------------------------------------------
    # Modding project panel
    # ------------------------------------------------------------------
    def build_project_panel(self) -> None:
        box = ttk.LabelFrame(self, text="Modding Project", padding=8)
        box.pack(fill="x", padx=10, pady=(0, 6))
        box.columnconfigure(1, weight=1)

        # Game folder + grab
        ttk.Label(box, text="Game folder:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(box, textvariable=self.s["game_root"]).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Button(box, text="Browse…", command=lambda: self.browse_path("game_root", "dir", "Game folder")).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(box, text="Grab .db / .wdd from game", command=self.grab_from_game).grid(row=0, column=3, padx=(6, 0))

        # Level / DB / Clip selectors
        sel = ttk.Frame(box)
        sel.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 4))
        sel.columnconfigure(5, weight=1)
        ttk.Label(sel, text="Level:").grid(row=0, column=0, sticky="w")
        self.level_combo = ttk.Combobox(sel, textvariable=self.s["level"], values=LEVELS, state="readonly", width=10)
        self.level_combo.grid(row=0, column=1, padx=(4, 12))
        self.level_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_db_list())
        ttk.Label(sel, text="DB:").grid(row=0, column=2, sticky="w")
        self.db_combo = ttk.Combobox(sel, textvariable=self.s["db"], values=[], state="readonly", width=24)
        self.db_combo.grid(row=0, column=3, padx=(4, 12))
        self.db_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_clip_list())
        ttk.Label(sel, text="Clip:").grid(row=0, column=4, sticky="w")
        self.clip_combo = ttk.Combobox(sel, textvariable=self.s["clip"], values=[], state="readonly")
        self.clip_combo.grid(row=0, column=5, sticky="ew", padx=(4, 0))
        self.clip_combo.bind("<<ComboboxSelected>>", self.update_work_label)

        actions = ttk.Frame(box)
        actions.grid(row=2, column=0, columnspan=4, sticky="w", pady=(0, 6))
        self.extract_button = ttk.Button(actions, text="Extract selected DB", command=self.extract_selected)
        self.extract_button.pack(side="left", padx=(0, 6))
        self.extract_all_button = ttk.Button(actions, text="Auto-extract ALL DBs", command=self.extract_all)
        self.extract_all_button.pack(side="left", padx=6)
        self.fbx_button = ttk.Button(actions, text="Export FBX (selected DB)", command=self.export_fbx_selected)
        self.fbx_button.pack(side="left", padx=6)
        self.fbx_all_button = ttk.Button(actions, text="Export ALL FBX", command=self.export_fbx_all)
        self.fbx_all_button.pack(side="left", padx=6)
        ttk.Button(actions, text="Refresh lists", command=self.refresh_db_list).pack(side="left", padx=6)
        ttk.Button(actions, text="Open extracted folder", command=self.open_extracted_folder).pack(side="left", padx=6)
        ttk.Button(actions, text="Open modded anims folder", command=self.open_mod_folder).pack(side="left", padx=6)
        ttk.Button(actions, text="Open FBX folder", command=self.open_fbx_folder).pack(side="left", padx=6)
        ttk.Button(actions, text="Open project folder", command=lambda: self.open_path(PROJECT_DIR)).pack(side="left", padx=6)

        self.path_row(box, 3, "Tools folder (scripts):", "tools_dir", "dir")
        self.path_row(box, 4, "Extract command:", "extract_template", None)
        self.path_row(box, 5, "FBX export command:", "fbx_template", None)
        self.path_row(box, 6, "reload_baked.json:", "baked_json", "file", [("JSON", "*.json"), ("All files", "*.*")])
        self.path_row(box, 7, "reload_edit.json:", "edit_json", "file", [("JSON", "*.json"), ("All files", "*.*")])
        self.path_row(box, 8, "reload_edit_position_only.json:", "pos_json", "file", [("JSON", "*.json"), ("All files", "*.*")])
        self.path_row(box, 9, "Modded DB (output):", "out_db", "file", [("DB", "*.db"), ("All files", "*.*")])
        self.path_row(box, 10, "Modded ANIM (output):", "out_anim", "file", [("ANIM", "*.anim"), ("All files", "*.*")])
        ttk.Checkbutton(box, text="Auto per-clip work folders (then the 5 path fields above are ignored)", variable=self.s["auto_paths"],
                        onvalue="1", offvalue="0", command=self.update_work_label).grid(row=11, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.work_label = tk.StringVar()
        ttk.Label(box, textvariable=self.work_label, foreground="#555555").grid(row=12, column=0, columnspan=4, sticky="w")

    def path_row(self, parent: ttk.Frame, row: int, label: str, key: str, mode: str | None, filetypes: list[tuple[str, str]] | None = None) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(parent, textvariable=self.s[key]).grid(row=row, column=1, sticky="ew", pady=2)
        if mode:
            ttk.Button(parent, text="Browse…", command=lambda: self.browse_path(key, mode, label.rstrip(":"), filetypes)).grid(row=row, column=2, padx=(6, 0), pady=2)

    def browse_path(self, key: str, mode: str, title: str, filetypes: list[tuple[str, str]] | None = None) -> None:
        current = self.s[key].get().strip()
        if mode == "dir":
            chosen = filedialog.askdirectory(parent=self, initialdir=current or str(APP_DIR), title=title)
        else:
            initial_dir = str(Path(current).parent) if current else str(WORK_DIR)
            if not Path(initial_dir).is_dir():
                initial_dir = str(WORK_DIR)
            # Save dialog without overwrite prompt: lets you pick an existing file OR type a new name.
            chosen = filedialog.asksaveasfilename(
                parent=self, title=title, initialdir=initial_dir,
                initialfile=Path(current).name if current else "",
                filetypes=filetypes or [("All files", "*.*")], confirmoverwrite=False,
            )
        if chosen:
            self.s[key].set(os.path.normpath(chosen))

    # ---- selectors ----
    def refresh_db_list(self) -> None:
        level = self.s["level"].get()
        folder = ORG_DIR / level
        dbs = sorted(p.name for p in folder.glob("*.db")) if folder.is_dir() else []
        self.db_combo.configure(values=dbs)
        if self.s["db"].get() not in dbs:
            self.s["db"].set(dbs[0] if dbs else "")
        self.refresh_clip_list()

    def update_work_label(self, *_: Any) -> None:
        m = self.token_map()
        if self.s["auto_paths"].get() == "1":
            mod_dir = Path(m["mod_anim_dir"]) if m["mod_anim_dir"] else None
            count = len(list(mod_dir.rglob("*.anim"))) if mod_dir and mod_dir.is_dir() else 0
            self.work_label.set(f"Work folder: {m['clip_work_dir'] or '(pick a clip)'}   |   modded anims in pfguns_anim_mod: {count}   |   modded DB: {m['out_db'] or '-'}")
        else:
            self.work_label.set("Manual paths in use.")

    def refresh_clip_list(self) -> None:
        level, db = self.s["level"].get(), self.s["db"].get()
        clips: list[str] = []
        if level and db:
            folder = self.extract_dir_for(level, db)
            if folder.is_dir():
                clips = sorted(str(p.relative_to(folder)) for p in folder.rglob("*.anim"))
        self.clip_combo.configure(values=clips)
        if self.s["clip"].get() not in clips:
            self.s["clip"].set(clips[0] if clips else "")
        self.update_work_label()

    # ---- grab from game ----
    @staticmethod
    def find_fpguns(root: Path) -> tuple[Path, list[tuple[str, Path]]]:
        """Return (game_root, [(level_name, fpguns_folder), ...]) for whatever folder the user picked."""
        level_re = re.compile(r"level_\d+", re.I)
        if root.name.lower() == "fpguns" and level_re.fullmatch(root.parent.name):
            return root.parent.parent.parent, [(root.parent.name, root)]
        if level_re.fullmatch(root.name) and (root / "fpguns").is_dir():
            return root.parent.parent, [(root.name, root / "fpguns")]
        if root.name.lower() == "levels":
            game_root, base = root.parent, root
        elif (root / "levels").is_dir():
            game_root, base = root, root / "levels"
        else:
            game_root, base = root, root
        found = []
        if base.is_dir():
            for child in sorted(base.iterdir()):
                if child.is_dir() and level_re.fullmatch(child.name) and (child / "fpguns").is_dir():
                    found.append((child.name, child / "fpguns"))
        return game_root, found

    def grab_from_game(self) -> None:
        if self.is_running:
            messagebox.showwarning("Busy", "Wait for the running command to finish.", parent=self)
            return
        start = self.s["game_root"].get().strip()
        chosen = filedialog.askdirectory(
            parent=self, initialdir=start if start and Path(start).is_dir() else "/",
            title="Choose the game folder (e.g. R:\\black\\game) or a level's fpguns folder",
        )
        if not chosen:
            return
        game_root, jobs = self.find_fpguns(Path(chosen))
        if not jobs:
            messagebox.showinfo("Nothing found", "No level_XX\\fpguns folders were found under:\n" + chosen, parent=self)
            return
        self.s["game_root"].set(os.path.normpath(str(game_root)))
        self.status_var.set("Copying .db / .wdd from game…")
        threading.Thread(target=self.grab_worker, args=(jobs,), daemon=True).start()

    def grab_worker(self, jobs: list[tuple[str, Path]]) -> None:
        put = self.output_queue.put
        put(("log", "\n" + "=" * 72 + "\nGrabbing .db / .wdd from game\n"))
        total = 0
        for level, fpguns in jobs:
            dest = ORG_DIR / level
            dest.mkdir(parents=True, exist_ok=True)
            files = sorted(f for f in fpguns.iterdir() if f.is_file() and f.suffix.lower() in (".db", ".wdd"))
            put(("log", f"{level}: {len(files)} file(s) from {fpguns}\n"))
            for f in files:
                try:
                    shutil.copy2(f, dest / f.name)
                    total += 1
                except Exception as exc:
                    put(("log", f"  ERROR copying {f.name}: {exc}\n"))
        put(("log", f"Done. Copied {total} file(s) into {ORG_DIR}\n" + "=" * 72 + "\n"))
        put(("refresh", None))

    # ---- extraction ----
    def extract_command_for(self, level: str, db_name: str) -> dict[str, Any]:
        out_dir = self.extract_dir_for(level, db_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "id": str(uuid.uuid4()),
            "name": f"extract {level}/{db_name}",
            "command": self.s["extract_template"].get().strip(),
            "comment": "",
            "working_directory": "{tools_dir}",
            "_extra": {"db": str(ORG_DIR / level / db_name), "out_dir": str(out_dir)},
        }

    def extract_selected(self) -> None:
        level, db = self.s["level"].get(), self.s["db"].get()
        if not level or not db:
            messagebox.showinfo("Select a DB", "Pick a level and a DB first (use 'Grab .db / .wdd from game' if the list is empty).", parent=self)
            return
        self.prepare_and_run([self.extract_command_for(level, db)])

    def extract_all(self) -> None:
        jobs = [(lvl, p.name) for lvl in LEVELS for p in sorted((ORG_DIR / lvl).glob("*.db"))]
        if not jobs:
            messagebox.showinfo("No DBs", "No .db files in Modding_project\\Org_pfguns\\levels. Grab them from the game first.", parent=self)
            return
        if not messagebox.askyesno("Auto-extract all", f"Extract {len(jobs)} DB file(s) from all levels?", parent=self):
            return
        self.prepare_and_run([self.extract_command_for(lvl, name) for lvl, name in jobs])

    def fbx_command_for(self, level: str, db_name: str) -> dict[str, Any]:
        out_fbx = FBX_DIR / level / (Path(db_name).stem + ".fbx")
        out_fbx.parent.mkdir(parents=True, exist_ok=True)
        return {
            "id": str(uuid.uuid4()),
            "name": f"fbx {level}/{db_name}",
            "command": self.s["fbx_template"].get().strip(),
            "comment": "",
            "working_directory": "{tools_dir}",
            "_extra": {"db": str(ORG_DIR / level / db_name), "out_fbx": str(out_fbx)},
        }

    def export_fbx_selected(self) -> None:
        level, db = self.s["level"].get(), self.s["db"].get()
        if not level or not db:
            messagebox.showinfo("Select a DB", "Pick a level and a DB first.", parent=self)
            return
        self.prepare_and_run([self.fbx_command_for(level, db)])

    def export_fbx_all(self) -> None:
        jobs = [(lvl, p.name) for lvl in LEVELS for p in sorted((ORG_DIR / lvl).glob("*.db"))]
        if not jobs:
            messagebox.showinfo("No DBs", "No .db files in Modding_project\\Org_pfguns\\levels. Grab them from the game first.", parent=self)
            return
        if not messagebox.askyesno("Export all FBX", f"Export FBX for {len(jobs)} DB file(s) from all levels?\nExisting FBX files are overwritten.", parent=self):
            return
        self.prepare_and_run([self.fbx_command_for(lvl, name) for lvl, name in jobs])

    # ---- folders ----
    @staticmethod
    def open_path(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Could not open folder", str(exc))

    def open_mod_folder(self) -> None:
        level, db = self.s["level"].get(), self.s["db"].get()
        self.open_path(MOD_DIR / level / Path(db).stem if level and db else MOD_DIR)

    def open_fbx_folder(self) -> None:
        level = self.s["level"].get()
        self.open_path(FBX_DIR / level if level else FBX_DIR)

    def open_extracted_folder(self) -> None:
        level, db = self.s["level"].get(), self.s["db"].get()
        self.open_path(self.extract_dir_for(level, db) if level and db else ANIM_DIR)

    def load_config(self) -> None:
        if not CONFIG_PATH.exists():
            self.commands = []
            return
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            raw_commands = data.get("commands", []) if isinstance(data, dict) else []
            cleaned: list[dict[str, Any]] = []
            for i, item in enumerate(raw_commands, start=1):
                if not isinstance(item, dict):
                    continue
                try:
                    order = max(1, int(item.get("order", i)))
                except (TypeError, ValueError):
                    order = i
                cleaned.append({
                    "id": str(item.get("id") or uuid.uuid4()),
                    "order": order,
                    "name": str(item.get("name", f"Command {i}")),
                    "command": str(item.get("command", "")),
                    "comment": str(item.get("comment", "")),
                    "working_directory": str(item.get("working_directory", "")),
                })
            self.commands = cleaned
        except Exception as exc:
            self.commands = []
            messagebox.showwarning(
                "Could not load settings",
                f"The settings file could not be read:\n{CONFIG_PATH}\n\n{exc}\n\nThe UI will open with an empty command list. The existing file has not been overwritten yet.",
            )

    def save_config(self, show_message: bool = False) -> bool:
        payload = {"version": 1, "commands": sorted(self.commands, key=lambda x: (int(x.get("order", 1)), x.get("name", "").lower()))}
        try:
            temporary = CONFIG_PATH.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(CONFIG_PATH)
            if show_message:
                self.status_var.set(f"Saved settings to {CONFIG_PATH}")
            return True
        except Exception as exc:
            messagebox.showerror("Save failed", f"Could not save settings:\n{CONFIG_PATH}\n\n{exc}", parent=self)
            return False

    def sorted_commands(self) -> list[dict[str, Any]]:
        return sorted(self.commands, key=lambda x: (int(x.get("order", 1)), x.get("name", "").lower()))

    def next_order(self) -> int:
        return max((int(c.get("order", 0)) for c in self.commands), default=0) + 1

    def refresh_command_list(self, select_id: str | None = None) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for command in self.sorted_commands():
            comment = " ".join(command.get("comment", "").split())
            self.tree.insert("", "end", iid=command["id"], values=(command["order"], command["name"], comment))
        if select_id and self.tree.exists(select_id):
            self.tree.selection_set(select_id)
            self.tree.focus(select_id)
            self.tree.see(select_id)
        elif self.tree.get_children():
            current = self.tree.selection()
            if not current:
                first = self.tree.get_children()[0]
                self.tree.selection_set(first)
        self.update_selected_details()

    def selected_command(self) -> dict[str, Any] | None:
        selection = self.tree.selection()
        if not selection:
            return None
        selected_id = selection[0]
        return next((c for c in self.commands if c["id"] == selected_id), None)

    def update_selected_details(self) -> None:
        command = self.selected_command()
        if not command:
            self.detail_title.configure(text="Select a command")
            self.detail_order.configure(text="Order: —")
            self.detail_cwd.configure(text="Working folder: (default)")
            self.set_readonly_text(self.detail_comment, "")
            self.set_readonly_text(self.detail_command, "")
            return
        self.detail_title.configure(text=command["name"])
        self.detail_order.configure(text=f"Order: {command['order']}")
        self.detail_cwd.configure(text=f"Working folder: {command['working_directory'] or '(default / this app folder)'}")
        self.set_readonly_text(self.detail_comment, command.get("comment", ""))
        self.set_readonly_text(self.detail_command, command.get("command", ""))

    @staticmethod
    def set_readonly_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def add_command(self) -> None:
        editor = CommandEditor(self)
        self.wait_window(editor)
        if editor.result:
            self.commands.append(editor.result)
            self.save_config()
            self.refresh_command_list(select_id=editor.result["id"])
            self.status_var.set(f"Added and saved: {editor.result['name']}")

    def edit_command(self) -> None:
        current = self.selected_command()
        if not current:
            messagebox.showinfo("Select a command", "Select a command to edit.", parent=self)
            return
        editor = CommandEditor(self, dict(current))
        self.wait_window(editor)
        if editor.result:
            for index, command in enumerate(self.commands):
                if command["id"] == editor.result["id"]:
                    self.commands[index] = editor.result
                    break
            self.save_config()
            self.refresh_command_list(select_id=editor.result["id"])
            self.status_var.set(f"Updated and saved: {editor.result['name']}")

    def delete_command(self) -> None:
        current = self.selected_command()
        if not current:
            messagebox.showinfo("Select a command", "Select a command to delete.", parent=self)
            return
        if not messagebox.askyesno("Delete command", f"Delete '{current['name']}'?", parent=self):
            return
        self.commands = [c for c in self.commands if c["id"] != current["id"]]
        self.save_config()
        self.refresh_command_list()
        self.status_var.set(f"Deleted: {current['name']}")

    def move_selected(self, direction: int) -> None:
        current = self.selected_command()
        if not current:
            messagebox.showinfo("Select a command", "Select a command to move.", parent=self)
            return
        ordered = self.sorted_commands()
        index = next((i for i, c in enumerate(ordered) if c["id"] == current["id"]), -1)
        other_index = index + direction
        if index < 0 or other_index < 0 or other_index >= len(ordered):
            return
        ordered[index], ordered[other_index] = ordered[other_index], ordered[index]
        for new_order, command in enumerate(ordered, start=1):
            command["order"] = new_order
        self.commands = ordered
        self.save_config()
        self.refresh_command_list(select_id=current["id"])
        self.status_var.set("Command order changed and saved.")

    def run_selected(self) -> None:
        command = self.selected_command()
        if not command:
            messagebox.showinfo("Select a command", "Select a command to run.", parent=self)
            return
        self.prepare_and_run([dict(command)])

    def run_all(self) -> None:
        commands = self.sorted_commands()
        if not commands:
            messagebox.showinfo("No commands", "Add a command first, then use Run All.", parent=self)
            return
        if not messagebox.askyesno("Run all commands", f"Run all {len(commands)} commands in numeric order?", parent=self):
            return
        self.prepare_and_run([dict(c) for c in commands])

    def start_run(self, commands: list[dict[str, Any]]) -> None:
        if self.is_running:
            messagebox.showwarning("Already running", "A command is already running. Stop it before starting another.", parent=self)
            return
        self.is_running = True
        self.stop_requested = False
        self.run_selected_button.configure(state="disabled")
        self.run_all_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.output_queue.put(("log", "\n" + "=" * 72 + "\n"))
        self.status_var.set("Running command(s)…")
        self.run_thread = threading.Thread(target=self.run_worker, args=(commands,), daemon=True)
        self.run_thread.start()

    def run_worker(self, commands: list[dict[str, Any]]) -> None:
        total = len(commands)
        outcome = "finished"
        try:
            for index, command in enumerate(commands, start=1):
                if self.stop_requested:
                    outcome = "stopped"
                    break
                name = command.get("name", f"Command {index}")
                command_line = command.get("command", "").strip()
                cwd = command.get("working_directory", "").strip() or str(APP_DIR)
                self.output_queue.put(("log", f"\n[{index}/{total}] {name}\nFolder: {cwd}\n> {command_line}\n"))
                if not command_line:
                    self.output_queue.put(("log", "Skipped: command text is empty.\n"))
                    continue
                if not Path(cwd).is_dir():
                    self.output_queue.put(("log", f"ERROR: working folder does not exist: {cwd}\n"))
                    outcome = "completed with errors"
                    continue
                try:
                    # shell=True supports Windows CMD built-ins and ordinary commands such as `python script.py`.
                    process = subprocess.Popen(
                        command_line,
                        cwd=cwd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        stdin=None,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    self.current_process = process
                    assert process.stdout is not None
                    for line in iter(process.stdout.readline, ""):
                        self.output_queue.put(("log", line))
                        if self.stop_requested and process.poll() is None:
                            self.terminate_process(process)
                    process.stdout.close()
                    return_code = process.wait()
                    self.current_process = None
                    self.output_queue.put(("log", f"[Exit code: {return_code}]\n"))
                    if return_code != 0:
                        outcome = "completed with errors"
                except Exception as exc:
                    self.current_process = None
                    self.output_queue.put(("log", f"ERROR while running command: {exc}\n"))
                    outcome = "completed with errors"
                if self.stop_requested:
                    outcome = "stopped"
                    break
        finally:
            self.current_process = None
            self.output_queue.put(("finished", outcome))

    @staticmethod
    def terminate_process(process: subprocess.Popen[str]) -> None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                process.terminate()
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def stop_running(self) -> None:
        if not self.is_running:
            return
        self.stop_requested = True
        process = self.current_process
        if process and process.poll() is None:
            self.terminate_process(process)
        self.status_var.set("Stopping command(s)…")
        self.output_queue.put(("log", "\nStop requested…\n"))

    def process_ui_queue(self) -> None:
        try:
            while True:
                kind, value = self.output_queue.get_nowait()
                if kind == "log":
                    self.append_output(str(value))
                elif kind == "refresh":
                    self.refresh_db_list()
                elif kind == "finished":
                    self.is_running = False
                    self.current_process = None
                    self.run_selected_button.configure(state="normal")
                    self.run_all_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status_var.set(f"Run {value}.")
                    self.refresh_clip_list()
                    self.append_output(f"\nRun {value}.\n" + "=" * 72 + "\n")
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self.process_ui_queue)

    def append_output(self, text: str) -> None:
        self.output_text.configure(state="normal")
        self.output_text.insert("end", text)
        self.output_text.see("end")
        self.output_text.configure(state="disabled")

    def clear_output(self) -> None:
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.configure(state="disabled")

    def open_config_folder(self) -> None:
        try:
            if os.name == "nt":
                os.startfile(str(APP_DIR))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(APP_DIR)])
            else:
                subprocess.Popen(["xdg-open", str(APP_DIR)])
        except Exception as exc:
            messagebox.showerror("Could not open folder", str(exc), parent=self)

    def on_close(self) -> None:
        if self.is_running:
            if not messagebox.askyesno("Command running", "A command is still running. Stop it and close the UI?", parent=self):
                return
            self.stop_running()
        self.save_config()
        self.destroy()


if __name__ == "__main__":
    app = CommandManager()
    app.mainloop()
