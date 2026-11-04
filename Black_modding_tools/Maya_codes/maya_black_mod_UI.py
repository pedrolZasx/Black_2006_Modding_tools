# -*- coding: utf-8 -*-
"""maya_black_modding_ui.py - Maya front-end for the Black modding project.

Reads ALL paths / the current Level-DB-Clip from the command manager's settings.json and runs
the pipeline steps straight from the command manager's commands.json, so both tools always agree.

HOW TO USE (Maya 2022+, Python 3)
  Script Editor > Python:   exec(open(r"<folder>/maya_black_modding_ui.py", encoding="utf-8").read())
  or put this file on Maya's script path and run:   import maya_black_modding_ui as m; m.show()

It needs these next to each other (or in your tools folder), and loads them fresh on every click:
  maya_animation_import_json.py
  maya_animation_export_json_all_joints_workfolder.py
FBX export calls black_fps_extract_fbx_maya_v5.py through the "fbx_template" command in settings.json
(system Python, like the command manager does).
"""
import importlib
import importlib.util
import json
import os
import re
import subprocess
import traceback

import maya.cmds as cmds
import maya.mel as mel
# ---------------------------------------------------------------- EDIT IF YOUR LOCATIONS CHANGE
SETTINGS_JSON = r"C:/Users/workhard/Documents/ChatGPT/sub/__Release_source/settings.json"
MAYA_SCRIPTS_DIR = ""          # optional extra folder to look for the two Maya scripts
EXPORT_SCRIPT = "maya_animation_export_json_all_joints_workfolder.py"
IMPORT_SCRIPT = "maya_animation_import_json.py"
MESH_EXPORT_SCRIPT = "maya_mesh_export.py"
# ids of the steps inside commands.json (ids survive renaming / reordering in the command manager)
STEP_IDS = {
    "bake": "3c9b80be-6aa6-48f0-9d9b-694dccc8c21e",       # 1  db > anim to json
    "pos_json": "a2787e9a-a1aa-46df-b910-99f237883f95",   # 3  create only pos json
    "create_db": "6c79e530-ad88-4183-b00c-f0ed5b6f57c3",  # 4  patch clip > modded anim
    "build_db": "9d3f4c6e-5b0a-4f8e-8e51-2c7a1b6d9f10",   # 5  build db from pfguns_anim_mod
    "move_db": "f5074120-dbae-4588-bc92-ba242e902f61",    # 6  copy db to game
}
DEFAULT_FBX_TEMPLATE = 'python black_fps_extract_fbx_maya_v5.py "{db}" "{out_fbx}"'
# ----------------------------------------------------------------

LEVELS = ["level_%02d" % i for i in range(9)]
WINDOW = "blackModdingUI"
NONE_LABEL = "(none)"
TOKEN_RE = re.compile(r"\{(\w+)\}")
W = {}   # widget names


# ================================================================ settings / paths
def app_dir():
    if os.path.isfile(SETTINGS_JSON):
        return os.path.dirname(os.path.abspath(SETTINGS_JSON))
    # Keep this checked-out pipeline usable when the optional command-manager settings
    # file is not installed beside the user's original modding project.
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def project_dir():
    return os.path.join(app_dir(), "Modding_project")

ORG = lambda: os.path.join(project_dir(), "Org_pfguns", "levels")
ANIM = lambda: os.path.join(project_dir(), "org_pfguns_anim_extracted", "levels")
FBX = lambda: os.path.join(project_dir(), "org_pfguns_fbx_extracted", "levels")

def read_settings():
    try:
        with open(SETTINGS_JSON, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log("ERROR reading %s: %s" % (SETTINGS_JSON, exc))
        return {}

def list_dbs(level):
    folder = os.path.join(ORG(), level)
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".db")) if os.path.isdir(folder) else []

def list_clips(level, db):
    if not level or not db:
        return []
    folder = os.path.join(ANIM(), level, os.path.splitext(db)[0])
    out = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".anim"):
                out.append(os.path.relpath(os.path.join(root, f), folder))
    return sorted(out)

def fbx_path(level, db):
    return os.path.join(FBX(), level, os.path.splitext(db)[0] + ".fbx") if level and db else ""

def current_selection():
    def val(key):
        v = cmds.optionMenu(W[key], q=True, value=True) if W.get(key) else ""
        return "" if v == NONE_LABEL else v
    return val("level"), val("db"), val("clip")

def token_map():
    s = read_settings()
    level, db, clip = current_selection()
    game = str(s.get("game_root", "")).strip()
    extract_dir = os.path.join(ANIM(), level, os.path.splitext(db)[0]) if level and db else ""
    clip_name = os.path.splitext(os.path.basename(clip))[0] if clip else ""
    baked, edit = str(s.get("baked_json", "")).strip(), str(s.get("edit_json", "")).strip()
    pos, out_db, out_anim = str(s.get("pos_json", "")).strip(), str(s.get("out_db", "")).strip(), str(s.get("out_anim", "")).strip()
    source_db = os.path.join(ORG(), level, db) if level and db else ""
    clip_work, patch_source = "", source_db
    mod_dir = os.path.join(project_dir(), "pfguns_anim_mod", "levels", level, os.path.splitext(db)[0]) if level and db else ""
    if str(s.get("auto_paths", "1")) == "1":      # same rules as the command manager
        baked = edit = pos = out_anim = out_db = ""
        if level and db:
            db_work = os.path.join(project_dir(), "work", level, os.path.splitext(db)[0])
            out_db = os.path.join(db_work, db)
            if os.path.isfile(os.path.normpath(out_db)):
                patch_source = out_db
            if clip_name:
                clip_work = os.path.join(db_work, clip_name)
                baked = os.path.join(clip_work, "reload_baked.json")
                edit = os.path.join(clip_work, "reload_edit.json")
                pos = os.path.join(clip_work, "reload_edit_position_only.json")
                out_anim = os.path.join(mod_dir, os.path.basename(clip))
    m = {
        "clip_work_dir": clip_work, "patch_source_db": patch_source, "mod_anim_dir": mod_dir,
        "scratch_db": os.path.join(clip_work or os.path.join(project_dir(), "work"), "_scratch_patched.db"),
        "app_dir": app_dir(), "project_dir": project_dir(),
        "tools_dir": str(s.get("tools_dir", "")).strip(), "game_root": game,
        "level": level, "db_name": db, "db_stem": os.path.splitext(db)[0] if db else "",
        "source_db": os.path.join(ORG(), level, db) if level and db else "",
        "extract_dir": extract_dir,
        "fbx_path": fbx_path(level, db), "fbx_dir": os.path.join(FBX(), level) if level else "",
        "clip_anim": os.path.join(extract_dir, clip) if extract_dir and clip else "",
        "clip_name": os.path.splitext(os.path.basename(clip))[0] if clip else "",
        "baked_json": baked, "edit_json": edit, "pos_json": pos, "out_db": out_db, "out_anim": out_anim,
        "game_db_dest": os.path.join(game, "levels", level, "fpguns", db) if game and level and db else "",
    }
    plain = ("level", "db_name", "db_stem", "clip_name")
    return {k: (os.path.normpath(v) if v and k not in plain else v) for k, v in m.items()}

def resolve_text(text, mapping):
    missing = set()
    def sub(match):
        key = match.group(1)
        if key not in mapping:
            return match.group(0)
        if not mapping[key]:
            missing.add(key)
        return mapping[key]
    return TOKEN_RE.sub(sub, text), missing


# ================================================================ logging / running
def log(msg):
    print(msg)
    try:
        if W.get("log") and cmds.scrollField(W["log"], exists=True):
            cmds.scrollField(W["log"], e=True, appendText=str(msg) + "\n")
    except Exception:
        pass

def safe(fn):
    def wrapper(*_args):
        try:
            return fn()
        except Exception:
            log(traceback.format_exc())
            cmds.warning("%s failed - see the log in the Black Modding window" % fn.__name__)
    wrapper.__name__ = fn.__name__
    return wrapper

def run_shell(name, command, cwd, extra=None):
    """resolve {placeholders}, run through the shell, stream output to the log. returns True on exit code 0"""
    mapping = dict(token_map()); mapping.update(extra or {})
    for key in ("clip_work_dir", "mod_anim_dir"):
        if mapping.get(key):
            os.makedirs(mapping[key], exist_ok=True)
    text, miss_a = resolve_text(command, mapping)
    cwd, miss_b = resolve_text(cwd or "", mapping)
    missing = miss_a | miss_b
    if missing:
        log("[%s] MISSING values: %s  (pick them in the command manager / the dropdowns here)" % (name, ", ".join(sorted(missing))))
        return False
    cwd = cwd or app_dir()
    if not os.path.isdir(cwd):
        log("[%s] ERROR working folder does not exist: %s" % (name, cwd))
        return False
    log("\n[%s]\nfolder: %s\n> %s" % (name, cwd, text))
    env = dict(os.environ)
    for var in ("PYTHONHOME", "PYTHONPATH"):   # Maya's values would break the system python we start
        env.pop(var, None)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    cmds.waitCursor(state=True)
    try:
        proc = subprocess.Popen(text, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, encoding="utf-8", errors="replace", env=env, creationflags=flags)
        for line in proc.stdout:
            log(line.rstrip())
        code = proc.wait()
    finally:
        cmds.waitCursor(state=False)
    log("[exit code %s]" % code)
    return code == 0

def run_step(key):
    """run one saved command from the command manager's commands.json, with the current selections"""
    path = os.path.join(app_dir(), "commands.json")
    with open(path, "r", encoding="utf-8") as fh:
        commands = json.load(fh).get("commands", [])
    cmd = next((c for c in commands if c.get("id") == STEP_IDS[key]), None)
    if not cmd:
        raise RuntimeError("Step '%s' (id %s) is not in %s - was it deleted in the command manager?" % (key, STEP_IDS[key], path))
    return run_shell(cmd.get("name", key), cmd.get("command", ""), cmd.get("working_directory", ""))

def need(**fields):
    """make sure the named token values are non-empty, else raise a readable error"""
    m = token_map()
    empty = [k for k in fields if not m.get(k)]
    if empty:
        raise RuntimeError("Missing: %s" % ", ".join(empty))
    return m

def load_module(filename):
    dirs = []
    try:
        dirs.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    dirs += [MAYA_SCRIPTS_DIR, str(read_settings().get("tools_dir", "")), app_dir(), os.path.join(app_dir(), "maya"),
             os.path.join(app_dir(), "maya_codes"), os.path.join(app_dir(), "Mod_codes")]
    for d in dirs:
        full = os.path.join(d, filename) if d else ""
        if full and os.path.isfile(full):
            spec = importlib.util.spec_from_file_location("blk_" + os.path.splitext(filename)[0], full)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)          # fresh copy every click, so edits to the scripts take effect
            log("loaded %s" % full)
            return mod
    try:
        return importlib.import_module(os.path.splitext(filename)[0])
    except ImportError:
        raise RuntimeError("Could not find %s. Looked in: %s" % (filename, ", ".join(d for d in dirs if d)))

def flag(key):
    return bool(cmds.checkBox(W[key], q=True, value=True))


# ================================================================ FBX
def run_fbx_export(level, db):
    out = fbx_path(level, db)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    template = str(read_settings().get("fbx_template") or DEFAULT_FBX_TEMPLATE)
    ok = run_shell("fbx %s/%s" % (level, db), template, "{tools_dir}",
                   {"db": os.path.join(ORG(), level, db), "out_fbx": out})
    return ok and os.path.isfile(out)

@safe
def export_fbx_selected():
    level, db, _ = current_selection()
    if not level or not db:
        raise RuntimeError("Pick a level and a DB first.")
    log("FBX export %s" % ("OK -> " + fbx_path(level, db) if run_fbx_export(level, db) else "FAILED"))
    update_path_fields()

@safe
def export_fbx_all():
    skip = flag("skip_existing")
    jobs = [(lvl, db) for lvl in LEVELS for db in list_dbs(lvl)]
    if not jobs:
        raise RuntimeError("No .db files in %s - grab them from the game in the command manager first." % ORG())
    done = failed = skipped = 0
    for lvl, db in jobs:
        if skip and os.path.isfile(fbx_path(lvl, db)):
            skipped += 1
            continue
        if run_fbx_export(lvl, db): done += 1
        else: failed += 1
    log("\nFBX batch finished: %d exported, %d skipped, %d failed (of %d)" % (done, skipped, failed, len(jobs)))

@safe
def load_fbx():
    level, db, _ = current_selection()
    path = fbx_path(level, db)
    if not path or not os.path.isfile(path):
        raise RuntimeError("FBX not found: %s\nUse 'Export FBX (selected DB)' first." % path)
    if flag("new_scene_first"):
        if cmds.confirmDialog(title="New scene", message="Start a NEW scene before loading the FBX?\nUnsaved changes will be lost.",
                              button=["Yes, new scene", "Cancel"], defaultButton="Cancel", cancelButton="Cancel", dismissString="Cancel") != "Yes, new scene":
            return
        cmds.file(new=True, force=True)
    try:
        cmds.loadPlugin("fbxmaya", quiet=True)
    except Exception:
        pass
    mel.eval('FBXImportMode -v add; FBXImport -f "%s";' % path.replace("\\", "/"))
    log("FBX loaded: %s" % path)


# ================================================================ skinned mesh replacement
def _mesh_work_paths():
    level, db, _clip = current_selection()
    if not level or not db:
        raise RuntimeError("Pick a level and DB first.")
    folder = os.path.join(project_dir(), "work", level, os.path.splitext(db)[0], "mesh_replacement")
    stem = os.path.splitext(db)[0]
    return {"folder": folder, "obj": os.path.join(folder, "replacement.obj"),
            "weights": os.path.join(folder, "replacement.weights.json"),
            "out": os.path.join(folder, stem + "_mesh.db")}


@safe
def export_skinned_mesh():
    paths = _mesh_work_paths()
    os.makedirs(paths["folder"], exist_ok=True)
    mod = load_module(MESH_EXPORT_SCRIPT)
    nverts, njoints = mod.export_selected(paths["obj"], paths["weights"])
    log("Exported selected skinned mesh: %s vertices, %s influences\nOBJ: %s\nWeights: %s" %
        (nverts, njoints, paths["obj"], paths["weights"]))


@safe
def build_mesh_replacement():
    m = need(source_db=1)
    paths = _mesh_work_paths()
    if not os.path.isfile(paths["obj"]) or not os.path.isfile(paths["weights"]):
        raise RuntimeError("Export a selected, skinned replacement mesh from Maya first.")
    # If the animation pipeline already built its output DB, preserve those animation edits.
    if m.get("out_db") and os.path.isfile(m["out_db"]):
        base = m["out_db"]
    elif os.path.isfile(paths["out"]):
        base = paths["out"]
    else:
        base = m["source_db"]
    out_path = m["out_db"] if base == m.get("out_db") and m.get("out_db") else paths["out"]
    temp_path = out_path + ".mesh_tmp"
    if os.path.exists(temp_path):
        os.remove(temp_path)
    tool_candidates = []
    for name in ("black_fps_model_replace_v7.py", "black_fps_model_replace_v6.py"):   # v7 first: fixes object-relative pointers
        tool_candidates += [os.path.join(app_dir(), "Mod_codes", name),
                            os.path.join(str(read_settings().get("tools_dir", "")), name)]
    tool = next((p for p in tool_candidates if p and os.path.isfile(p)), "")
    if not tool:
        raise RuntimeError("Could not locate black_fps_model_replace_v7.py (or v6)")
    log("Using exporter: %s" % tool)
    group = cmds.intField(W["mesh_group"], query=True, value=True)
    scale = cmds.floatField(W["mesh_scale"], query=True, value=True)
    args = [tool, base, paths["obj"], temp_path, "--group", ("auto" if group < 0 else str(group)), "--scale", str(scale), "--weights-json", paths["weights"]]
    if tool.endswith("_v7.py"):
        args += ["--obj-base", "auto"]      # the tool reads the weapon object offset (e.g. 0x12A80) out of the db itself
    command = "python " + " ".join(subprocess.list2cmdline([arg]) for arg in args)
    ok = run_shell("Build mesh replacement", command, app_dir())
    if not ok or not os.path.isfile(temp_path):
        raise RuntimeError("Mesh replacement failed; see the log above. Existing DB was left untouched.")
    os.replace(temp_path, out_path)
    log("Mesh replacement DB ready: %s" % out_path)


# ================================================================ animation json
@safe
def import_json():
    m = need(baked_json=1)
    if not os.path.isfile(m["baked_json"]):
        raise RuntimeError("Baked JSON not found: %s\nRun '1. Bake clip -> JSON' first." % m["baked_json"])
    mod = load_module(IMPORT_SCRIPT)
    mod.ALLOW_REANIMATE = flag("allow_reanimate")
    mod.run(m["baked_json"])

@safe
def export_json():
    m = need(baked_json=1, edit_json=1, fbx_path=1)
    if not os.path.isfile(m["baked_json"]):
        raise RuntimeError("Baked JSON (source) not found: %s" % m["baked_json"])
    if not os.path.isfile(m["fbx_path"]):
        raise RuntimeError("Rest FBX not found: %s\nExport the FBX for this DB first." % m["fbx_path"])
    mod = load_module(EXPORT_SCRIPT)
    mod.REST_FBX = m["fbx_path"]
    mod.REST_JSON = ""
    mod.SCENE_ROOT = cmds.textField(W["scene_root"], q=True, text=True).strip()
    os.makedirs(os.path.dirname(m["edit_json"]), exist_ok=True)
    mod.run(m["baked_json"], m["edit_json"])


# ================================================================ pipeline steps (from commands.json)
@safe
def step_bake():
    need(clip_anim=1, source_db=1, baked_json=1)
    log("Step 1 OK" if run_step("bake") else "Step 1 FAILED")

@safe
def step_pos_json():
    need(baked_json=1, edit_json=1, pos_json=1)
    log("Step 3 OK" if run_step("pos_json") else "Step 3 FAILED")

@safe
def step_create_db():
    need(edit_json=1, baked_json=1, clip_anim=1, source_db=1, out_anim=1)
    log("Step 4 OK" if run_step("create_db") else "Step 4 FAILED")

@safe
def step_build_db():
    need(source_db=1, extract_dir=1, mod_anim_dir=1, out_db=1)
    log("Step 5 OK" if run_step("build_db") else "Step 5 FAILED")

@safe
def step_move_db():
    m = need(out_db=1, game_db_dest=1)
    if not os.path.isfile(m["out_db"]):
        raise RuntimeError("Modded DB not found: %s\nRun 'Create DB' first." % m["out_db"])
    if cmds.confirmDialog(title="Copy DB to game", message="Copy\n%s\n\nto the game folder (overwrites the existing file):\n%s" % (m["out_db"], m["game_db_dest"]),
                          button=["Copy", "Cancel"], defaultButton="Cancel", cancelButton="Cancel", dismissString="Cancel") != "Copy":
        return
    log("Step 6 OK" if run_step("move_db") else "Step 6 FAILED")


# ================================================================ UI
def _fill_menu(menu, items, current):
    for old in cmds.optionMenu(menu, q=True, itemListLong=True) or []:
        cmds.deleteUI(old)
    for item in (items or [NONE_LABEL]):
        cmds.menuItem(label=item, parent=menu)
    cmds.optionMenu(menu, e=True, value=current if current in items else (items[0] if items else NONE_LABEL))

def on_level(*_):
    level = current_selection()[0]
    dbs = list_dbs(level)
    _fill_menu(W["db"], dbs, current_selection()[1])
    on_db()

def on_db(*_):
    level, db, _clip = current_selection()
    _fill_menu(W["clip"], list_clips(level, db), _clip)
    update_path_fields()

def update_path_fields(*_):
    m = token_map()
    for key in ("baked_json", "edit_json", "pos_json", "out_db", "out_anim", "fbx_path", "clip_anim"):
        if W.get("p_" + key):
            cmds.textField(W["p_" + key], e=True, text=m.get(key, ""))

def reload_settings(*_):
    s = read_settings()
    level = s.get("level") if s.get("level") in LEVELS else LEVELS[0]
    cmds.optionMenu(W["level"], e=True, value=level)
    _fill_menu(W["db"], list_dbs(level), s.get("db", ""))
    _fill_menu(W["clip"], list_clips(level, current_selection()[1]), s.get("clip", ""))
    update_path_fields()
    log("settings reloaded from %s" % SETTINGS_JSON)

def _button(label, fn, annotation=""):
    return cmds.button(label=label, height=28, command=lambda *_: fn(), annotation=annotation)

def show():
    if cmds.window(WINDOW, exists=True):
        cmds.deleteUI(WINDOW)
    cmds.window(WINDOW, title="Black Modding", widthHeight=(640, 900), sizeable=True)
    cmds.scrollLayout(childResizable=True)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=6, columnOffset=("both", 8))

    cmds.frameLayout(label="Selection (initial values come from settings.json)", collapsable=True, marginWidth=8, marginHeight=6)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=4)
    W["level"] = cmds.optionMenu(label="Level  ", changeCommand=on_level)
    for lvl in LEVELS:
        cmds.menuItem(label=lvl)
    W["db"] = cmds.optionMenu(label="DB     ", changeCommand=on_db)
    cmds.menuItem(label=NONE_LABEL)
    W["clip"] = cmds.optionMenu(label="Clip   ", changeCommand=update_path_fields)
    cmds.menuItem(label=NONE_LABEL)
    _button("Reload settings.json", reload_settings, "Re-read paths and selection saved by the command manager")
    cmds.setParent("..")
    cmds.setParent("..")

    cmds.frameLayout(label="A. FBX", collapsable=True, marginWidth=8, marginHeight=6)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=4)
    _button("Export FBX (selected DB)", export_fbx_selected)
    _button("Export ALL FBX", export_fbx_all)
    W["skip_existing"] = cmds.checkBox(label="Skip FBX files that already exist (Export ALL)", value=True)
    _button("Load FBX into Maya", load_fbx, "Imports the FBX of the selected DB")
    W["new_scene_first"] = cmds.checkBox(label="Start a new scene before loading", value=True)
    cmds.setParent("..")
    cmds.setParent("..")

    cmds.frameLayout(label="A2. Skinned mesh replacement", collapsable=True, marginWidth=8, marginHeight=6)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=4)
    cmds.text(label="Select one replacement polygon mesh in its bind/rest pose. Skin it to joints from this DB skeleton first.", align="left")
    _button("Export selected mesh + Maya skin weights", export_skinned_mesh)
    cmds.rowLayout(numberOfColumns=2, adjustableColumn=2, columnAttach=[(1, "left", 0), (2, "both", 4)])
    cmds.text(label="Target group index (-1 = auto: the group that is not the arms)")
    W["mesh_group"] = cmds.intField(value=-1, minValue=-1)
    cmds.setParent("..")
    cmds.rowLayout(numberOfColumns=2, adjustableColumn=2, columnAttach=[(1, "left", 0), (2, "both", 4)])
    cmds.text(label="Maya units to game units scale (FBX centimeters -> game meters)")
    W["mesh_scale"] = cmds.floatField(value=0.01, precision=5, minValue=0.00001)
    cmds.setParent("..")
    _button("Build mesh replacement DB", build_mesh_replacement,
            "Uses the animation-built DB if present; otherwise starts from the selected source DB")
    cmds.setParent("..")
    cmds.setParent("..")

    cmds.frameLayout(label="B. Animation JSON", collapsable=True, marginWidth=8, marginHeight=6)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=4)
    _button("1. Bake clip -> baked JSON (command manager step 1)", step_bake)
    _button("Import baked JSON -> Maya", import_json, "Keys the rig in the scene from reload_baked.json")
    W["allow_reanimate"] = cmds.checkBox(label="Allow re-animate (joints already have keys)", value=False)
    _button("Export edited animation <- Maya (to reload_edit.json)", export_json, "Uses the selected DB's FBX as the rest pose")
    cmds.rowLayout(numberOfColumns=2, adjustableColumn=2, columnAttach=[(1, "left", 0), (2, "both", 4)])
    cmds.text(label="Scene root (optional)")
    W["scene_root"] = cmds.textField(placeholderText="only needed if the scene has duplicate rigs")
    cmds.setParent("..")
    cmds.setParent("..")
    cmds.setParent("..")

    cmds.frameLayout(label="C. Build and deploy (command manager steps 3, 4, 5, 6)", collapsable=True, marginWidth=8, marginHeight=6)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=4)
    _button("3. Create position-only JSON", step_pos_json)
    _button("4. Patch clip -> modded anim (pfguns_anim_mod)", step_create_db)
    _button("5. Build DB from pfguns_anim_mod (all modded clips)", step_build_db)
    _button("6. Copy DB to game", step_move_db, "Copies the modded db over the game's db after confirmation")
    cmds.setParent("..")
    cmds.setParent("..")

    cmds.frameLayout(label="Paths in use (edit them in the command manager)", collapsable=True, collapse=True, marginWidth=8, marginHeight=6)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=2)
    for key, label in (("clip_anim", "Clip anim"), ("fbx_path", "FBX"), ("baked_json", "Baked JSON"), ("edit_json", "Edited JSON"),
                       ("pos_json", "Position-only JSON"), ("out_db", "Modded DB"), ("out_anim", "Modded ANIM")):
        cmds.text(label=label, align="left")
        W["p_" + key] = cmds.textField(editable=False)
    cmds.setParent("..")
    cmds.setParent("..")

    cmds.frameLayout(label="Log", collapsable=True, marginWidth=8, marginHeight=6)
    W["log"] = cmds.scrollField(editable=False, wordWrap=True, height=220)
    cmds.setParent("..")

    cmds.showWindow(WINDOW)
    if not os.path.isfile(SETTINGS_JSON):
        log("WARNING settings.json not found: %s (edit SETTINGS_JSON at the top of this script)" % SETTINGS_JSON)
    reload_settings()


if __name__ == "__main__":
    show()
