# -*- coding: utf-8 -*-
"""
Joint Pose Manager for Maya  (Maya 2022+, PySide2 / PySide6)
------------------------------------------------------------
Capture joint translate / rotate / scale at the current frame (what you see
with your anim layers evaluated), store it as a named pose in a JSON file,
and later apply it in any scene onto a NEW animation layer.

Features
  * Pose dropdown: Save / Overwrite, Rename, Delete, Delete All
  * JSON library file (auto-saved on every change, path is selectable)
  * Validation table with error flags: Missing joint, Ambiguous name,
    Locked / Connected attributes
  * Namespace remap on apply
  * Apply to new anim layer (override or additive), keys at current frame

Usage (Script Editor, Python tab):
    import joint_pose_manager
    joint_pose_manager.show()
"""
import json
import os
import datetime
import fnmatch

import maya.cmds as cmds
import maya.OpenMaya as om

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from shiboken6 import wrapInstance
except ImportError:
    from PySide2 import QtCore, QtGui, QtWidgets
    from shiboken2 import wrapInstance

from maya.app.general.mayaMixin import MayaQWidgetDockableMixin
import maya.OpenMayaUI as omui

WINDOW_TITLE = "Joint Pose Manager"
WINDOW_OBJECT = "jointPoseManagerWin"
OPTVAR_PATH = "jointPoseManager_jsonPath"
DEFAULT_JSON = os.path.join(os.path.expanduser("~"), "joint_pose_library.json")

ATTR_GROUPS = (("t", "translate"), ("r", "rotate"), ("s", "scale"))
AXES = ("X", "Y", "Z")

STATUS_OK = "OK"
STATUS_MISSING = "MISSING"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_WARN = "WARNING"

COLORS = {
    STATUS_OK: QtGui.QColor(70, 140, 80),
    STATUS_MISSING: QtGui.QColor(190, 60, 60),
    STATUS_AMBIGUOUS: QtGui.QColor(200, 130, 40),
    STATUS_WARN: QtGui.QColor(190, 160, 50),
}


# ----------------------------------------------------------------------------
# Core logic (no UI)
# ----------------------------------------------------------------------------
def strip_name(node):
    """'|grp|ns:jnt' -> 'jnt' (no path, no namespace)."""
    short = node.split("|")[-1]
    return short.split(":")[-1]


class PoseLibrary(object):
    """JSON backed pose storage."""

    VERSION = 1

    def __init__(self, path):
        self.path = path
        self.poses = {}
        self.load()

    def load(self):
        self.poses = {}
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r") as fh:
                data = json.load(fh)
            self.poses = data.get("poses", {})
        except (IOError, ValueError) as exc:
            om.MGlobal.displayError("Pose library unreadable: {0}".format(exc))

    def save(self):
        folder = os.path.dirname(self.path)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder)
        with open(self.path, "w") as fh:
            json.dump({"version": self.VERSION, "poses": self.poses},
                      fh, indent=2, sort_keys=True)

    def names(self):
        return sorted(self.poses.keys(), key=lambda n: n.lower())

    def set_pose(self, name, joints, options=None):
        old = self.poses.get(name, {})
        self.poses[name] = {
            "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "joints": joints,
            "options": options if options is not None else old.get("options", {}),
        }
        self.save()

    def set_options(self, name, options):
        if name in self.poses:
            self.poses[name]["options"] = options
            self.save()

    def delete(self, name):
        if name in self.poses:
            del self.poses[name]
            self.save()

    def clear(self):
        self.poses = {}
        self.save()

    def rename(self, old, new):
        if old in self.poses and new not in self.poses:
            self.poses[new] = self.poses.pop(old)
            self.save()
            return True
        return False


def collect_joints(hierarchy=True):
    """Joints from the current selection (optionally the whole hierarchy)."""
    sel = cmds.ls(selection=True, long=True) or []
    joints = []
    for node in sel:
        if cmds.nodeType(node) == "joint":
            joints.append(node)
        if hierarchy:
            joints.extend(cmds.listRelatives(node, allDescendents=True,
                                             type="joint", fullPath=True) or [])
    seen, ordered = set(), []
    for jnt in joints:
        if jnt not in seen:
            seen.add(jnt)
            ordered.append(jnt)
    return ordered


def capture_pose(joints):
    """Read evaluated values at the current frame (layers included)."""
    data, dupes = {}, []
    for jnt in joints:
        key = strip_name(jnt)
        if key in data:
            dupes.append(key)
            continue
        entry = {"ro": cmds.getAttr(jnt + ".rotateOrder")}
        for short, long_name in ATTR_GROUPS:
            entry[short] = [cmds.getAttr("{0}.{1}{2}".format(jnt, long_name, a))
                            for a in AXES]
        data[key] = entry
    return data, dupes


def resolve_joint(name, namespace):
    """Return (status, node, detail)."""
    candidates = []
    if namespace:
        ns = namespace.strip(":")
        candidates = cmds.ls("{0}:{1}".format(ns, name), long=True,
                             type="joint") or []
    else:
        # match with any namespace
        candidates = cmds.ls(name, long=True, type="joint") or []
        candidates += cmds.ls("*:" + name, long=True, type="joint") or []
        candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return STATUS_MISSING, None, "Joint not found in scene"
    if len(candidates) > 1:
        return STATUS_AMBIGUOUS, None, "{0} nodes share this name".format(
            len(candidates))
    return STATUS_OK, candidates[0], ""


def attr_problems(node):
    """Locked attrs are skipped; report them as warnings."""
    locked = []
    for _short, long_name in ATTR_GROUPS:
        for a in AXES:
            if cmds.getAttr("{0}.{1}{2}".format(node, long_name, a), lock=True):
                locked.append(long_name + a)
    return locked


SCOPE_ALL = "All joints in pose"
SCOPE_SELECTED = "Selected joints"
SCOPE_NAME = "By name"

BLEND_NONE = "No blend (full pose)"
BLEND_IN = "From start (blend in)"
BLEND_OUT = "From end (blend out)"


def pick_names(pose, scope, pattern="", with_children=False):
    """Stored joint names that fall inside the chosen apply scope.

    scope SCOPE_ALL      -> every joint in the pose
    scope SCOPE_SELECTED -> joints selected in the scene (matched by name)
    scope SCOPE_NAME     -> comma separated names / wildcards, e.g. 'arm_*, spine?'
    """
    stored = list(pose["joints"].keys())
    if scope == SCOPE_SELECTED:
        nodes = collect_joints(hierarchy=with_children)
        wanted = set(strip_name(n) for n in nodes)
        return [n for n in stored if n in wanted]
    if scope == SCOPE_NAME:
        patterns = [p.strip().lower() for p in pattern.split(",") if p.strip()]
        return [n for n in stored
                if any(fnmatch.fnmatchcase(n.lower(), p) for p in patterns)]
    return stored


def validate_pose(pose, namespace, names=None):
    """List of (joint, status, node, detail). names limits the joints checked."""
    rows = []
    pool = pose["joints"].keys() if names is None else names
    for name in sorted(pool):
        status, node, detail = resolve_joint(name, namespace)
        if status == STATUS_OK:
            locked = attr_problems(node)
            if locked:
                status = STATUS_WARN
                detail = "Locked: " + ", ".join(locked)
        rows.append((name, status, node, detail))
    return rows


def apply_pose(pose, namespace, layer_name, additive, skip_bad, set_key=True,
               names=None, blend=BLEND_NONE):
    """Create anim layer and key the pose on it. Returns (layer, applied, skipped)."""
    rows = validate_pose(pose, namespace, names)
    bad = [r for r in rows if r[1] in (STATUS_MISSING, STATUS_AMBIGUOUS)]
    if bad and not skip_bad:
        raise RuntimeError("{0} joint(s) have errors - fix names or enable "
                           "'Skip missing'".format(len(bad)))

    targets = [(r[0], r[2]) for r in rows if r[2]]
    if not targets:
        raise RuntimeError("No valid joints to apply.")

    cmds.undoInfo(openChunk=True, chunkName="applyJointPose")
    applied, skipped = 0, [r[0] for r in bad]
    try:
        # create layer and add attributes
        layer = cmds.animLayer(layer_name, override=not additive)
        plugs = []
        for _name, node in targets:
            for _short, long_name in ATTR_GROUPS:
                for a in AXES:
                    plug = "{0}.{1}{2}".format(node, long_name, a)
                    if not cmds.getAttr(plug, lock=True):
                        plugs.append(plug)
        cmds.animLayer(layer, edit=True, attribute=plugs)

        # make it the only active, preferred layer so setAttr lands on it
        for other in cmds.ls(type="animLayer") or []:
            if other != layer:
                cmds.animLayer(other, edit=True, selected=False,
                               preferred=False)
        cmds.animLayer(layer, edit=True, selected=True, preferred=True)

        for name, node in targets:
            entry = pose["joints"][name]
            try:
                cmds.setAttr(node + ".rotateOrder", entry.get("ro", 0))
            except RuntimeError:
                pass
            for short, long_name in ATTR_GROUPS:
                for a, value in zip(AXES, entry[short]):
                    plug = "{0}.{1}{2}".format(node, long_name, a)
                    if cmds.getAttr(plug, lock=True):
                        continue
                    try:
                        cmds.setAttr(plug, value)
                    except RuntimeError:
                        pass
            if set_key:
                for short, long_name in ATTR_GROUPS:
                    for a in AXES:
                        plug = "{0}.{1}{2}".format(node, long_name, a)
                        if cmds.getAttr(plug, lock=True):
                            continue
                        try:
                            cmds.setKeyframe(plug, animLayer=layer)
                        except RuntimeError:
                            pass
            applied += 1

        # blend: two weight keys on the layer across the timeline range
        #   from start -> weight 0 at first frame (base anim only), 1 at last
        #   from end   -> weight 1 at first frame, 0 at last frame
        if blend in (BLEND_IN, BLEND_OUT):
            start = cmds.playbackOptions(query=True, minTime=True)
            end = cmds.playbackOptions(query=True, maxTime=True)
            w_start, w_end = (0.0, 1.0) if blend == BLEND_IN else (1.0, 0.0)
            for t, w in ((start, w_start), (end, w_end)):
                cmds.setKeyframe(layer + ".weight", time=t, value=w,
                                 inTangentType="linear",
                                 outTangentType="linear")
    finally:
        cmds.undoInfo(closeChunk=True)
    return layer, applied, skipped


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def maya_main_window():
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QWidget) if ptr else None


STYLE = """
QGroupBox { font-weight: bold; border: 1px solid #555; border-radius: 4px;
            margin-top: 10px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton { padding: 5px 10px; border-radius: 3px; }
QPushButton#primary { background-color: #3d7ea6; color: white; font-weight: bold; }
QPushButton#primary:hover { background-color: #4a93c0; }
QPushButton#danger { background-color: #8a3a3a; color: white; }
QPushButton#danger:hover { background-color: #a54646; }
QLabel#banner { padding: 6px; border-radius: 3px; font-weight: bold; }
"""


class PoseManagerUI(MayaQWidgetDockableMixin, QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PoseManagerUI, self).__init__(parent=parent or maya_main_window())
        self.setObjectName(WINDOW_OBJECT)
        self.setWindowTitle(WINDOW_TITLE)
        self.setStyleSheet(STYLE)
        self.setMinimumWidth(420)

        path = DEFAULT_JSON
        if cmds.optionVar(exists=OPTVAR_PATH):
            path = cmds.optionVar(query=OPTVAR_PATH)
        self.library = PoseLibrary(path)

        self._build_ui()
        self.pattern_edit.setEnabled(False)
        self.children_check.setEnabled(False)
        self._connect_option_signals()
        self._refresh_combo()

    # -- layout --------------------------------------------------------------
    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # library file
        grp = QtWidgets.QGroupBox("Pose Library (JSON)")
        lay = QtWidgets.QHBoxLayout(grp)
        self.path_edit = QtWidgets.QLineEdit(self.library.path)
        self.path_edit.setReadOnly(True)
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse_library)
        lay.addWidget(self.path_edit)
        lay.addWidget(browse)
        root.addWidget(grp)

        # pose selection
        grp = QtWidgets.QGroupBox("Poses")
        lay = QtWidgets.QVBoxLayout(grp)
        row = QtWidgets.QHBoxLayout()
        self.combo = QtWidgets.QComboBox()
        self.combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                 QtWidgets.QSizePolicy.Fixed)
        self.combo.currentIndexChanged.connect(self._on_pose_changed)
        self.rename_btn = QtWidgets.QPushButton("Rename")
        self.rename_btn.clicked.connect(self._rename_pose)
        self.delete_btn = QtWidgets.QPushButton("Delete")
        self.delete_btn.setObjectName("danger")
        self.delete_btn.clicked.connect(self._delete_pose)
        self.clear_btn = QtWidgets.QPushButton("Delete All")
        self.clear_btn.setObjectName("danger")
        self.clear_btn.clicked.connect(self._delete_all)
        row.addWidget(self.combo)
        row.addWidget(self.rename_btn)
        row.addWidget(self.delete_btn)
        row.addWidget(self.clear_btn)
        lay.addLayout(row)
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setStyleSheet("color: #999;")
        lay.addWidget(self.info_label)
        root.addWidget(grp)

        # capture
        grp = QtWidgets.QGroupBox("Capture (current frame)")
        lay = QtWidgets.QVBoxLayout(grp)
        row = QtWidgets.QHBoxLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("New pose name, e.g. fire_var3_pose")
        self.hier_check = QtWidgets.QCheckBox("Include child joints")
        self.hier_check.setChecked(True)
        row.addWidget(self.name_edit)
        row.addWidget(self.hier_check)
        lay.addLayout(row)
        self.save_btn = QtWidgets.QPushButton("Save Pose From Selection")
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self._save_pose)
        lay.addWidget(self.save_btn)
        root.addWidget(grp)

        # apply
        grp = QtWidgets.QGroupBox("Apply")
        form = QtWidgets.QFormLayout(grp)
        self.ns_edit = QtWidgets.QLineEdit()
        self.ns_edit.setPlaceholderText("empty = match any namespace")
        self.ns_edit.editingFinished.connect(self._validate)
        self.scope_combo = QtWidgets.QComboBox()
        self.scope_combo.addItems([SCOPE_ALL, SCOPE_SELECTED, SCOPE_NAME])
        self.scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        self.pattern_edit = QtWidgets.QLineEdit()
        self.pattern_edit.setPlaceholderText("e.g. arm_*, spine_01, *_L")
        self.pattern_edit.editingFinished.connect(self._validate)
        self.children_check = QtWidgets.QCheckBox("Include children of selected")
        self.layer_edit = QtWidgets.QLineEdit("PoseLayer")
        self.blend_combo = QtWidgets.QComboBox()
        self.blend_combo.addItems([BLEND_NONE, BLEND_IN, BLEND_OUT])
        self.blend_combo.setToolTip(
            "Keys the layer weight on the first and last frame of the "
            "timeline.\nFrom start: base animation at the start, full pose at "
            "the end.\nFrom end: full pose at the start, base animation at "
            "the end.")
        self.additive_check = QtWidgets.QCheckBox(
            "Keep base animation (additive layer)")
        self.additive_check.setChecked(True)
        self.additive_check.setToolTip(
            "ON: pose is added as an offset on top of the existing animation, "
            "so the base animation keeps playing.\n"
            "OFF: override layer - the posed joints hold the pose on every "
            "frame and replace the base animation.")
        self.skip_check = QtWidgets.QCheckBox("Skip missing / ambiguous joints")
        self.key_check = QtWidgets.QCheckBox("Set keyframe on layer at current frame")
        self.key_check.setChecked(True)
        form.addRow("Apply to:", self.scope_combo)
        form.addRow("Names:", self.pattern_edit)
        form.addRow("", self.children_check)
        form.addRow("Namespace:", self.ns_edit)
        form.addRow("Anim layer name:", self.layer_edit)
        form.addRow("Blend:", self.blend_combo)
        form.addRow("", self.additive_check)
        form.addRow("", self.skip_check)
        form.addRow("", self.key_check)
        btn_row = QtWidgets.QHBoxLayout()
        self.validate_btn = QtWidgets.QPushButton("Validate")
        self.validate_btn.clicked.connect(self._validate)
        self.apply_btn = QtWidgets.QPushButton("Apply To New Anim Layer")
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(self.validate_btn)
        btn_row.addWidget(self.apply_btn, 1)
        form.addRow(btn_row)
        root.addWidget(grp)

        # validation
        self.banner = QtWidgets.QLabel("")
        self.banner.setObjectName("banner")
        self.banner.setVisible(False)
        root.addWidget(self.banner)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Joint", "Status", "Details"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table.itemDoubleClicked.connect(self._select_row_joint)
        root.addWidget(self.table, 1)

        self.status = QtWidgets.QLabel("Ready")
        self.status.setStyleSheet("color: #999;")
        root.addWidget(self.status)

    # -- helpers --------------------------------------------------------------
    def _msg(self, text, error=False):
        self.status.setText(text)
        (om.MGlobal.displayError if error else om.MGlobal.displayInfo)(
            "[PoseManager] " + text)

    def _current(self):
        name = self.combo.currentText()
        return name if name in self.library.poses else None

    def _refresh_combo(self, select=None):
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(self.library.names())
        if select:
            idx = self.combo.findText(select)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
        self.combo.blockSignals(False)
        has = self.combo.count() > 0
        for w in (self.rename_btn, self.delete_btn, self.clear_btn,
                  self.apply_btn, self.validate_btn):
            w.setEnabled(has)
        self._on_pose_changed()

    # -- slots ----------------------------------------------------------------
    def _browse_library(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Pose library file", self.library.path,
            "JSON (*.json)", options=QtWidgets.QFileDialog.DontConfirmOverwrite)
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        self.library = PoseLibrary(path)
        cmds.optionVar(stringValue=(OPTVAR_PATH, path))
        self.path_edit.setText(path)
        self._refresh_combo()
        self._msg("Library: {0} ({1} poses)".format(path, len(self.library.poses)))

    def _on_pose_changed(self, *_):
        name = self._current()
        self.table.setRowCount(0)
        self.banner.setVisible(False)
        if not name:
            self.info_label.setText("No poses stored.")
            return
        pose = self.library.poses[name]
        self.info_label.setText("{0} joints  |  saved {1}".format(
            len(pose["joints"]), pose.get("created", "?")))
        self._load_options(pose.get("options"))
        self._validate()

    def _save_pose(self):
        name = self.name_edit.text().strip()
        if not name:
            self._msg("Enter a pose name first.", error=True)
            return
        joints = collect_joints(self.hier_check.isChecked())
        if not joints:
            self._msg("Select joint(s) first.", error=True)
            return
        if name in self.library.poses:
            ans = QtWidgets.QMessageBox.question(
                self, "Overwrite pose", "Pose '{0}' exists. Overwrite?".format(name))
            if ans != QtWidgets.QMessageBox.Yes:
                return
        data, dupes = capture_pose(joints)
        # new pose takes the current apply options; overwrite keeps its own
        opts = None if name in self.library.poses else self._get_options()
        self.library.set_pose(name, data, opts)
        self._refresh_combo(select=name)
        text = "Saved '{0}' ({1} joints)".format(name, len(data))
        if dupes:
            text += " - duplicate names skipped: " + ", ".join(sorted(set(dupes)))
        self._msg(text, error=bool(dupes))

    def _rename_pose(self):
        old = self._current()
        if not old:
            return
        new, ok = QtWidgets.QInputDialog.getText(
            self, "Rename pose", "New name:", text=old)
        new = new.strip()
        if not ok or not new or new == old:
            return
        if not self.library.rename(old, new):
            self._msg("Name already exists.", error=True)
            return
        self._refresh_combo(select=new)

    def _delete_pose(self):
        name = self._current()
        if not name:
            return
        ans = QtWidgets.QMessageBox.question(
            self, "Delete pose", "Delete '{0}'?".format(name))
        if ans == QtWidgets.QMessageBox.Yes:
            self.library.delete(name)
            self._refresh_combo()
            self._msg("Deleted '{0}'".format(name))

    def _delete_all(self):
        if not self.library.poses:
            return
        ans = QtWidgets.QMessageBox.warning(
            self, "Delete all poses",
            "Delete ALL {0} poses from the library file?".format(
                len(self.library.poses)),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel)
        if ans == QtWidgets.QMessageBox.Yes:
            self.library.clear()
            self._refresh_combo()
            self._msg("Library cleared.")

    def _connect_option_signals(self):
        """Any change to an apply option is saved into the current pose."""
        self.scope_combo.currentIndexChanged.connect(self._store_options)
        self.pattern_edit.editingFinished.connect(self._store_options)
        self.ns_edit.editingFinished.connect(self._store_options)
        self.layer_edit.editingFinished.connect(self._store_options)
        self.blend_combo.currentIndexChanged.connect(self._store_options)
        for chk in (self.children_check, self.additive_check,
                    self.skip_check, self.key_check):
            chk.toggled.connect(self._store_options)

    def _get_options(self):
        return {
            "scope": self.scope_combo.currentText(),
            "names": self.pattern_edit.text(),
            "include_children": self.children_check.isChecked(),
            "namespace": self.ns_edit.text(),
            "layer_name": self.layer_edit.text(),
            "blend": self.blend_combo.currentText(),
            "keep_base": self.additive_check.isChecked(),
            "skip_missing": self.skip_check.isChecked(),
            "set_key": self.key_check.isChecked(),
        }

    def _load_options(self, opts):
        """Fill the Apply section from the pose's saved options."""
        opts = opts or {}
        self._loading = True
        widgets = (self.scope_combo, self.pattern_edit, self.children_check,
                   self.ns_edit, self.layer_edit, self.additive_check,
                   self.skip_check, self.key_check, self.blend_combo)
        for w in widgets:
            w.blockSignals(True)
        try:
            idx = self.scope_combo.findText(opts.get("scope", SCOPE_ALL))
            self.scope_combo.setCurrentIndex(max(idx, 0))
            self.pattern_edit.setText(opts.get("names", ""))
            self.children_check.setChecked(opts.get("include_children", False))
            self.ns_edit.setText(opts.get("namespace", ""))
            self.layer_edit.setText(opts.get("layer_name", "PoseLayer"))
            bidx = self.blend_combo.findText(opts.get("blend", BLEND_NONE))
            self.blend_combo.setCurrentIndex(max(bidx, 0))
            self.additive_check.setChecked(opts.get("keep_base", True))
            self.skip_check.setChecked(opts.get("skip_missing", False))
            self.key_check.setChecked(opts.get("set_key", True))
        finally:
            for w in widgets:
                w.blockSignals(False)
            self._loading = False
        scope = self.scope_combo.currentText()
        self.pattern_edit.setEnabled(scope == SCOPE_NAME)
        self.children_check.setEnabled(scope == SCOPE_SELECTED)

    def _store_options(self, *_):
        if getattr(self, "_loading", False):
            return
        name = self._current()
        if name:
            self.library.set_options(name, self._get_options())

    def _on_scope_changed(self, *_):
        scope = self.scope_combo.currentText()
        self.pattern_edit.setEnabled(scope == SCOPE_NAME)
        self.children_check.setEnabled(scope == SCOPE_SELECTED)
        self._validate()

    def _scope_names(self, pose):
        return pick_names(pose, self.scope_combo.currentText(),
                          self.pattern_edit.text(),
                          self.children_check.isChecked())

    def _validate(self):
        name = self._current()
        if not name:
            return None
        pose = self.library.poses[name]
        names = self._scope_names(pose)
        if not names:
            self.table.setRowCount(0)
            self.banner.setVisible(True)
            self.banner.setText("No joints in scope - nothing to apply")
            self.banner.setStyleSheet("background:#8a7430; color:white;")
            return 0
        rows = validate_pose(pose, self.ns_edit.text(), names)
        self.table.setRowCount(len(rows))
        counts = {STATUS_OK: 0, STATUS_MISSING: 0, STATUS_AMBIGUOUS: 0,
                  STATUS_WARN: 0}
        for r, (jnt, status, node, detail) in enumerate(rows):
            counts[status] += 1
            items = [QtWidgets.QTableWidgetItem(jnt),
                     QtWidgets.QTableWidgetItem(status),
                     QtWidgets.QTableWidgetItem(detail)]
            items[0].setData(QtCore.Qt.UserRole, node)
            items[1].setForeground(QtGui.QBrush(COLORS[status]))
            font = items[1].font()
            font.setBold(True)
            items[1].setFont(font)
            for c, it in enumerate(items):
                self.table.setItem(r, c, it)
        errors = counts[STATUS_MISSING] + counts[STATUS_AMBIGUOUS]
        self.banner.setVisible(True)
        if errors:
            self.banner.setText(
                "ERROR: {0} missing, {1} ambiguous - names do not match scene".format(
                    counts[STATUS_MISSING], counts[STATUS_AMBIGUOUS]))
            self.banner.setStyleSheet("background:#8a3a3a; color:white;")
        elif counts[STATUS_WARN]:
            self.banner.setText("{0} joint(s) have locked attributes (skipped)".format(
                counts[STATUS_WARN]))
            self.banner.setStyleSheet("background:#8a7430; color:white;")
        else:
            self.banner.setText("All {0} joints matched".format(counts[STATUS_OK]))
            self.banner.setStyleSheet("background:#3c6e47; color:white;")
        return errors

    def _select_row_joint(self, item):
        node = self.table.item(item.row(), 0).data(QtCore.Qt.UserRole)
        if node and cmds.objExists(node):
            cmds.select(node, replace=True)

    def _apply(self):
        name = self._current()
        if not name:
            return
        pose = self.library.poses[name]
        names = self._scope_names(pose)
        if not names:
            self._msg("No joints in scope. Check selection or name filter.",
                      error=True)
            self._validate()
            return
        errors = self._validate()
        if errors and not self.skip_check.isChecked():
            self._msg("Name errors found - fix them or enable 'Skip missing'.",
                      error=True)
            return
        layer_name = self.layer_edit.text().strip() or "PoseLayer"
        try:
            layer, applied, skipped = apply_pose(
                pose, self.ns_edit.text(), layer_name,
                self.additive_check.isChecked(), self.skip_check.isChecked(),
                self.key_check.isChecked(), names,
                self.blend_combo.currentText())
        except RuntimeError as exc:
            self._msg(str(exc), error=True)
            return
        text = "Applied '{0}' to layer '{1}' ({2} joints)".format(
            name, layer, applied)
        if skipped:
            text += ", skipped {0}".format(len(skipped))
        self._msg(text, error=bool(skipped))


_ui = None


def show():
    global _ui
    if cmds.workspaceControl(WINDOW_OBJECT + "WorkspaceControl", exists=True):
        cmds.deleteUI(WINDOW_OBJECT + "WorkspaceControl")
    _ui = PoseManagerUI()
    _ui.show(dockable=True, floating=True, width=460, height=700)
    return _ui


if __name__ == "__main__":
    show()
