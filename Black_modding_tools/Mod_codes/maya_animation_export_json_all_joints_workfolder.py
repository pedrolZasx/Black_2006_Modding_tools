"""black_anim_export_maya.py  -  bake the animation that is on the rig in Maya back to a baker-format JSON
(the exact format black_anim_bake_v5_2.py writes: clip, frames, fps, units, matrix, events, bones[name, parent, animated, rest_world, world]),
ready for black_anim_build.py.

HOW TO USE
 1. Edit SOURCE_JSON, OUT_JSON and REST_FBX below.
 2. Open the edited rig in Maya and run this file from Script Editor > Python.
 3. By default, this version scans ALL joints in the scene, not just the current selection.
    This prevents selecting Pistol_Slide (or a wrist) from causing all other bones to be copied unchanged.
 4. If the scene contains duplicate skeletons with repeated joint names, set SCENE_ROOT to the full DAG path
    of a group/transform containing the intended rig. The root must include every joint you want exported.
 The exported range is the Maya playback range (START_FRAME / END_FRAME override it).

WHY A REST FBX?   The importer fitted the JSON->Maya similarity transform C from the REST positions of the joints, and expressed every
 pose as a delta from the rest pose.  Your joints are keyed now, so their rest pose is gone from the scene.  The script re-imports REST_FBX
 into a throw-away namespace (BLKREST), reads the rest matrices, and deletes it again.  If you would rather not depend on that, run
 black_anim_export_maya.dump_rest(r"...\\rest.json") ONCE in a freshly imported, un-animated rig and set REST_JSON instead.

THE MATHS (inverse of the importer)
   importer:  pose_maya = C * (W_json * Gr^-1) * C^-1 * R_maya          (Gr = JSON rest_world, R_maya = rest world of the Maya joint)
   exporter:  W_json    = C^-1 * (M_maya * R_maya^-1) * C * Gr          (M_maya = joint world matrix at that frame)
 C is re-fitted from the same rest positions with the same Umeyama routine, so it is the importer's C.

WHAT YOU EDIT IN MAYA, WHAT IS EXPORTED
 Every joint's WORLD matrix is exported as it looks in the viewport; black_anim_build.py turns that into local tracks in the CLIP hierarchy.
 The Maya (FBX) hierarchy differs from the clip's: moving Wrist_rt does NOT move Pistol_Root in Maya.  If you move the wrist but leave the gun
 where it is, the clip gets the gun's track changed to compensate (it stays where it is in world space).  Key the joints you want to follow.
 Joints missing from the Maya scene keep their source data.  Scale on joints is ignored (rotation is taken from the nearest rotation matrix).
"""
import json, math, os, sys
import numpy as np

SOURCE_JSON = r"S:/black_mod/reload_baked.json"     # <- change me
OUT_JSON    = r"S:/black_mod/reload_edit.json"     # <- change me
REST_FBX    = r"S:/black_mod/pistol.fbx"             # <- change me (same rig FBX you imported)
REST_JSON   = ""                                                       # alternative to REST_FBX, written by dump_rest()
START_FRAME = None            # None = Maya playback range min / max
END_FRAME   = None
REST_NAMESPACE = "BLKREST"
CHANGED_TOL_CM = 1e-3         # only used for the "what did you change" printout
SCENE_ROOT = ""               # optional Maya DAG path; blank = scan every joint in the scene


# ------------------------------------------------------------------ pure-numpy part (testable outside Maya)
def to4(row12):
    m = np.eye(4); m[:3] = np.asarray(row12, float).reshape(3, 4); return m

def umeyama(src, dst):
    """similarity transform (scale*R, t) that maps src points onto dst points   (identical to black_anim_import_maya.umeyama)"""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0); S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3); E[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ E @ Vt; scale = (sig * np.diag(E)).sum() / ((S ** 2).sum() / len(src))
    C = np.eye(4); C[:3, :3] = scale * R; C[:3, 3] = mu_d - scale * R @ mu_s
    return C, scale

def fit_C(common_bones, rest_maya):
    """JSON space -> Maya space, from the rest positions of the joints present in both (same call as the importer)"""
    src = np.array([to4(b['rest_world'])[:3, 3] for b in common_bones]); dst = np.array([rest_maya[b['name']][:3, 3] for b in common_bones])
    C, scale = umeyama(src, dst)
    err = np.linalg.norm((src @ C[:3, :3].T + C[:3, 3]) - dst, axis=1)
    return C, scale, err

def maya_world_to_json_world(M, rest_maya, Gr, C):
    """M (F,4,4) Maya world matrices (column-vector) of one joint -> (F,4,4) JSON-space world matrices"""
    Ci = np.linalg.inv(C); Dm = np.asarray(M, float) @ np.linalg.inv(rest_maya)
    return Ci @ Dm @ C @ Gr

def build_json(src, maya_worlds, rest_maya, C, fps, log=print):
    """src: the source JSON dict; maya_worlds: {name: (F,4,4)}; returns the output dict (same schema as the baker's)"""
    F = len(next(iter(maya_worlds.values()))); F0 = src['frames']; out_b = []; changed = []; missing = []
    for b in src['bones']:
        n = b['name']; Gr = to4(b['rest_world'])
        if n in maya_worlds:
            W = maya_world_to_json_world(maya_worlds[n], rest_maya[n], Gr, C)
            if F == F0:
                dd = np.abs(W - np.array([to4(w) for w in b['world']]))
                if dd[:, :3, 3].max() > CHANGED_TOL_CM or dd[:, :3, :3].max() > 1e-4: changed.append((n, dd[:, :3, 3].max(), dd[:, :3, :3].max()))
            world = np.round(W[:, :3].reshape(F, 12), 5)
        else:
            missing.append(n); old = np.asarray(b['world'], float)
            idx = np.arange(F) if F == F0 else np.minimum(np.rint(np.arange(F) * (F0 - 1) / max(F - 1, 1)).astype(int), F0 - 1)
            world = old[idx]
        out_b.append(dict(name=n, parent=b['parent'], animated=b['animated'], **({'rec': b['rec']} if 'rec' in b else {}), rest_world=b['rest_world'], world=world.tolist()))   # 'rec' = record index (per-weapon order from the baker)
    out = dict(clip=src['clip'], frames=F, fps=fps, units=src.get('units', 'cm'), matrix=src['matrix'], events=src.get('events', []), bones=out_b)
    if 'event_table' in src: out['event_table'] = src['event_table']
    if 'records' in src: out['records'] = src['records']
    if missing: log('NOTE: bones not in the Maya scene keep their source data%s: %s' % ('' if F == F0 else ' (resampled by nearest frame)', ', '.join(missing)))
    return out, changed


# ------------------------------------------------------------------ Maya part
def _short(path): return path.split('|')[-1].split(':')[-1]
_TIME_UNITS = {'game': 15, 'film': 24, 'pal': 25, 'ntsc': 30, 'show': 48, 'palf': 50, 'ntscf': 60}

def _wm(cmds, node, time=None):
    m = cmds.getAttr(node + '.worldMatrix[0]', time=time) if time is not None else cmds.xform(node, q=True, ws=True, matrix=True)
    return np.array(m, float).reshape(4, 4).T                         # Maya returns the row-vector layout

def _scene_joints(cmds, root=None):
    """Return {short joint name: DAG path}.

    Default behavior deliberately ignores Maya's current selection and scans all
    scene joints. The older exporter restricted the scan to the selected node and
    its descendants; if Pistol_Slide was selected, hand/arm joints were then
    treated as missing and copied unchanged from SOURCE_JSON.

    Set SCENE_ROOT to a transform/group or joint DAG path to restrict the scan
    when the Maya scene contains more than one rig with repeated joint names.
    """
    root = (root or '').strip()
    if root:
        if not cmds.objExists(root):
            raise RuntimeError('SCENE_ROOT does not exist: ' + root)
        pool = set(cmds.listRelatives(root, allDescendents=True, type='joint', fullPath=True) or [])
        if cmds.nodeType(root) == 'joint':
            pool.add((cmds.ls(root, long=True) or [root])[0])
    else:
        pool = set(cmds.ls(type='joint', long=True) or [])
    by_short = {}
    for j in sorted(pool):
        by_short.setdefault(_short(j), []).append(j)
    dup = [k for k, v in by_short.items() if len(v) > 1]
    if dup:
        sample = ', '.join(dup[:20])
        if len(dup) > 20: sample += ', ...'
        raise RuntimeError(
            'Multiple scene joints share short names: ' + sample + '. '
            'Set SCENE_ROOT to a group containing the intended rig, or remove/close duplicate rigs.'
        )
    return {k: v[0] for k, v in by_short.items()}

def rest_from_fbx(cmds, fbx, wanted=None):
    """rest world matrices of a freshly imported copy of the rig, then removed again.
    The new joints are found by DIFFING the scene before / after the import (the FBX importer's return value and namespace are unreliable;
    a trailing number Maya adds on a name clash is stripped).  Two import routes are tried: the FBX plugin's own FBXImport command, then cmds.file."""
    if not os.path.isfile(fbx): raise RuntimeError('REST_FBX does not exist: ' + fbx)
    try: cmds.loadPlugin('fbxmaya', quiet=True)
    except Exception: pass
    before_j = set(cmds.ls(type='joint', long=True) or []); before_a = set(cmds.ls(assemblies=True, long=True) or [])
    new_j = []; errs = []
    def attempt(name, fn):
        try: fn()
        except Exception as e: errs.append('%s: %s' % (name, e))
        return [j for j in (cmds.ls(type='joint', long=True) or []) if j not in before_j]
    def via_mel():
        import maya.mel as mel
        mel.eval('FBXImportMode -v add; FBXImport -f "%s";' % fbx.replace('\\', '/'))
    def via_file():
        cmds.file(fbx, i=True, type='FBX', ignoreVersion=True, mergeNamespacesOnClash=False, namespace=REST_NAMESPACE, options='fbx', pr=True, importTimeRange='combine')
    for name, fn in (('FBXImport', via_mel), ('cmds.file', via_file)):
        new_j = attempt(name, fn)
        if new_j: break
    try:
        rest = {}
        for j in new_j:
            n = _short(j)
            if wanted and n not in wanted and n.rstrip('0123456789') in wanted: n = n.rstrip('0123456789')
            rest[n] = _wm(cmds, j)
    finally:
        for t in set(cmds.ls(assemblies=True, long=True) or []) - before_a:
            try:
                if cmds.objExists(t): cmds.delete(t)
            except Exception: pass
        try: cmds.namespace(removeNamespace=REST_NAMESPACE, deleteNamespaceContent=True)
        except Exception: pass
    if not rest:
        raise RuntimeError('importing %s added no joints (scene had %d joints before). %s Use dump_rest() + REST_JSON instead (see the header).'
                           % (fbx, len(before_j), ('Import errors: ' + ' | '.join(errs) + '.') if errs else 'No import error was raised.'))
    print('rest pose read from a temporary import of %s (%d joints)' % (os.path.basename(fbx), len(rest)))
    return rest

def dump_rest(path=None):
    """run ONCE on a freshly imported, un-animated rig (before the importer): stores the rest world matrices for REST_JSON"""
    import maya.cmds as cmds
    path = path or REST_JSON or (os.path.splitext(SOURCE_JSON)[0] + '_rest.json')
    rest = {n: _wm(cmds, p).tolist() for n, p in _scene_joints(cmds).items()}
    json.dump(rest, open(path, 'w')); print('rest pose of %d joints -> %s' % (len(rest), path))

def run(source_json=None, out_json=None):
    import maya.cmds as cmds
    source_json = source_json or SOURCE_JSON; out_json = out_json or OUT_JSON
    src = json.load(open(source_json)); bones = src['bones']
    # rest pose FIRST (the temporary import must not disturb the joint search below)
    if REST_JSON: rest = {k: np.array(v, float) for k, v in json.load(open(REST_JSON)).items()}
    elif REST_FBX: rest = rest_from_fbx(cmds, REST_FBX, {b['name'] for b in bones})
    else: raise RuntimeError('set REST_FBX (or REST_JSON written by dump_rest())')
    path = _scene_joints(cmds, SCENE_ROOT)
    print('Maya joint discovery: %d unique joint names (%s)' % (len(path), 'SCENE_ROOT=' + SCENE_ROOT if SCENE_ROOT else 'all joints in scene; current selection ignored'))
    common = [b for b in bones if b['name'] in path and b['name'] in rest]
    absent_scene = [b['name'] for b in bones if b['name'] not in path]
    absent_rest = [b['name'] for b in bones if b['name'] in path and b['name'] not in rest]
    print('Matched joints available for export: %d / %d JSON bones' % (len(common), len(bones)))
    if absent_scene: print('WARNING JSON bones not found in the Maya scene (will retain source tracks):', absent_scene)
    if absent_rest: print('WARNING joints found in Maya but missing from the rest reference (will retain source tracks):', absent_rest)
    if len(common) < 3: raise RuntimeError('fewer than 3 matching joints - is the rig in the scene / selected?')
    C, scale, err = fit_C(common, rest)
    print('JSON->Maya fit: scale=%.5f  max rest-position error=%.5f  (should be ~0)' % (scale, err.max()))
    if err.max() > 1e-2 * max(1.0, abs(scale) * 100): print('WARNING: poor fit - REST_FBX is not the rig you imported/animated (or the scene was scaled / moved)')
    t0 = int(round(cmds.playbackOptions(q=True, min=True))) if START_FRAME is None else int(START_FRAME)
    t1 = int(round(cmds.playbackOptions(q=True, max=True))) if END_FRAME is None else int(END_FRAME)
    unit = cmds.currentUnit(q=True, time=True); fps = _TIME_UNITS.get(unit, src.get('fps', 30))
    if unit not in _TIME_UNITS: print('WARNING: time unit "%s" has no preset: keeping fps=%s from the source JSON' % (unit, fps))
    frames = list(range(t0, t1 + 1))
    worlds = {}
    for b in common:
        n = b['name']
        try: worlds[n] = np.array([_wm(cmds, path[n], time=f) for f in frames])
        except Exception:                                              # getAttr(time=) unavailable: walk the timeline instead
            cur = cmds.currentTime(q=True); arr = []
            for f in frames: cmds.currentTime(f); arr.append(_wm(cmds, path[n]))
            cmds.currentTime(cur); worlds[n] = np.array(arr)
    out, changed = build_json(src, worlds, rest, C, fps)
    json.dump(out, open(out_json, 'w'))
    print('exported %d bones x %d frames (Maya %d..%d, %s fps)  ->  %s' % (len(out['bones']), len(frames), t0, t1, fps, out_json))
    if len(frames) == src['frames']:
        print('bones that differ from the source clip (%d):' % len(changed))
        for n, dt, dr in changed: print('   %-22s max move %.4f cm, max rotation-entry diff %.5f' % (n, dt, dr))
        if len(changed) == len(common): print('NOTE: EVERY joint differs from the source - if you did not edit everything, REST_FBX / scene scale does not match the import')
    else: print('NOTE: frame count changed %d -> %d: the encoder will re-encode every channel' % (src['frames'], len(frames)))
    return out

if __name__ == '__main__':
    run()
