"""black_anim_import_maya.py  -  apply a world-space baked Black (2006) clip (JSON from black_anim_bake_v5_2.py) to a joint rig in Maya.

HOW TO USE
 1. Import the rig FBX into a NEW scene (File > Import). Do not move or animate the joints: the script reads the current
    joint positions as the rest pose.  (Run it only once per fresh import, see ALLOW_REANIMATE below.)
 2. Optional: select the root joint (or its group) to limit the script to that skeleton.
 3. Script Editor > Python tab > open this file > edit JSON_PATH below > run it (Ctrl+Enter).

WHY IT WORKS (same idea as the Blender importer)
 The clip's skeleton is not the FBX skeleton (gun is parented to the right wrist, wrist/twist are siblings), so local channels cannot be
 copied 1:1.  The JSON holds each bone's WORLD matrix per frame.  Here we turn that into the delta from the rest pose, move it into
 Maya's space (fitted automatically from the rest joint positions, so units / axes / joint orients don't matter) and solve each joint's
 local translate + rotate from its parent.  Result: plain translate/rotate keys on the existing joints.

The maths (numpy only, no maya needed) is in the first half of the file and can be tested outside Maya.
"""
import json, math, sys
import numpy as np

JSON_PATH = r"S:/black_mod/reload_baked.json"   # <- change me
START_FRAME = 1
ALLOW_REANIMATE = False        # False: stop if the joints already have animation (rest pose would be wrong)
LINEAR_TANGENTS = True

# ------------------------------------------------------------------ pure-numpy part
def to4(row12):
    m = np.eye(4); m[:3] = np.asarray(row12, float).reshape(3, 4); return m

def umeyama(src, dst):
    """similarity transform (scale*R, t) that maps src points onto dst points"""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0); S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3); E[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ E @ Vt; scale = (sig * np.diag(E)).sum() / ((S ** 2).sum() / len(src))
    C = np.eye(4); C[:3, :3] = scale * R; C[:3, 3] = mu_d - scale * R @ mu_s
    return C, scale

def solve_local(bones_json, rest_world, parent_of, parent_static, C):
    """bones_json : JSON bone dicts (only those that exist in the Maya scene)
    rest_world    : {name: 4x4 column-vector world rest matrix of the Maya joint}
    parent_of     : {name: parent JOINT name (in this set) or None}
    parent_static : {name: 4x4 world matrix of the non-joint parent transform (identity if none)}  - used when parent_of[name] is None
    C             : JSON-space -> Maya-space similarity transform
    returns {name: (frames,4,4) LOCAL matrices (column-vector) relative to the joint's parent}  for every bone that has data"""
    Ci = np.linalg.inv(C); frames = len(bones_json[0]['world'])
    pose = {}
    for b in bones_json:
        n = b['name']; Gr_inv = np.linalg.inv(to4(b['rest_world']))
        pose[n] = [C @ (to4(w) @ Gr_inv) @ Ci @ rest_world[n] for w in b['world']]      # target WORLD pose in Maya space
    memo = {}
    def world_of(n, f):                                      # joints we have no data for just follow their parent
        if n in pose: return pose[n][f]
        if (n, f) in memo: return memo[(n, f)]
        p = parent_of.get(n)
        r = rest_world[n] if p is None else world_of(p, f) @ np.linalg.inv(rest_world[p]) @ rest_world[n]
        memo[(n, f)] = r; return r
    local = {}
    for n in pose:
        p = parent_of.get(n)
        pw = (lambda f: world_of(p, f)) if p else (lambda f, _m=parent_static.get(n, np.eye(4)): _m)
        local[n] = np.array([np.linalg.inv(pw(f)) @ pose[n][f] for f in range(frames)])
    return local

# ---- Euler helpers, Maya convention: ROW vectors, rotateOrder 'xyz' = rotate about X first, then Y, then Z  (M_row = Rx * Ry * Rz)
ORDERS = ['xyz', 'yzx', 'zxy', 'xzy', 'yxz', 'zyx']          # index == Maya's rotateOrder attribute value
def _axis_row(ax, a):
    c, s = math.cos(a), math.sin(a); M = np.eye(3)
    i, j = [(1, 2), (2, 0), (0, 1)][ax]                       # column-vector rotation about ax is [[c,-s],[s,c]] in (i,j); row = transpose
    M[i, i] = c; M[j, j] = c; M[i, j] = s; M[j, i] = -s; return M

def euler_to_row(angles, order):
    """angles in radians, indexed (x, y, z) -> 3x3 row-vector rotation matrix for the given rotate order"""
    M = np.eye(3)
    for ch in order:
        ax = 'xyz'.index(ch); M = M @ _axis_row(ax, angles[ax])
    return M

def row_to_euler(Mrow, order, prev=None):
    """3x3 row-vector rotation matrix -> (x, y, z) radians for the given rotate order; picks the solution closest to prev"""
    C = np.asarray(Mrow, float).T                              # column-vector form  C = Rk * Rj * Ri  (axis i is applied first)
    i, j, k = ['xyz'.index(ch) for ch in order]; s = 1 if (j - i) % 3 == 1 else -1
    b = math.asin(max(-1.0, min(1.0, -s * C[k, i])))
    if abs(C[k, i]) < 0.999999:
        a = math.atan2(s * C[k, j], C[k, k]); g = math.atan2(s * C[j, i], C[i, i])
    else:                                                      # gimbal lock
        a = math.atan2(-s * C[j, k], C[j, j]); g = 0.0
    best = None
    for ai, bj, gk in ((a, b, g), (a + math.pi, math.pi - b, g + math.pi)):
        sol = [0.0, 0.0, 0.0]; sol[i], sol[j], sol[k] = ai, bj, gk                 # now indexed x, y, z
        if prev is None: sol = [((x + math.pi) % (2 * math.pi)) - math.pi for x in sol]; cost = 0.0
        else: sol = [x + 2 * math.pi * round((p - x) / (2 * math.pi)) for x, p in zip(sol, prev)]; cost = sum(abs(x - p) for x, p in zip(sol, prev))
        if best is None or cost < best[0] - 1e-12: best = (cost, sol)
    return tuple(best[1])

def local_to_attrs(L_col, ra_row, jo_row, order, prev=None):
    """column-vector local matrix -> (translate xyz, rotate xyz in radians) for a Maya joint.
    Maya joint matrix (row vectors):  S * RotateAxis * Rotate * JointOrient * T   (scale assumed 1)"""
    Mr = np.asarray(L_col, float).T
    t = Mr[3, :3].copy(); Rr = Mr[:3, :3]
    Rr = Rr / np.linalg.norm(Rr, axis=1, keepdims=True)        # drop any inherited scale
    R = np.linalg.inv(ra_row) @ Rr @ np.linalg.inv(jo_row)
    return t, row_to_euler(R, order, prev)

# ------------------------------------------------------------------ Maya part
def _short(path): return path.split('|')[-1].split(':')[-1]

def run(json_path=None):
    import maya.cmds as cmds
    json_path = json_path or JSON_PATH
    data = json.load(open(json_path)); bones = data['bones']; frames = data['frames']; fps = int(round(data.get('fps', 30)))

    # ---- find joints
    sel = cmds.ls(selection=True, long=True)
    if sel:
        pool = set(cmds.listRelatives(sel, allDescendents=True, type='joint', fullPath=True) or []) | set(cmds.ls(sel, type='joint', long=True))
    else:
        pool = set(cmds.ls(type='joint', long=True))
    by_short = {}
    for j in sorted(pool): by_short.setdefault(_short(j), []).append(j)
    dup = {k: v for k, v in by_short.items() if len(v) > 1 and k in {b['name'] for b in bones}}
    if dup: raise RuntimeError('several joints share a name (select the root of ONE skeleton first): ' + ', '.join(dup))
    common = [b for b in bones if b['name'] in by_short]
    missing = [b['name'] for b in bones if b['name'] not in by_short]
    if missing: print('WARNING bones in JSON but not in the Maya scene:', missing)
    if len(common) < 3: raise RuntimeError('fewer than 3 matching joints - is the rig imported / selected?')
    path = {b['name']: by_short[b['name']][0] for b in common}
    in_set = {p: n for n, p in path.items()}

    animated = [n for n, p in path.items() if cmds.listConnections(p, type='animCurve', source=True, destination=False)]
    if animated and not ALLOW_REANIMATE:
        raise RuntimeError('joints already have animation (%d). Re-import the rig in a clean scene so the rest pose is correct, '
                           'or set ALLOW_REANIMATE = True if the current pose really is the rest pose.' % len(animated))

    # ---- rest pose, hierarchy
    def wm(node): return np.array(cmds.xform(node, q=True, ws=True, matrix=True), float).reshape(4, 4).T    # Maya returns row-vector layout
    rest_world = {n: wm(p) for n, p in path.items()}
    parent_of, parent_static = {}, {}
    for n, p in path.items():
        par = cmds.listRelatives(p, parent=True, fullPath=True)
        par = par[0] if par else None
        parent_of[n] = in_set.get(par) if par else None
        parent_static[n] = wm(par) if (par and par not in in_set) else np.eye(4)

    # ---- JSON space -> Maya space
    src = np.array([to4(b['rest_world'])[:3, 3] for b in common]); dst = np.array([rest_world[b['name']][:3, 3] for b in common])
    C, scale = umeyama(src, dst)
    err = np.linalg.norm((src @ C[:3, :3].T + C[:3, 3]) - dst, axis=1)
    print('JSON->Maya fit: scale=%.5f  max rest-position error=%.5f  (should be ~0)' % (scale, err.max()))
    if err.max() > 1e-2 * max(1.0, abs(scale) * 100):
        print('WARNING: poor fit - joint names/positions in the scene differ from the JSON skeleton (wrong rig, or joints not in rest pose)')

    local = solve_local(common, rest_world, parent_of, parent_static, C)

    # ---- scene settings
    tmap = {15: 'game', 24: 'film', 25: 'pal', 30: 'ntsc', 48: 'show', 50: 'palf', 60: 'ntscf'}
    if fps in tmap: cmds.currentUnit(time=tmap[fps], updateAnimation=False)
    else: print('WARNING: %d fps has no Maya preset - leaving the scene frame rate alone' % fps)
    cmds.playbackOptions(min=START_FRAME, max=START_FRAME + frames - 1, animationStartTime=START_FRAME, animationEndTime=START_FRAME + frames - 1)
    to_unit = 1.0 if cmds.currentUnit(q=True, angle=True) == 'rad' else 180.0 / math.pi

    cmds.undoInfo(openChunk=True)
    try:
        for n, mats in local.items():
            p = path[n]; order = ORDERS[cmds.getAttr(p + '.rotateOrder')]
            ra = euler_to_row(np.radians(cmds.getAttr(p + '.rotateAxis')[0]), 'xyz')
            jo = euler_to_row(np.radians(cmds.getAttr(p + '.jointOrient')[0]), 'xyz')
            prev = None
            for f in range(frames):
                t, prev = local_to_attrs(mats[f], ra, jo, order, prev)
                fr = START_FRAME + f
                for ax, v in zip('XYZ', t): cmds.setKeyframe(p, attribute='translate' + ax, time=fr, value=float(v))
                for ax, v in zip('XYZ', prev): cmds.setKeyframe(p, attribute='rotate' + ax, time=fr, value=float(v) * to_unit)
            if LINEAR_TANGENTS:
                try: cmds.keyTangent(p, edit=True, inTangentType='linear', outTangentType='linear')
                except Exception: pass
    finally:
        cmds.undoInfo(closeChunk=True)
    cmds.currentTime(START_FRAME)
    print('Done: %d joints x %d frames (clip "%s")' % (len(local), frames, data.get('clip', '?')))

if __name__ == '__main__':
    run(sys.argv[sys.argv.index('--') + 1] if '--' in sys.argv else None)
