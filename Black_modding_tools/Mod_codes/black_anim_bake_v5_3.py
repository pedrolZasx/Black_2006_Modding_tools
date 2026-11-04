#!/usr/bin/env python3
"""
black_anim_bake_v5.py - universal baker: decode ANY Black (2006) first-person .anim clip (AK, pistol, ...) and bake world-space
bone matrices for black_anim_import_blender.py.   Needs only numpy.

  python black_anim_bake_v5.py clip.anim chunk01_skeleton.bin clip_baked.json
  python black_anim_bake_v5.py clip.anim pistol.fbx           clip_baked.json      (an FBX rig works too)
  options:  --fps 30   --parent Child=Parent   --map RECORD=BoneName   --list (just print the record->bone table)

The skeleton can be the weapon's chunk01_skeleton.bin / bg1_xxx.db (preferred: exact bind pose) or an FBX whose bone names match.
No weapon-specific tables: the clip's record order is the level-order (BFS) of the clip hierarchy, so records are matched to bones
by walking that order and picking, among the bones whose parent is already assigned, the one whose rest offset equals the record's
translation.  A clip with 42 records and the AK bone names keeps using the verified AK table.
Rest values for 'no key' rotation (no track / type-17 segments) and for missing translations come from the bind pose, relative to the
CLIP hierarchy (weapon root hangs off Wrist_rt) - that is what keeps the gun from flipping.
"""
import argparse, json, math, os, re, struct, sys
import numpy as np

# ============================================================ decoder (reverse-engineered)
def f24(b): return struct.unpack('<f', b'\0' + bytes(b))[0]
def al4(x): return (x + 3) & ~3

SIZES = {1: lambda n: 4, 2: lambda n: 16, 3: lambda n: 28, 4: lambda n: 40, 5: lambda n: 52, 6: lambda n: 4 + 12*n,
         7: lambda n: al4(28 + 3*n), 8: lambda n: al4(28 + 6*n), 9: lambda n: 4, 10: lambda n: 16, 11: lambda n: 28,
         12: lambda n: 40, 13: lambda n: 52, 15: lambda n: al4(28 + 4*n), 16: lambda n: al4(28 + 8*n), 17: lambda n: 4}

def split_tracks(d):
    p = 0x30; out = []
    while p + 4 <= len(d) and d[p+2] == 0 and d[p+3] == 0x7f:
        ch, n = d[p], d[p+1]
        if ch not in SIZES: break
        s = SIZES[ch](n); out.append(dict(off=p, ch=ch, n=n, size=s)); p += s
    return out, p

def quat_norm(q):
    q = np.asarray(q, float); return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)

def eval_rot(d, t):
    p, ch, n, s = t['off'], t['ch'], t['n'], t['size']; b = d[p+4:p+s]
    if ch == 17: return np.full((n, 4), np.nan)                 # 'no key' -> rest rotation (filled later)
    if ch == 10:
        q = np.array([f24(b[i*3:i*3+3]) for i in range(4)]); return quat_norm(np.tile(q, (n, 1)))
    if ch in (11, 12, 13):
        k = {11: 2, 12: 3, 13: 4}[ch]
        c = np.array([f24(b[i*3:i*3+3]) for i in range(4*k)]).reshape(k, 4)      # highest power first, c0 last
        u = np.linspace(0, 1, n)[:, None]
        return quat_norm(sum(c[k-1-i] * u**i for i in range(k)))
    if ch in (15, 16):
        h = [f24(b[i*3:i*3+3]) for i in range(8)]; mn, sc = np.array(h[:4]), np.array(h[4:])
        k = (np.frombuffer(b[24:24+4*n], np.uint8) if ch == 15 else np.frombuffer(b[24:24+8*n], '<u2')).reshape(n, 4)
        return quat_norm(mn + k.astype(float)*sc)
    raise ValueError(f'rotation channel {ch}')

def eval_pos(d, t):
    p, ch, n, s = t['off'], t['ch'], t['n'], t['size']; b = d[p+4:p+s]
    if ch in (3, 4, 5):             # polynomials in FRAME INDEX, float32, highest power first: 3 linear, 4 quadratic, 5 cubic
        k = {3: 2, 4: 3, 5: 4}[ch]; c = np.frombuffer(b[:12*k], '<f4').astype(float).reshape(k, 3); tt = np.arange(n, dtype=float)[:, None]
        return sum(c[k-1-i] * tt**i for i in range(k))
    if ch == 2: return np.tile(np.frombuffer(b[:12], '<f4').astype(float), (n, 1))
    if ch == 6: return np.frombuffer(b[:12*n], '<f4').reshape(n, 3).astype(float)
    if ch in (7, 8):
        h = np.frombuffer(b[:24], '<f4').astype(float)
        k = np.frombuffer(b[24:24+3*n], np.uint8).reshape(n, 3) if ch == 7 else np.frombuffer(b[24:24+6*n], '<u2').reshape(n, 3)
        return h[:3] + k*h[3:]
    raise ValueError(f'position channel {ch}')

def join(segs, total):
    out = [segs[0]]
    for s in segs[1:]: out.append(s[1:])                         # segments overlap by one frame
    r = np.concatenate(out)
    if len(r) != total: raise ValueError(f'segments add up to {len(r)} frames, header says {total}')
    return r

def parse_bones(d, frames=None):
    tr, tail = split_tracks(d)
    frames = frames or struct.unpack_from('<I', d, 0x14)[0]
    if tail + 4 <= len(d) and d[tail+2] == 0 and d[tail+3] == 0x7f:
        raise ValueError(f'unknown track channel {d[tail]} (n={d[tail+1]}) at offset 0x{tail:x} - decoder needs a new channel type')
    i = 0; bones = []
    def segs(i):
        if tr[i]['ch'] == 1: k = tr[i]['n']; return tr[i+1:i+1+k], i+1+k
        return [tr[i]], i+1
    while i < len(tr):
        if tr[i]['ch'] == 17: rot = None; i += 1
        else: ss, i = segs(i); rot = join([eval_rot(d, s) for s in ss], frames)
        if tr[i]['ch'] == 9: pos = None; i += 1
        else: ss, i = segs(i); pos = join([eval_pos(d, s) for s in ss], frames)
        bones.append(dict(rot=rot, pos=pos))
    return bones, tail

def trailer_events(d, tail):
    return [s.decode('latin1') for s in re.findall(rb'[A-Za-z_][A-Za-z0-9_:]{3,}', d[tail:])]

# ============================================================ skeleton sources
def quat_to_mat3(q):
    q = np.asarray(q, float); x, y, z, w = (q / np.linalg.norm(q, axis=-1, keepdims=True)).T
    M = np.empty(q.shape[:-1] + (3, 3))
    M[..., 0, 0] = 1-2*(y*y+z*z); M[..., 0, 1] = 2*(x*y-z*w);   M[..., 0, 2] = 2*(x*z+y*w)
    M[..., 1, 0] = 2*(x*y+z*w);   M[..., 1, 1] = 1-2*(x*x+z*z); M[..., 1, 2] = 2*(y*z-x*w)
    M[..., 2, 0] = 2*(x*z-y*w);   M[..., 2, 1] = 2*(y*z+x*w);   M[..., 2, 2] = 1-2*(x*x+y*y)
    return M

def eul_xyz_mat3(rx, ry, rz):                                    # FBX default order: R = Rz*Ry*Rx
    rx, ry, rz = np.radians([rx, ry, rz]); c, s = np.cos, np.sin
    Rx = np.array([[1, 0, 0], [0, c(rx), -s(rx)], [0, s(rx), c(rx)]]); Ry = np.array([[c(ry), 0, s(ry)], [0, 1, 0], [-s(ry), 0, c(ry)]])
    Rz = np.array([[c(rz), -s(rz), 0], [s(rz), c(rz), 0], [0, 0, 1]]); return Rz @ Ry @ Rx

def skeleton_from_chunk(path):
    full = open(path, 'rb').read()
    if path.lower().endswith('.db'):
        n = struct.unpack_from('<I', full, 0x18)[0]; offs = sorted(set([struct.unpack_from('<I', full, 0x1C + 4*i)[0] for i in range(n + 1)] + [len(full)]))
        s = struct.unpack_from('<I', full, 0x20)[0]; c1 = full[s:offs[offs.index(s) + 1]]
    else: c1 = full
    def _ok(b):
        if b < 0 or b + 0x64 > len(c1): return False
        m_, p_, n_, o_ = struct.unpack_from('<4I', c1, b + 0x54)
        return m_ == 0x70 and 8 < n_ < 128 and p_ == m_ + 64 * n_ and o_ == p_ + 4 * n_ and b + o_ + 4 * n_ <= len(c1)
    base = struct.unpack_from('<I', c1, 0x98)[0]                  # 0 in some chunks (e.g. heavy: base 0x100) -> scan for the header
    if not _ok(base):
        base = next((b for b in range(0, min(len(c1), 0x10000), 16) if _ok(b)), None)
        if base is None: raise SystemExit('[!] not a skeleton chunk (bone table not found)')
    off_m, off_p, n, off_n = struct.unpack_from('<4I', c1, base + 0x54)
    if not (0 < n < 256): raise SystemExit('[!] not a skeleton chunk (bone table not found)')
    names, world = [], []
    for i in range(n):
        o = base + struct.unpack_from('<I', c1, base + off_n + 4*i)[0]; names.append(c1[o:c1.index(b'\0', o)].decode('latin1'))
        m = np.array(struct.unpack_from('<16f', c1, base + off_m + 64*i), float).reshape(4, 4); m[:, 3] = (0, 0, 0, 1)
        w = np.linalg.inv(m).T; w[:3, 3] *= 100; world.append(w)   # column-vector, cm
    return names, world

def skeleton_from_fbx_ascii(text):
    """ASCII FBX (';  FBX 7.x project file' header) - e.g. re-exported from Maya/Blender as ASCII."""
    models = {}; cur = None
    for line in text.splitlines():
        s = line.strip()
        if line.startswith('\tModel:') or line.startswith('\tNodeAttribute:') or line.startswith('\tGeometry:') or line.startswith('\tMaterial:'):
            m = re.match(r'Model:\s*(-?\d+),\s*"([^"]*)",\s*"([^"]*)"', s)
            if m:
                uid = int(m.group(1)); nm = m.group(2); nm = nm.split('::', 1)[1] if '::' in nm else nm
                cur = models[uid] = dict(name=nm, t=(0, 0, 0), r=(0, 0, 0), parent=None, kind=m.group(3))
            else: cur = None
        elif cur is not None and s.startswith('P: "Lcl Translation"'):
            cur['t'] = tuple(float(x) for x in s.split(',')[-3:])
        elif cur is not None and s.startswith('P: "Lcl Rotation"'):
            cur['r'] = tuple(float(x) for x in s.split(',')[-3:])
        elif s.startswith('C: "OO"'):
            a, b = [int(x) for x in s.split(',')[1:3]]
            if a in models and b in models: models[a]['parent'] = b
    return models

def skeleton_from_fbx(path):
    d = open(path, 'rb').read()
    if d[:1] == b';' or d[:20].lstrip().startswith(b'; FBX'):       # ASCII FBX
        models = skeleton_from_fbx_ascii(d.decode('latin1'))
        names, world, cache = [], [], {}
        def wm(u):
            if u in cache: return cache[u]
            m = models[u]; L = np.eye(4); L[:3, :3] = eul_xyz_mat3(*m['r']); L[:3, 3] = m['t']
            cache[u] = (wm(m['parent']) if m['parent'] in models else np.eye(4)) @ L; return cache[u]
        for u, m in models.items():
            if m['kind'] == 'Mesh' or m['name'].startswith('bg1_'): continue
            names.append(m['name']); world.append(wm(u))
        return names, world
    if d[:20] != b'Kaydara FBX Binary  ': raise SystemExit('[!] not an FBX file')
    big = struct.unpack_from('<I', d, 23)[0] >= 7500
    def rd(p):
        if big: end, npr, pl, nl = struct.unpack_from('<QQQB', d, p); p += 25
        else: end, npr, pl, nl = struct.unpack_from('<IIIB', d, p); p += 13
        if end == 0: return None, p
        name = d[p:p+nl].decode('latin1'); p += nl; props = []; q = p
        for _ in range(npr):
            t = chr(d[q]); q += 1
            if t == 'Y': props.append(struct.unpack_from('<h', d, q)[0]); q += 2
            elif t == 'C': props.append(d[q]); q += 1
            elif t == 'I': props.append(struct.unpack_from('<i', d, q)[0]); q += 4
            elif t == 'F': props.append(struct.unpack_from('<f', d, q)[0]); q += 4
            elif t == 'D': props.append(struct.unpack_from('<d', d, q)[0]); q += 8
            elif t == 'L': props.append(struct.unpack_from('<q', d, q)[0]); q += 8
            elif t in 'SR':
                l = struct.unpack_from('<I', d, q)[0]; q += 4; raw = d[q:q+l]; q += l; props.append(raw.decode('latin1') if t == 'S' else raw)
            elif t in 'fdilb': cnt, enc, cl = struct.unpack_from('<III', d, q); q += 12 + cl; props.append(None)
        p += pl; kids = []
        while p < end:
            k, p = rd(p)
            if k is None: break
            kids.append(k)
        return (name, props, kids), end
    p = 27; nodes = []
    while p < len(d) - 30:
        n, p = rd(p)
        if n is None: break
        nodes.append(n)
    obj = [n for n in nodes if n[0] == 'Objects'][0][2]; models = {}
    for n in obj:
        if n[0] != 'Model': continue
        lt = (0, 0, 0); lr = (0, 0, 0)
        for k in n[2]:
            if k[0] == 'Properties70':
                for pp in k[2]:
                    if pp[1][0] == 'Lcl Translation': lt = tuple(pp[1][4:7])
                    if pp[1][0] == 'Lcl Rotation': lr = tuple(pp[1][4:7])
        models[n[1][0]] = dict(name=n[1][1].split('\x00')[0], t=lt, r=lr, parent=None, kind=n[1][2] if len(n[1]) > 2 else '')
    for c in [n for n in nodes if n[0] == 'Connections'][0][2]:
        if c[1][0] == 'OO' and c[1][1] in models and c[1][2] in models: models[c[1][1]]['parent'] = c[1][2]
    names, world, cache = [], [], {}
    def wm(u):
        if u in cache: return cache[u]
        m = models[u]; L = np.eye(4); L[:3, :3] = eul_xyz_mat3(*m['r']); L[:3, 3] = m['t']
        cache[u] = (wm(m['parent']) if m['parent'] in models else np.eye(4)) @ L; return cache[u]
    for u, m in models.items():
        if m['kind'] == 'Mesh' or m['name'].startswith('bg1_'): continue
        names.append(m['name']); world.append(wm(u))
    return names, world

FINGER = re.compile(r'^(Finger\d|Thumb)_(Top|Mid|Tip)_(lft|rt)$')
def derive_parents(names, world, overrides):
    """Clip-rig hierarchy from bone names (same as the game's anim rig)."""
    nm = set(names); par = {}; weapon_roots = [n for n in names if n.endswith('_Root') and n != 'Root']
    for n in names:
        p = None; m = FINGER.match(n)
        if n == 'Extra': p = 'World'
        elif n == 'Root': p = 'Extra'
        elif n.startswith('Shoulder_'): p = 'Root'
        elif n.startswith('Elbow_'): p = 'Shoulder_' + n.split('_')[1]
        elif n in ('Joint_wrist_twist', 'Wrist_lft'): p = 'Elbow_lft'
        elif n == 'Wrist_rt': p = 'Elbow_rt'
        elif m:
            k, part, side = m.groups(); p = f'Wrist_{side}' if part == 'Top' else f'{k}_{"Top" if part == "Mid" else "Mid"}_{side}'
        elif n in weapon_roots: p = 'Wrist_rt'
        elif n.split('_')[0] + '_Root' in nm: p = n.split('_')[0] + '_Root'
        elif weapon_roots and n != 'World' and not n.startswith(('Wrist', 'Joint', 'Elbow', 'Shoulder')): p = weapon_roots[0]   # extra gun bones (Bullets, G_Slide, Flap_Bone, ...)
        if p is not None and p not in nm: p = None
        if p is None and n != names[0] and n != 'World':
            i = names.index(n); dist = [np.linalg.norm(world[i][:3, 3] - world[j][:3, 3]) for j in range(i)]
            p = names[int(np.argmin(dist))]; print(f'[!] no naming rule for {n}: parent guessed = {p} (override with --parent {n}=...)')
        par[n] = p
    for c, p in overrides.items():
        if c not in nm or (p not in nm and p != 'None'): raise SystemExit(f'[!] --parent {c}={p}: unknown bone')
        par[c] = None if p == 'None' else p
    return par

# AK clips (42 records) - table verified against the game's chunk06 skeleton
AK_RECS = ['World', 'Extra', 'Root', 'Shoulder_lft', 'Shoulder_rt', 'Elbow_lft', 'Elbow_rt', 'Wrist_lft', 'Joint_wrist_twist', 'Wrist_rt',
    'Finger1_Top_lft', 'Finger2_Top_lft', 'Finger3_Top_lft', 'Finger4_Top_lft', 'Thumb_Top_lft',
    'Finger1_Top_rt', 'Finger2_Top_rt', 'Finger3_Top_rt', 'Finger4_Top_rt', 'Thumb_Top_rt', 'AssualtRifle_Root',
    'Finger1_Mid_lft', 'Finger2_Mid_lft', 'Finger3_Mid_lft', 'Finger4_Mid_lft', 'Thumb_Mid_lft',
    'Finger1_Mid_rt', 'Finger2_Mid_rt', 'Finger3_Mid_rt', 'Finger4_Mid_rt', 'Thumb_Mid_rt',
    'AssualtRifle_Magazine', 'AssualtRifle_Slide', 'Bullets',
    'Finger1_Tip_lft', 'Finger2_Tip_lft', 'Finger3_Tip_lft', 'Finger4_Tip_lft', 'Thumb_Tip_lft',
    'Finger1_Tip_rt', 'Finger2_Tip_rt', 'Thumb_Tip_rt']

# Pistol clips (44 records) - verified by matching every record's translation to the pistol FBX rest offsets.
# Note the weapon children come BEFORE the tip bones, and Pistol_Root is record 20 (same slot as AssualtRifle_Root in AK clips).
PISTOL_RECS = AK_RECS[:20] + ['Pistol_Root',
    'Finger1_Mid_lft', 'Finger2_Mid_lft', 'Finger3_Mid_lft', 'Finger4_Mid_lft', 'Thumb_Mid_lft',
    'Finger1_Mid_rt', 'Finger2_Mid_rt', 'Finger3_Mid_rt', 'Finger4_Mid_rt', 'Thumb_Mid_rt',
    'Pistol_Trigger', 'Pistol_Magazine', 'Pistol_Hammer', 'Pistol_Slide',
    'Finger1_Tip_lft', 'Finger2_Tip_lft', 'Finger3_Tip_lft', 'Finger4_Tip_lft', 'Thumb_Tip_lft',
    'Finger1_Tip_rt', 'Finger2_Tip_rt', 'Finger3_Tip_rt', 'Thumb_Tip_rt']

def match_records(bones, names, world, par, overrides):
    n = len(names); idx = {x: i for i, x in enumerate(names)}
    rest_local = {}
    for x in names:
        W = world[idx[x]]; p = par[x]
        rest_local[x] = W if p is None else np.linalg.inv(world[idx[p]]) @ W
    if len(bones) == len(AK_RECS) and all(x in idx for x in AK_RECS) and not overrides:
        print('[*] 42 records + AK bone names -> using the verified AK record table')
        return list(AK_RECS), rest_local, {x: 0.0 for x in AK_RECS}
    if len(bones) == len(PISTOL_RECS) and all(x in idx for x in PISTOL_RECS) and not overrides:
        print('[*] 44 records + pistol bone names -> using the verified pistol record table')
        return list(PISTOL_RECS), rest_local, {x: 0.0 for x in PISTOL_RECS}
    rec = [None] * len(bones); cost_of = {}; used = set()
    for i, nm in overrides.items():
        if nm not in idx: raise SystemExit(f'[!] --map {i}={nm}: unknown bone')
        rec[i] = nm; used.add(nm)
    for i, b in enumerate(bones):
        if rec[i]: continue
        cand = [x for x in names if x not in used and (par[x] is None or par[x] in used)]
        if not cand: raise SystemExit(f'[!] record {i}: no bone left whose parent is already assigned (tell me - hierarchy/rig differs)')
        pos0 = None if b['pos'] is None else b['pos'][0] * 100
        sc = sorted((np.linalg.norm(rest_local[x][:3, 3] - (pos0 if pos0 is not None else 0)), idx[x], x) for x in cand)
        best = [s for s in sc if s[0] < sc[0][0] + 0.01][0]        # ties -> first in skeleton order
        rec[i] = best[2]; used.add(best[2]); cost_of[best[2]] = best[0]
    return rec, rest_local, cost_of

# ============================================================ bake
def _recover_clip(anim_path, db_path, got):
    """The extractor wrote an empty/short .anim (it cuts clips between table entries, which fails for unused table slots and for DBs whose
    table order differs from storage order). Find the clip in the .db by the clip name inside the file name (NN_<FP_xx_S_nnn>_....anim)."""
    import black_weapon_table as _T
    fn = os.path.basename(anim_path).lower(); db = open(db_path, 'rb').read()
    hits = [(nm, data) for nm, data in _T.db_clips(db) if nm and nm.lower() in fn and len(data) >= 0x20]
    if len(hits) > 1:
        best = max(len(nm) for nm, _ in hits); hits = [h for h in hits if len(h[0]) == best]
    if len(hits) != 1:
        raise SystemExit(f'[!] {os.path.basename(anim_path)} is {got} bytes and no usable clip with that name exists in {os.path.basename(db_path)} '
                         f'(the table slot is probably UNUSED filler, so there is nothing to bake). Pick another clip.')
    print(f'[*] {os.path.basename(anim_path)} was {got} bytes: recovered clip {hits[0][0]} ({len(hits[0][1]):,} bytes) directly from the .db')
    return hits[0][1]

def bake(anim_path, skel_path, out_path, fps=30.0, overrides_parent=None, overrides_map=None, list_only=False, table_path=None):
    d = open(anim_path, 'rb').read()
    if len(d) < 0x20 and skel_path.lower().endswith('.db'):      # empty/broken .anim from the extractor: take the clip straight from the .db
        d = _recover_clip(anim_path, skel_path, len(d))
    elif len(d) < 0x20: raise SystemExit(f'[!] {os.path.basename(anim_path)} is only {len(d)} bytes - not a clip (re-extract it, or pass the weapon .db as the skeleton so it can be recovered)')
    frames = struct.unpack_from('<I', d, 0x14)[0]
    if len(d) < 0x100 or frames > 5000:
        raise SystemExit(f'[!] {os.path.basename(anim_path)} is a {len(d)}-byte stub with no animation data (a placeholder slot in the .db), nothing to bake. '
                         'Run black_clip_info.py on the .db to see which clips are real.')
    bones, tail = parse_bones(d, frames)
    names, world = skeleton_from_fbx(skel_path) if skel_path.lower().endswith('.fbx') else skeleton_from_chunk(skel_path)
    par = derive_parents(names, world, overrides_parent or {})
    # per-weapon clip rig (parents + record order) inferred from ALL clips of the .db (black_weapon_table.py), or given with --table
    table = None
    if table_path: table = json.load(open(table_path))
    elif skel_path.lower().endswith('.db') and not overrides_map and not overrides_parent:
        try:
            import black_weapon_table as _T
            table = _T.infer(skel_path, verbose=False)
        except (SystemExit, Exception) as e: print(f'[!] weapon table inference failed ({e}); using the built-in tables / per-clip matching')
    if table and len(table['records']) == len(bones) and all(x in names for x in table['records']) and not overrides_map:
        par.update({k: v for k, v in table['parents'].items() if k in names})
        idx0 = {x: i for i, x in enumerate(names)}
        rest_local = {x: (world[idx0[x]] if par[x] is None else np.linalg.inv(world[idx0[par[x]]]) @ world[idx0[x]]) for x in names}
        rec = list(table['records']); cost = {}
        print(f'[*] {len(bones)} records: weapon table for {table.get("weapon", "this weapon")} (record order + gun-part parents inferred from the .db clips)')
    else:
        rec, rest_local, cost = match_records(bones, names, world, par, overrides_map or {})
    print(f'{os.path.basename(anim_path)}: {frames} frames, {len(bones)} records, skeleton {len(names)} bones')
    for i, (nm, b) in enumerate(zip(rec, bones)):
        rr = 'rest' if b['rot'] is None else ('keyed' if not np.isnan(b['rot']).any() else 'keyed+gaps')
        mv = '' if b['pos'] is None else f"pos moves {np.ptp(b['pos'], axis=0).max()*100:6.2f}cm"
        warn = '  <-- check mapping' if cost.get(nm, 0) > 8 else ''
        print(f'  {i:2d} -> {nm:20s} rot:{rr:10s} {mv}{warn}')
    free = [x for x in names if x not in rec]
    print('  bones without a record (follow their parent in rest pose):', ', '.join(free) or '-')
    if list_only: return
    idx = {x: i for i, x in enumerate(names)}; byrec = dict(zip(rec, bones))
    order = []; seen = set()
    def visit(x):
        if x in seen: return
        if par[x]: visit(par[x])
        seen.add(x); order.append(x)
    for x in names: visit(x)
    W = {}
    for x in order:
        Lr = rest_local[x]; b = byrec.get(x)
        L = np.tile(Lr, (frames, 1, 1))
        if b is not None:
            if b['rot'] is not None:
                R = quat_to_mat3(np.nan_to_num(b['rot'], nan=0.0)); gap = np.isnan(b['rot']).any(axis=1)
                R[gap] = Lr[:3, :3]; L[:, :3, :3] = R
            if b['pos'] is not None: L[:, :3, 3] = b['pos'] * 100
        W[x] = L if par[x] is None else np.matmul(W[par[x]], L)
    out_b = []
    for x in names:
        p = par[x]
        out_b.append(dict(name=x, parent=p, animated=x in byrec, rec=(rec.index(x) if x in byrec else None), rest_world=np.round(world[idx[x]][:3].ravel(), 6).tolist(),
                          world=np.round(W[x][:, :3].reshape(frames, 12), 5).tolist()))
    out = dict(clip=os.path.basename(anim_path), frames=frames, fps=fps, units='cm',
               matrix='3x4 row-major, column-vector convention, FBX (Y-up) world space', events=trailer_events(d, tail), records=list(rec), bones=out_b)
    json.dump(out, open(out_path, 'w'))
    print(f'  events in clip: {out["events"]}\n  -> {out_path}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('anim'); ap.add_argument('skeleton', help='chunk01_skeleton.bin / .db / .fbx'); ap.add_argument('out_json', nargs='?')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--parent', action='append', default=[], help='Child=Parent (Parent may be None)')
    ap.add_argument('--map', action='append', default=[], help='RECORD=BoneName override')
    ap.add_argument('--table', help='weapon table json from black_weapon_table.py (default: inferred automatically when the skeleton is a .db)')
    ap.add_argument('--list', action='store_true', help='only print the record -> bone table')
    a = ap.parse_args()
    if not a.out_json and not a.list: ap.error('out_json is required (or use --list)')
    bake(a.anim, a.skeleton, a.out_json, a.fps, dict(x.split('=', 1) for x in a.parent), {int(k): v for k, v in (x.split('=', 1) for x in a.map)}, a.list, a.table)
