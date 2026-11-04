#!/usr/bin/env python3
"""
black_anim_build.py - ENCODER for Black (2006) first-person .anim clips.     JSON (baker format)  ->  .anim

    python black_anim_build.py edited.json template.anim out.anim [options]
    python black_anim_build.py --dump-events template.anim [events.json]      (print / save the trailer event table)

INPUT   the JSON written by black_anim_bake_v5_2.py (or by black_anim_export_maya.py - same format): per bone and frame a
        WORLD 3x4 matrix (cm, FBX space), plus rest_world and the clip-hierarchy `parent`.
TEMPLATE the original .anim.  Header, trailer (events) and every track that did not change are copied BYTE FOR BYTE.

WHAT IT DOES
  1. world -> LOCAL in the clip hierarchy:  L = inv(W_parent) * W   (weapon root hangs off Wrist_rt, Wrist_lft / Joint_wrist_twist
     are siblings under Elbow_lft, ...) -> rotation quaternion (x,y,z,w; sign-continuous, unit) + position in METRES (cm / 100).
  2. records are written in the template's record order (PISTOL_RECS for 44 records, AK_RECS for 42; --records-file overrides).
  3. per record, per channel (rotation / position) independently:
        - the new data equals what the template already holds (within --rot-tol-deg / --pos-tol-cm) -> original bytes are kept
          (this includes the 'no key' rest tracks 17 / 9, polynomial, 8/16-bit quantized and segmented tracks);
        - otherwise it is re-encoded with the simplest channel the decoder understands:
              position  ch 2  constant xyz (float32)           | ch 6  raw float32 xyz per frame (exact)
              rotation  ch 10 constant quaternion (float24)    | ch 16 uint16 per component, per-track min/scale (float24)
          (there is NO raw-float rotation channel in the format, ch 16 is the most precise one)
        - frames > 255 are split into segments (ch 1 header + segments, 255 frames max, overlapping by one frame, like the game's files).
  4. trailer: copied; pointers are shifted by the size change.  Event names / times can be edited with --events (see below).
  5. header: frame count (0x14) and the trailer base address (0x28) are updated.  Everything else is untouched.
  6. self-check: the new file is decoded with the existing decoder (black_anim_bake_v5_2.parse_bones) and compared with the input.

EVENTS  (trailer)   python black_anim_build.py --dump-events clip.anim events.json   -> edit "time" (or "frame30") / "name" -> --events events.json
  The trailer has N event records {time(float32), class id, ->sub record -> ->name string}.  The time unit is NOT frames: for the pistol
  reload the times are 17, 143, 85, 56, 95 on an 86-frame clip; matched against the motion (magazine leaves ~frame 9-12, slide moves
  frames 64-77, magazine seats ~frame 47) they behave like 1/60 s ticks, i.e. frame@30fps = time/2.  This is an inference, not proven.
  Only renaming / retiming of existing events is supported (count must stay the same).
"""
import argparse, contextlib, io, json, math, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import black_anim_bake_v5_3 as B

HDR = 0x30
MAXSEG = 255
HDR_FRAMES, HDR_EVCOUNT, HDR_BASE = 0x14, 0x24, 0x28


# ============================================================ float24 (the format's 3-byte float = float32 without its low byte)
def _bits(v): return struct.unpack('<I', struct.pack('<f', float(v)))[0]
def _f24_val(t): return struct.unpack('<f', struct.pack('<I', t << 8))[0]
def f24_bytes(t): return struct.pack('<I', t << 8)[1:]

def f24_round(v, direction='nearest'):
    """integer code (24 bit) of the float24 nearest to / just below / just above v"""
    t0 = _bits(v) >> 8
    cands = [t for t in (t0 - 1, t0, t0 + 1) if 0 <= t <= 0xFFFFFF]
    vals = {t: _f24_val(t) for t in cands}
    if direction == 'nearest': return min(cands, key=lambda t: (abs(vals[t] - v), t))
    if direction == 'down':
        ok = [t for t in cands if vals[t] <= v]; return max(ok, key=lambda t: vals[t]) if ok else min(cands, key=lambda t: vals[t])
    ok = [t for t in cands if vals[t] >= v]; return min(ok, key=lambda t: vals[t]) if ok else max(cands, key=lambda t: vals[t])


# ============================================================ small maths
def to4(row12):
    m = np.tile(np.eye(4), np.shape(row12)[:-1] + (1, 1)); m[..., :3, :] = np.asarray(row12, float).reshape(np.shape(row12)[:-1] + (3, 4)); return m

def polar_rot(A):
    """nearest rotation matrix to A (drops scale / shear / noise); A (...,3,3)"""
    U, _, Vt = np.linalg.svd(A); R = U @ Vt
    bad = np.linalg.det(R) < 0
    if np.any(bad): U[bad, :, -1] *= -1; R = U @ Vt
    return R

def mat3_to_quat(R):
    R = np.asarray(R, float).reshape(-1, 3, 3); out = np.empty((len(R), 4))
    for i, M in enumerate(R):
        tr = M[0, 0] + M[1, 1] + M[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1) * 2; w = .25 * s; x = (M[2, 1] - M[1, 2]) / s; y = (M[0, 2] - M[2, 0]) / s; z = (M[1, 0] - M[0, 1]) / s
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = math.sqrt(1 + M[0, 0] - M[1, 1] - M[2, 2]) * 2; w = (M[2, 1] - M[1, 2]) / s; x = .25 * s; y = (M[0, 1] + M[1, 0]) / s; z = (M[0, 2] + M[2, 0]) / s
        elif M[1, 1] > M[2, 2]:
            s = math.sqrt(1 + M[1, 1] - M[0, 0] - M[2, 2]) * 2; w = (M[0, 2] - M[2, 0]) / s; x = (M[0, 1] + M[1, 0]) / s; y = .25 * s; z = (M[1, 2] + M[2, 1]) / s
        else:
            s = math.sqrt(1 + M[2, 2] - M[0, 0] - M[1, 1]) * 2; w = (M[1, 0] - M[0, 1]) / s; x = (M[0, 2] + M[2, 0]) / s; y = (M[1, 2] + M[2, 1]) / s; z = .25 * s
        out[i] = (x, y, z, w)
    return out / np.linalg.norm(out, axis=1, keepdims=True)

def make_continuous(q, ref0=None):
    """flip signs so consecutive quaternions have dot >= 0; first one follows ref0 (or w >= 0)"""
    q = np.array(q, float)
    if ref0 is not None:
        if np.dot(q[0], ref0) < 0: q[0] = -q[0]
    elif q[0, 3] < 0: q[0] = -q[0]
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0: q[i] = -q[i]
    return q

def quat_angle(q1, q2):
    """angle (radians) of the rotation taking q1 to q2, sign-agnostic, numerically stable for tiny angles"""
    q1 = np.asarray(q1, float); q2 = np.asarray(q2, float)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True); q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    v1, w1, v2, w2 = q1[..., :3], q1[..., 3], q2[..., :3], q2[..., 3]
    v = w1[..., None] * v2 - w2[..., None] * v1 - np.cross(v1, v2); w = w1 * w2 + (v1 * v2).sum(-1)
    return 2 * np.arctan2(np.linalg.norm(v, axis=-1), np.abs(w))


# ============================================================ template parsing
def parse_units(d):
    """records -> {'rot': unit, 'pos': unit}; unit = dict(tracks, rest, start, end) with exact byte ranges (a ch-1 group is one unit)"""
    tr, tail = B.split_tracks(d); units = []; i = 0
    def seg(i):
        if tr[i]['ch'] == 1: k = tr[i]['n']; return tr[i:i + 1 + k], i + 1 + k
        return [tr[i]], i + 1
    while i < len(tr):
        u = {}
        for kind, rest_ch in (('rot', 17), ('pos', 9)):
            if tr[i]['ch'] == rest_ch: ts = [tr[i]]; i += 1; rest = True
            else: ts, i = seg(i); rest = False
            u[kind] = dict(tracks=ts, rest=rest, start=ts[0]['off'], end=ts[-1]['off'] + ts[-1]['size'])
        units.append(u)
    return units, tail


# ============================================================ channel encoders
def seg_ranges(N):
    if N <= MAXSEG: return [(0, N)]
    out = []; s = 0
    while True:
        e = min(s + MAXSEG, N); out.append((s, e))
        if e == N: break
        s = e - 1                                                  # segments overlap by one frame
    return out

def _hdr(ch, n): return bytes([ch, n, 0, 0x7f])
def _group(parts): return parts[0] if len(parts) == 1 else bytes([1, len(parts), 0, 0x7f]) + b''.join(parts)

def enc_pos_const(p, N):
    v = np.asarray(p, float).mean(0)
    return _group([_hdr(2, e - s) + np.asarray(v, '<f4').tobytes() for s, e in seg_ranges(N)])

def enc_pos_raw(p, N):
    return _group([_hdr(6, e - s) + np.asarray(p[s:e], '<f4').tobytes() for s, e in seg_ranges(N)])

def enc_rot_const(q, N):
    m = np.asarray(q, float).mean(0); m /= np.linalg.norm(m)
    body = b''.join(f24_bytes(f24_round(c)) for c in m)
    return _group([_hdr(10, e - s) + body for s, e in seg_ranges(N)])

def enc_rot_q16(q, N):
    parts = []
    for s, e in seg_ranges(N):
        x = np.asarray(q[s:e], float); mn_c, sc_c, mn_v, sc_v = [], [], [], []
        for c in range(4):
            lo, hi = x[:, c].min(), x[:, c].max()
            tm = f24_round(lo, 'down'); mn = _f24_val(tm)
            if hi - mn <= 0: ts = f24_round(0.0); sc = 0.0
            else:
                ts = f24_round((hi - mn) / 65535.0, 'up'); sc = _f24_val(ts)
                while mn + 65535 * sc < hi and ts < 0xFFFFFF: ts += 1; sc = _f24_val(ts)        # guarantee the range is covered
            mn_c.append(tm); sc_c.append(ts); mn_v.append(mn); sc_v.append(sc)
        k = np.zeros((e - s, 4), '<u2')
        for c in range(4):
            if sc_v[c] > 0: k[:, c] = np.clip(np.rint((x[:, c] - mn_v[c]) / sc_v[c]), 0, 65535)
        parts.append(_hdr(16, e - s) + b''.join(f24_bytes(t) for t in mn_c + sc_c) + k.tobytes())
    return _group(parts)


# ============================================================ trailer (events)
def _words(b, o): return struct.unpack_from('<I', b, o)[0]

def parse_trailer(trailer, base, n_ev):
    """structured view of the event trailer, or None when it does not look like the known layout"""
    core_end = trailer.rfind(b'\0') + 1; core = trailer[:core_end]; S0 = 32 * n_ev
    if n_ev <= 0 or len(core) < S0: return None
    ev = []
    for i in range(n_ev):
        o = 16 * i; ptr = _words(core, o + 12) - base
        if not (16 * n_ev <= ptr < S0 and ptr % 16 == 0): return None
        so = ptr; nptr = _words(core, so + 12) - base
        if not (S0 <= nptr < len(core)): return None
        ev.append(dict(raw=core[o:o + 12], sub_index=(ptr - 16 * n_ev) // 16, sub_raw=core[so:so + 12], name_off=nptr))
    strings = []; o = S0
    while o < len(core):
        e = core.index(b'\0', o); strings.append((o, core[o:e])); o = e + 1
    soff = {o: i for i, (o, _) in enumerate(strings)}
    for e in ev:
        if e['name_off'] not in soff: return None
        e['name_idx'] = soff[e['name_off']]; e['time'] = struct.unpack('<f', e['raw'][:4])[0]; e['name'] = strings[e['name_idx']][1].decode('latin1')
    model = dict(events=ev, strings=[s for _, s in strings], n=n_ev)
    if rebuild_trailer(model, base) != core: return None            # unknown layout details -> do not touch the structure
    return model

def rebuild_trailer(model, base, edits=None):
    n = model['n']; ev = [dict(e) for e in model['events']]; strings = list(model['strings'])
    for i, ed in enumerate(edits or []):
        if ed is None: continue
        if 'time' in ed: ev[i]['time'] = float(ed['time'])
        if 'name' in ed and ed['name'] != ev[i]['name']:
            nb = ed['name'].encode('latin1')
            if sum(1 for e in ev if e['name_idx'] == ev[i]['name_idx']) == 1: strings[ev[i]['name_idx']] = nb
            else:                                                    # shared string: give this event its own copy right behind the old one
                at = ev[i]['name_idx'] + 1; strings.insert(at, nb)
                for e in ev:
                    if e['name_idx'] >= at: e['name_idx'] += 1
                ev[i]['name_idx'] = at
    offs = []; o = 32 * n
    for s in strings: offs.append(o); o += len(s) + 1
    out = bytearray()
    for i, e in enumerate(ev):
        raw = bytearray(e['raw']); raw[:4] = struct.pack('<f', e['time'])
        out += raw + struct.pack('<I', base + 16 * n + 16 * e['sub_index'])
    subs = [None] * n
    for e in ev: subs[e['sub_index']] = e
    for e in subs: out += e['sub_raw'] + struct.pack('<I', base + offs[e['name_idx']])
    for s in strings: out += s + b'\0'
    return bytes(out)

def load_event_edits(src, n_ev):
    ev = src.get('events') if isinstance(src, dict) else src
    if not isinstance(ev, list) or not ev or not isinstance(ev[0], dict): return None
    if len(ev) != n_ev: raise SystemExit(f'[!] event table has {len(ev)} entries, the template has {n_ev}: adding / removing events is not supported (header counts at 0x1c/0x24 are not understood)')
    out = []
    for i, e in enumerate(ev):
        ed = {}
        if 'time' in e: ed['time'] = e['time']
        elif 'frame30' in e: ed['time'] = 2.0 * float(e['frame30'])
        if 'name' in e: ed['name'] = e['name']
        out.append(ed)
    return out

def dump_events(anim_path, out_path=None):
    d = open(anim_path, 'rb').read(); units, tail = parse_units(d); base = _words(d, HDR_BASE); n_ev = _words(d, HDR_EVCOUNT)
    m = parse_trailer(d[tail:], base, n_ev)
    if m is None: raise SystemExit('[!] trailer layout not recognised (events not editable); strings: ' + ', '.join(B.trailer_events(d, tail)))
    rows = []
    for i, e in enumerate(m['events']):
        rows.append(dict(index=i, time=e['time'], frame30=e['time'] / 2, name=e['name'], class_id=hex(_words(e['raw'], 4)), raw_hex=e['raw'].hex()))
        print(f"  {i}: time={e['time']:8.2f} (frame30 ~ {e['time'] / 2:6.2f})  {e['name']:12s} class_id={hex(_words(e['raw'], 4))}")
    print('  frames in clip:', _words(d, HDR_FRAMES), '| other strings:', [s.decode('latin1') for k, s in enumerate(m['strings']) if k not in {e['name_idx'] for e in m['events']}])
    if out_path:
        json.dump(dict(note='time = raw float in the file (looks like 1/60 s ticks: frame@30fps = time/2). Edit time or frame30 and name; keep the count.', events=rows), open(out_path, 'w'), indent=1)
        print('  ->', out_path)


# ============================================================ JSON -> local tracks
def rec_names_for(count, bones_json, override=None):
    rec_idx = sorted((x['rec'], x['name']) for x in bones_json if x.get('rec') is not None)
    if override: names = override
    elif rec_idx and len(rec_idx) == count and [i for i, _ in rec_idx] == list(range(count)): names = [n for _, n in rec_idx]   # record order stored in the JSON by the baker (any weapon)
    elif count == len(B.PISTOL_RECS): names = B.PISTOL_RECS
    elif count == len(B.AK_RECS): names = B.AK_RECS
    else: raise SystemExit(f'[!] template has {count} records: no built-in record table for that (re-bake this clip with black_anim_bake_v5_3.py from the .db so the JSON carries its record order, or pass --records-file)')
    have = {b['name'] for b in bones_json}; miss = [n for n in names if n not in have]
    if miss: raise SystemExit(f'[!] JSON has no bone(s) {miss} that the record table needs')
    if len(names) != count: raise SystemExit(f'[!] record list has {len(names)} names, template has {count}')
    return list(names)

def compute_targets(js, rec_names):
    """per record: new local rotation quaternion (N,4), position in metres (N,3), plus the rest local (q, p)"""
    if js.get('units', 'cm') != 'cm': raise SystemExit('[!] JSON units are not cm')
    bones = {b['name']: b for b in js['bones']}; N = js['frames']
    for b in js['bones']:
        if len(b['world']) != N: raise SystemExit(f"[!] bone {b['name']}: {len(b['world'])} frames, JSON says {N}")
    W = {n: to4(np.asarray(b['world'], float)) for n, b in bones.items()}; G = {n: to4(np.asarray(b['rest_world'], float)) for n, b in bones.items()}
    out = {}
    for n in rec_names:
        p = bones[n]['parent']
        if p not in bones: p = None
        L = W[n] if p is None else np.linalg.inv(W[p]) @ W[n]
        Lr = G[n] if p is None else np.linalg.inv(G[p]) @ G[n]
        out[n] = dict(q=mat3_to_quat(polar_rot(L[:, :3, :3])), p=L[:, :3, 3] / 100.0, rq=mat3_to_quat(polar_rot(Lr[None, :3, :3]))[0], rp=Lr[:3, 3] / 100.0)
    return out, W, G, bones

def fk_from_decoded(js, rec_names, dec, N):
    """world matrices from decoded tracks, exactly the way the baker does it (rest local from the JSON rest pose for missing channels)"""
    bones = {b['name']: b for b in js['bones']}; G = {n: to4(np.asarray(b['rest_world'], float)) for n, b in bones.items()}
    byrec = dict(zip(rec_names, dec)); W = {}; seen = set()
    def rest_local(n):
        p = bones[n]['parent']; return G[n] if p not in bones else np.linalg.inv(G[p]) @ G[n]
    def visit(n):
        if n in seen: return
        p = bones[n]['parent']
        if p in bones: visit(p)
        seen.add(n); Lr = rest_local(n); L = np.tile(Lr, (N, 1, 1)); b = byrec.get(n)
        if b is not None:
            if b['rot'] is not None:
                R = B.quat_to_mat3(np.nan_to_num(b['rot'], nan=0.0)); gap = np.isnan(b['rot']).any(axis=1); R[gap] = Lr[:3, :3]; L[:, :3, :3] = R
            if b['pos'] is not None: L[:, :3, 3] = b['pos'] * 100
        W[n] = L if p not in bones else np.matmul(W[p], L)
    for n in bones: visit(n)
    return W


# ============================================================ the encoder
def encode(js, template, rot_tol_deg=0.005, pos_tol_cm=0.002, force=False, events=None, records=None, log=print, baseline=None, base_tol=(0.0005, 0.0005)):
    d = bytes(template); hdr = bytearray(d[:HDR]); N0 = struct.unpack_from('<I', d, HDR_FRAMES)[0]; N = int(js['frames'])
    units, tail0 = parse_units(d); old, _ = B.parse_bones(d, N0)
    rec = rec_names_for(len(units), js['bones'], records)
    tg, Wj, G, bones = compute_targets(js, rec)
    rot_tol = math.radians(rot_tol_deg); pos_tol = pos_tol_cm / 100.0
    tgb = None
    if baseline is not None:
        if baseline['frames'] != N or [b['name'] for b in baseline['bones']] != [b['name'] for b in js['bones']]: log('[!] baseline JSON does not match (frames / bones): ignored')
        else: tgb = compute_targets(baseline, rec)[0]
    b_rot = math.radians(base_tol[0]); b_pos = base_tol[1] / 100.0
    pieces = []; rep = []; same_n = (N == N0)
    for i, name in enumerate(rec):
        t = tg[name]; u = units[i]; o = old[i]; row = dict(name=name)
        # ---------- rotation
        orig_q = np.tile(t['rq'], (N0, 1)) if o['rot'] is None else np.where(np.isnan(o['rot']), t['rq'], o['rot'])
        qn = make_continuous(t['q'], orig_q[0])
        keep = False
        if tgb is not None and same_n and quat_angle(tgb[name]['q'], t['q']).max() <= b_rot: keep = True      # identical to the un-edited Maya export -> untouched
        elif same_n and (not force or u['rot']['rest']):
            e = quat_angle(orig_q / np.linalg.norm(orig_q, axis=1, keepdims=True), qn).max(); keep = e <= rot_tol
        if keep: b = d[u['rot']['start']:u['rot']['end']]; row['rot'] = 'rest' if u['rot']['rest'] else 'kept'
        elif u['rot']['rest'] and N <= MAXSEG and quat_angle(np.tile(t['rq'], (N, 1)), qn).max() <= rot_tol: b = _hdr(17, N); row['rot'] = 'rest'
        elif np.ptp(qn, axis=0).max() < 1e-6: b = enc_rot_const(qn, N); row['rot'] = 'const'
        else: b = enc_rot_q16(qn, N); row['rot'] = 'q16'
        pieces.append(b); row['rot_bytes'] = len(b)
        # ---------- position
        orig_p = o['pos'] if o['pos'] is not None else np.tile(t['rp'], (N0, 1))
        keep = False
        if tgb is not None and same_n and np.abs(tgb[name]['p'] - t['p']).max() <= b_pos: keep = True
        elif same_n and (not force or u['pos']['rest']):
            e = np.abs(orig_p - t['p']).max(); keep = e <= pos_tol
        if keep: b = d[u['pos']['start']:u['pos']['end']]; row['pos'] = 'rest' if u['pos']['rest'] else 'kept'
        elif u['pos']['rest'] and N <= MAXSEG and np.abs(t['rp'] - t['p']).max() <= pos_tol: b = _hdr(9, N); row['pos'] = 'rest'
        elif np.ptp(t['p'], axis=0).max() < 2e-7: b = enc_pos_const(t['p'], N); row['pos'] = 'const'
        else: b = enc_pos_raw(t['p'], N); row['pos'] = 'raw'
        pieces.append(b); row['pos_bytes'] = len(b); rep.append(row)
    body = b''.join(pieces)
    # ---------- trailer
    base0 = _words(d, HDR_BASE); n_ev = _words(d, HDR_EVCOUNT); trailer0 = d[tail0:]
    tail1 = HDR + len(body); delta = tail1 - tail0; base1 = base0 + delta
    model = parse_trailer(trailer0, base0, n_ev); edits = load_event_edits(events, n_ev) if events is not None else None
    if edits is not None and model is None: raise SystemExit('[!] trailer layout not recognised: cannot apply event edits')
    if model is not None:
        core = rebuild_trailer(model, base1, edits)
    else:                                                          # unknown layout: copy and shift every in-range pointer word
        core = bytearray(trailer0[:trailer0.rfind(b'\0') + 1]); core += bytes((-len(core)) % 4)
        for o in range(0, len(core) - 3, 4):
            v = _words(core, o)
            if base0 <= v < base0 + len(trailer0): struct.pack_into('<I', core, o, v + delta)
        core = bytes(core); log('[!] trailer layout not recognised: copied verbatim, in-file pointers shifted by %+d' % delta)
    tot = tail1 + len(core); pad = b'_' * ((-tot) % 16); trailer1 = core + pad
    struct.pack_into('<I', hdr, HDR_FRAMES, N); struct.pack_into('<I', hdr, HDR_BASE, base1)
    out = bytes(hdr) + body + trailer1
    info = dict(records=rep, size_old=len(d), size_new=len(out), tail_old=tail0, tail_new=tail1, delta=delta, frames_old=N0, frames_new=N)
    return out, info, dict(rec=rec, targets=tg, W=Wj, js=js)


def check(out, js, rec, targets, N, log=print):
    """decode `out` with the existing decoder; per-record error vs the targets + world-matrix error vs the JSON"""
    dec, tail = B.parse_bones(out, struct.unpack_from('<I', out, HDR_FRAMES)[0]); res = []
    for name, b in zip(rec, dec):
        t = targets[name]
        q = t['rq'][None].repeat(N, 0) if b['rot'] is None else np.where(np.isnan(b['rot']), t['rq'], b['rot'])
        p = t['rp'][None].repeat(N, 0) if b['pos'] is None else b['pos']
        res.append(dict(name=name, rot_deg=math.degrees(quat_angle(q, t['q']).max()), pos_cm=np.abs(p - t['p']).max() * 100))
    W = fk_from_decoded(js, rec, dec, N); wr = {}
    for bn in js['bones']:
        ref = to4(np.asarray(bn['world'], float)); d_ = W[bn['name']] - ref
        wr[bn['name']] = (np.abs(d_[:, :3, :3]).max(), np.abs(d_[:, :3, 3]).max())
    return res, wr


# ============================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('json'); ap.add_argument('template'); ap.add_argument('out')
    ap.add_argument('--events', help='events.json from --dump-events (or the "event_table" key inside the main JSON is used automatically)')
    ap.add_argument('--force-reencode', action='store_true', help='re-encode every keyed channel even if unchanged (encoder test)')
    ap.add_argument('--rot-tol-deg', type=float, default=0.005, help='unchanged-rotation tolerance (default 0.005 deg)')
    ap.add_argument('--pos-tol-cm', type=float, default=0.002, help='unchanged-position tolerance (default 0.002 cm)')
    ap.add_argument('--baseline', help='the UN-EDITED Maya export (same exporter, no changes): channels whose data equals it are treated as untouched and keep the original bytes, '
                                       'so Maya\'s numeric noise cannot trigger a re-encode (strongly recommended with Maya exports)')
    ap.add_argument('--baseline-tol-deg', type=float, default=0.0005); ap.add_argument('--baseline-tol-cm', type=float, default=0.0005)
    ap.add_argument('--records-file', help='text file, one bone name per line in record order (for clips that are not 42 / 44 records)')
    ap.add_argument('--verify', metavar='SKELETON', help='also re-bake the new .anim with black_anim_bake_v5_2.bake() (chunk01_skeleton.bin / .fbx binary or ASCII) and compare')
    a = ap.parse_args(argv)
    js = json.load(open(a.json)); tpl = open(a.template, 'rb').read()
    ev = json.load(open(a.events)) if a.events else js.get('event_table')
    recs = [l.strip() for l in open(a.records_file) if l.strip()] if a.records_file else None
    out, info, ctx = encode(js, tpl, a.rot_tol_deg, a.pos_tol_cm, a.force_reencode, ev, recs, baseline=json.load(open(a.baseline)) if a.baseline else None, base_tol=(a.baseline_tol_deg, a.baseline_tol_cm))
    open(a.out, 'wb').write(out)
    res, wr = check(out, js, ctx['rec'], ctx['targets'], info['frames_new'])
    print(f"{os.path.basename(a.template)} -> {a.out}: frames {info['frames_old']} -> {info['frames_new']}, size {info['size_old']} -> {info['size_new']} ({info['size_new'] - info['size_old']:+d}), trailer moved {info['delta']:+d}")
    print('  rec bone                 rot    pos    bytes   rot err(deg)  pos err(cm)')
    for i, (r, c) in enumerate(zip(info['records'], res)):
        print(f"  {i:3d} {r['name']:20s} {r['rot']:5s}  {r['pos']:5s} {r['rot_bytes'] + r['pos_bytes']:6d}   {c['rot_deg']:10.5f}  {c['pos_cm']:10.6f}")
    kept = sum(1 for r in info['records'] for k in ('rot', 'pos') if r[k] in ('kept', 'rest'))
    print(f"  channels kept byte-identical: {kept}/{2 * len(res)}   re-encoded: {2 * len(res) - kept}")
    print('  max rotation error %.5f deg | max position error %.6f cm | max world-matrix error: rot %.2e, pos %.5f cm' %
          (max(c['rot_deg'] for c in res), max(c['pos_cm'] for c in res), max(v[0] for v in wr.values()), max(v[1] for v in wr.values())))
    if a.verify:
        import tempfile
        try:
            import black_anim_skel_ascii; black_anim_skel_ascii.install()
        except ImportError: pass
        tmp = os.path.join(tempfile.mkdtemp(), 'rebake.json')
        with contextlib.redirect_stdout(io.StringIO()): B.bake(a.out, a.verify, tmp)
        nb = json.load(open(tmp)); worst = (0, 0, '')
        for x, y in zip(js['bones'], nb['bones']):
            dd = np.abs(to4(np.asarray(x['world'], float)) - to4(np.asarray(y['world'], float)))
            worst = max(worst, (dd[:, :3, 3].max(), dd[:, :3, :3].max(), x['name']))
        print('  re-bake with the original baker vs input JSON: worst translation error %.5f cm, worst rotation-entry error %.2e (%s)' % worst)


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--dump-events': dump_events(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else: main()
