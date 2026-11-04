#!/usr/bin/env python3
"""BLACK (2006) generic exact-size multi-track animation patcher.

Patches the selected existing .anim payload in-place while preserving its own
frame count, record/segment layout, original clip size, and DB archive size.
The baseline JSON and edited JSON must both be baked/exported from the SAME
clip as --template-anim and must have the same frame count.

Only existing track segments whose current encoding can represent the edited
transform using the same channel type and byte size are rewritten. Unsupported
segments retain their original bytes and are reported as skipped. This fails safe:
it does not invent new tracks or shift bytes.

Important: exact-size/decoder validation is not a game-engine compatibility
proof. Test the generated DB in a separate copy of the game.
"""
from __future__ import annotations
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np
import black_anim_bake_v5_3 as B
import black_anim_build as E

DB_CHUNK_COUNT_OFF = 0x18
DB_CHUNK_OFFS_OFF = 0x1C
DB_ANIM_CHUNK_INDEX = 4
BASE = 0x100
ANIM_COUNT_OFF = 0x124
ANIM_DESC_TABLE_OFF = 0x140
ANIM_DATA_TABLE_OFF = 0x190

# Conservative representation limits. Segments exceeding these are kept original.
CHANGED_ROT_DEG = 0.05
CHANGED_POS_CM = 0.01
CONST_ROT_MAX_DEG = 2.0
POLY_ROT_MAX_ERR_DEG = 1.5
POLY_POS_MAX_ERR_CM = 0.10
QUANT_ROT_MAX_ERR_DEG = 3.0
QUANT_POS_MAX_ERR_CM = 0.30


def u32(data, off):
    return struct.unpack_from('<I', data, off)[0]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def world_array(bone):
    return E.to4(np.asarray(bone['world'], dtype=float).reshape(-1, 12))


def seg_entries(unit):
    """Return (global_start, global_end_exclusive, segment_track) entries."""
    tracks = unit['tracks']
    if not tracks:
        return []
    child_tracks = tracks[1:] if tracks[0]['ch'] == 1 else tracks
    out, start = [], 0
    for tr in child_tracks:
        end = start + tr['n']
        out.append((start, end, tr))
        start = end - 1  # consecutive segments overlap by one frame
    return out


def q_angle_deg(q0, q1):
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    q0 = q0 / np.maximum(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-12)
    q1 = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-12)
    dot = np.clip(np.abs(np.sum(q0*q1, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def f24_float(code):
    return E._f24_val(int(code))


def f24_pack_values(vals, directions=None):
    out = bytearray()
    if directions is None:
        directions = ['nearest'] * len(vals)
    for v, direction in zip(vals, directions):
        out += E.f24_bytes(E.f24_round(float(v), direction))
    return bytes(out)


def fit_rot_poly(q, degree_plus_one):
    q = E.make_continuous(np.asarray(q, float))
    k = int(degree_plus_one)
    u = np.linspace(0.0, 1.0, len(q))
    X = np.stack([u**i for i in range(k)], axis=1)
    coeff_asc = np.linalg.lstsq(X, q, rcond=None)[0]
    coeff_desc = coeff_asc[::-1]
    body = f24_pack_values(coeff_desc.reshape(-1), ['nearest'] * (k*4))
    tmp = bytes([({2:11,3:12,4:13}[k]), len(q), 0, 0x7f]) + body
    tr = {'off': 0, 'ch': {2:11,3:12,4:13}[k], 'n': len(q), 'size': len(tmp)}
    decoded = B.eval_rot(tmp, tr)
    err = float(q_angle_deg(q, decoded).max())
    return tmp, err


def encode_rot_segment(template, tr, target_q):
    """Return (bytes, max_error_degrees, reason). None bytes means keep original."""
    ch, n, size, off = tr['ch'], tr['n'], tr['size'], tr['off']
    original = template[off:off+size]
    head = original[:4]
    q = E.make_continuous(np.asarray(target_q, float))
    if len(q) != n or not np.isfinite(q).all():
        return None, None, 'invalid target quaternion samples'

    if ch == 17:
        return None, None, 'rotation channel 17 is no-key/rest; cannot add rotation without changing layout'
    if ch == 10:
        mean = q.mean(axis=0); norm = np.linalg.norm(mean)
        if norm < 1e-10:
            return None, None, 'constant quaternion average is degenerate'
        mean /= norm
        err = float(q_angle_deg(q, np.tile(mean, (len(q), 1))).max())
        if err > CONST_ROT_MAX_DEG:
            return None, err, f'constant channel needs {err:.3f} deg variation (limit {CONST_ROT_MAX_DEG})'
        payload = f24_pack_values(mean)
        new = head + payload
        return (new, err, 'constant quaternion') if len(new) == size else (None, err, 'constant track size mismatch')

    if ch in (11, 12, 13):
        k = {11:2, 12:3, 13:4}[ch]
        new, err = fit_rot_poly(q, k)
        # Keep the source header verbatim (including the existing frame count).
        new = head + new[4:]
        if len(new) != size:
            return None, err, 'polynomial track size mismatch'
        if err > POLY_ROT_MAX_ERR_DEG:
            return None, err, f'polynomial fit error {err:.3f} deg (limit {POLY_ROT_MAX_ERR_DEG})'
        return new, err, 'polynomial quaternion fit'

    if ch not in (15, 16):
        return None, None, f'unsupported rotation channel {ch}'

    levels = 255 if ch == 15 else 65535
    ints_dtype = np.uint8 if ch == 15 else '<u2'
    mn_codes, sc_codes, mn_vals, sc_vals = [], [], [], []
    for c in range(4):
        lo, hi = float(q[:, c].min()), float(q[:, c].max())
        mc = E.f24_round(lo, 'down'); mv = f24_float(mc)
        if hi - mv <= 0.0:
            sc, sv = E.f24_round(0.0), 0.0
        else:
            sc = E.f24_round((hi - mv) / levels, 'up'); sv = f24_float(sc)
            while mv + levels * sv < hi and sc < 0xFFFFFF:
                sc += 1; sv = f24_float(sc)
        mn_codes.append(mc); sc_codes.append(sc); mn_vals.append(mv); sc_vals.append(sv)
    vals = np.zeros((n, 4), dtype=ints_dtype)
    for c in range(4):
        if sc_vals[c] > 0:
            vals[:, c] = np.clip(np.rint((q[:, c] - mn_vals[c]) / sc_vals[c]), 0, levels).astype(int)
    payload = f24_pack_values([f24_float(x) for x in mn_codes + sc_codes]) + vals.tobytes()
    new = head + payload
    if len(new) < size:
        new += b'\0' * (size - len(new))
    if len(new) != size:
        return None, None, 'quantized rotation track size mismatch'
    decoded = B.eval_rot(new, {'off':0, 'ch':ch, 'n':n, 'size':len(new)})
    err = float(q_angle_deg(q, decoded).max())
    if err > QUANT_ROT_MAX_ERR_DEG:
        return None, err, f'quantized rotation error {err:.3f} deg (limit {QUANT_ROT_MAX_ERR_DEG})'
    return new, err, f'{8 if ch==15 else 16}-bit quaternion quantization'


def fit_pos_poly(p, degree_plus_one):
    p = np.asarray(p, float); k = int(degree_plus_one)
    x = np.arange(len(p), dtype=float)
    coeff_asc = np.polynomial.polynomial.polyfit(x, p, deg=k-1)
    body = np.asarray(coeff_asc[::-1], dtype='<f4').tobytes()
    ch = {2:3,3:4,4:5}[k]
    tmp = bytes([ch, len(p), 0, 0x7f]) + body
    decoded = B.eval_pos(tmp, {'off':0, 'ch':ch, 'n':len(p), 'size':len(tmp)})
    err_cm = float(np.linalg.norm((p-decoded)*100.0, axis=1).max())
    return tmp, err_cm


def encode_pos_segment(template, tr, target_p):
    ch, n, size, off = tr['ch'], tr['n'], tr['size'], tr['off']
    original = template[off:off+size]
    head = original[:4]
    p = np.asarray(target_p, float)
    if len(p) != n or not np.isfinite(p).all():
        return None, None, 'invalid target position samples'
    if ch == 9:
        return None, None, 'position channel 9 is no-key/rest; cannot add translation without changing layout'
    if ch == 2:
        mean = p.mean(axis=0)
        err = float(np.linalg.norm((p-mean)*100.0, axis=1).max())
        if err > CHANGED_POS_CM:
            return None, err, f'constant position varies {err:.4f} cm (limit {CHANGED_POS_CM})'
        new = head + np.asarray(mean, dtype='<f4').tobytes()
        return (new, err, 'constant position') if len(new) == size else (None, err, 'constant position size mismatch')
    if ch in (3,4,5):
        k = {3:2, 4:3, 5:4}[ch]
        new, err = fit_pos_poly(p, k)
        new = head + new[4:]
        if len(new) != size:
            return None, err, 'polynomial position size mismatch'
        if err > POLY_POS_MAX_ERR_CM:
            return None, err, f'polynomial fit error {err:.4f} cm (limit {POLY_POS_MAX_ERR_CM})'
        return new, err, 'polynomial position fit'
    if ch == 6:
        new = head + np.asarray(p, dtype='<f4').tobytes()
        return (new, 0.0, 'raw float position') if len(new) == size else (None, None, 'raw position size mismatch')
    if ch not in (7,8):
        return None, None, f'unsupported position channel {ch}'

    levels = 255 if ch == 7 else 65535
    ints_dtype = np.uint8 if ch == 7 else '<u2'
    lo = p.min(axis=0); hi = p.max(axis=0)
    mn = np.empty(3, dtype=np.float32); sc = np.empty(3, dtype=np.float32)
    for c in range(3):
        mn[c] = np.nextafter(np.float32(lo[c]), np.float32(-np.inf))
        span = float(hi[c]) - float(mn[c])
        if span <= 1e-12:
            mn[c] = np.float32(p[:,c].mean()); sc[c] = np.float32(0.0)
        else:
            sc[c] = np.float32(span / levels)
            while float(mn[c]) + levels * float(sc[c]) < float(hi[c]):
                sc[c] = np.nextafter(sc[c], np.float32(np.inf), dtype=np.float32)
    vals = np.zeros((n,3), dtype=ints_dtype)
    for c in range(3):
        if sc[c] > 0:
            vals[:,c] = np.clip(np.rint((p[:,c]-float(mn[c]))/float(sc[c])),0,levels).astype(int)
    body = mn.astype('<f4').tobytes() + sc.astype('<f4').tobytes() + vals.tobytes()
    new = head + body
    if len(new) < size:
        new += b'\0'*(size-len(new))
    if len(new) != size:
        return None, None, 'quantized position track size mismatch'
    decoded = B.eval_pos(new, {'off':0,'ch':ch,'n':n,'size':len(new)})
    err = float(np.linalg.norm((p-decoded)*100.0,axis=1).max())
    if err > QUANT_POS_MAX_ERR_CM:
        return None, err, f'quantized position error {err:.4f} cm (limit {QUANT_POS_MAX_ERR_CM})'
    return new, err, f'{8 if ch==7 else 16}-bit position quantization'


def locate_anim_clip(db: bytes, template: bytes):
    chunk_count = u32(db, DB_CHUNK_COUNT_OFF)
    outer_offsets = [u32(db, DB_CHUNK_OFFS_OFF + 4*i) for i in range(chunk_count + 1)]
    c4 = outer_offsets[DB_ANIM_CHUNK_INDEX]
    greater = [x for x in outer_offsets if x > c4]
    if not greater:
        raise ValueError('Cannot locate end of DB animation chunk.')
    c4end = min(greater)
    chunk = db[c4:c4end]
    n = u32(chunk, ANIM_COUNT_OFF)
    raw = [BASE + u32(chunk, ANIM_DATA_TABLE_OFF + 4*i) for i in range(n)]
    # Some DBs (bg1_bns, bg1_hvy, bg1_sm5 ...) have UNUSED slots in the animation table that hold filler (0x5f5f5f5f) instead of an offset, and
    # some list clips in a different order than they are stored. Unused slots are skipped; a clip ends where the NEXT STORED clip begins,
    # so we work on the sorted, de-duplicated list of valid starts. idx below is the position in that sorted list.
    valid = [i for i, x in enumerate(raw) if BASE <= x < len(chunk)]
    if not valid:
        raise ValueError(f'No valid clip offsets in the animation table (clip count read as {n}); this DB has a different animation-table layout - send this .db.')
    starts = sorted({raw[i] for i in valid}) + [len(chunk)]
    desc = [BASE + u32(chunk, ANIM_DESC_TABLE_OFF + 4*i) for i in range(n)]
    names = [chunk[o+0x10:o+0x28].split(b'\0', 1)[0].decode('ascii', 'replace') if 0 <= o < len(chunk) - 0x28 else '?' for o in desc]
    # Match the exact animation payload, not a reload-specific internal clip name.
    order = {s: k for k, s in enumerate(starts)}
    exact = sorted({order[raw[i]] for i in valid if chunk[raw[i]:starts[order[raw[i]] + 1]] == template})
    names_by_start = {}
    for i in valid: names_by_start.setdefault(order[raw[i]], names[i])
    names = names_by_start
    if len(exact) != 1:
        raise ValueError(
            f'Source DB/template mismatch: expected one exact byte-for-byte clip match, found {len(exact)} (indices={exact}). '
            'Use an unmodified .anim extracted from this same source DB.'
        )
    return c4, c4end, starts, exact[0], names[exact[0]]


def slide_position_target(edited_json, baseline_json, bone='Pistol_Slide'):
    """Keep the slide-position convention from the previously game-loading patcher."""
    edits = {b['name']: b for b in edited_json['bones']}
    base = {b['name']: b for b in baseline_json['bones']}
    parent = edits[bone].get('parent')
    if parent not in base:
        raise ValueError(f'{bone} parent {parent!r} not found in baseline JSON.')
    Wp = world_array(base[parent])
    Ws = world_array(edits[bone])
    local = np.linalg.inv(Wp) @ Ws
    return local[:, :3, 3] / 100.0


def patch_anim(template, baseline, edited, slide_position_mode='auto'):
    frames = u32(template, E.HDR_FRAMES)
    base_frames = baseline.get('frames')
    edit_frames = edited.get('frames')
    if base_frames != frames or edit_frames != frames:
        raise ValueError(
            f'Frame-count mismatch: template={frames}, baseline JSON={base_frames}, edited JSON={edit_frames}. '
            'Bake/export a new baseline from this exact template .anim, then make the edited JSON from that baseline.'
        )
    if baseline.get('clip') != edited.get('clip'):
        raise ValueError('Baseline and edited JSON do not refer to the same clip.')
    if [b['name'] for b in baseline['bones']] != [b['name'] for b in edited['bones']]:
        raise ValueError('Baseline and edited JSON bone order differs.')

    units, tail = E.parse_units(template)
    records = E.rec_names_for(len(units), edited['bones'])
    if len(records) != len(units):
        raise ValueError(f'Record mapping mismatch: {len(records)} JSON records for {len(units)} template records.')
    base_targets = E.compute_targets(baseline, records)[0]
    edit_targets = E.compute_targets(edited, records)[0]
    old_dec, old_tail = B.parse_bones(template, frames)
    if old_tail != tail:
        raise ValueError('Original clip parser disagreement about trailer offset.')

    # The coordinate convention below was needed for the known-working reload clip.
    # In auto mode, keep it for clips whose exported clip name contains "reload";
    # do not apply the reload-specific override to movement/idle/other clips.
    clip_label = str(baseline.get('clip', '')).lower()
    names = {b['name'] for b in baseline['bones']}
    slide_bone = None                                   # legacy slide fix: only auto-applied to the pistol (the only weapon it was verified on)
    if slide_position_mode == 'on':                     # forced: use <Weapon>_Slide of whatever weapon this is
        for n in names:
            if n.endswith('_Root') and n != 'Root' and n[:-4] + 'Slide' in names: slide_bone = n[:-4] + 'Slide'
        if slide_bone is None: raise ValueError('Slide-position fix forced, but no <Weapon>_Slide bone found.')
    elif slide_position_mode == 'auto' and 'reload' in clip_label and 'Pistol_Slide' in names:
        slide_bone = 'Pistol_Slide'
    if slide_bone:
        edit_targets[slide_bone]['p'] = slide_position_target(edited, baseline, slide_bone)
    output = bytearray(template)
    patched = []
    skipped = []
    changed_seg_count = 0

    for i, (name, unit) in enumerate(zip(records, units)):
        for kind, field, encoder, compare in (
            ('rot', 'q', encode_rot_segment, 'rot'),
            ('pos', 'p', encode_pos_segment, 'pos'),
        ):
            target = np.asarray(edit_targets[name][field], float)
            base_track = np.asarray(base_targets[name][field], float)
            for s, e, tr in seg_entries(unit[kind]):
                # Avoid re-encoding unchanged segments because of Maya matrix noise.
                if kind == 'rot':
                    delta = float(q_angle_deg(base_track[s:e], target[s:e]).max())
                    changed = delta >= CHANGED_ROT_DEG
                    # The original file uses a no-key channel as rest rotation; compare above
                    # against the original baseline so only actual intended edits are attempted.
                else:
                    delta = float(np.linalg.norm((target[s:e] - base_track[s:e])*100.0, axis=1).max())
                    changed = delta >= CHANGED_POS_CM
                if not changed:
                    continue

                if tr['ch'] in (17,9):
                    new_bytes, err, why = None, None, ('no-key/rest channel; original segment preserved')
                else:
                    if kind == 'rot': vals = target[s:e]
                    else: vals = target[s:e]
                    new_bytes, err, why = encoder(bytes(output), tr, vals)
                if new_bytes is None:
                    skipped.append({
                        'bone': name, 'kind': kind, 'channel': tr['ch'],
                        'frames': [s, e-1], 'delta': delta,
                        'error': err, 'reason': why,
                    })
                    continue
                if len(new_bytes) != tr['size']:
                    raise ValueError(f'{name} {kind} ch{tr["ch"]} frame {s}-{e-1}: encoded size changed.')
                output[tr['off']:tr['off']+tr['size']] = new_bytes
                changed_seg_count += 1
                patched.append({
                    'bone': name, 'kind': kind, 'channel': tr['ch'],
                    'frames': [s, e-1], 'baseline_delta': delta,
                    'encoded_error': err, 'encoding': why,
                })

    out = bytes(output)
    if len(out) != len(template):
        raise ValueError('Clip size changed unexpectedly.')
    if out[0:0x30] != template[0:0x30]:
        raise ValueError('Animation header changed unexpectedly.')
    new_units, new_tail = E.parse_units(out)
    if new_tail != tail or new_tail != old_tail:
        raise ValueError('Animation trailer boundary changed; refusing to write.')
    new_dec, _ = B.parse_bones(out, frames)
    if len(new_dec) != len(old_dec):
        raise ValueError('Re-decoded record count differs from original.')
    # Ensure all segment byte ranges, channel IDs and frame counts are unchanged.
    for i, (u0,u1) in enumerate(zip(units,new_units)):
        for kind in ('rot','pos'):
            a=[(x['ch'],x['n'],x['size']) for x in u0[kind]['tracks']]
            b=[(x['ch'],x['n'],x['size']) for x in u1[kind]['tracks']]
            if a!=b: raise ValueError(f'Record {i} {kind} layout changed: {a} != {b}')

    return out, dict(frames=frames, original_size=len(template), output_size=len(out),
                     trailer_offset=tail, patched_segments=patched, skipped_segments=skipped,
                     changed_segment_count=changed_seg_count, original_sha256=sha(template), output_sha256=sha(out))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--edited-json', default='reload_edit.json')
    ap.add_argument('--baseline-json', default='reload_baked.json')
    ap.add_argument('--template-anim', default='10_FP_P_S_105_empty_reload_full_bolt_action.anim')
    ap.add_argument('--source-db', default='bg1_pst.db')
    ap.add_argument('--output-db', default='bg1_pst_multitrack_inplace_test.db')
    ap.add_argument('--output-anim', default='10_FP_P_S_105_multitrack_inplace.anim')
    ap.add_argument('--report', default='multitrack_patch_report.json')
    ap.add_argument('--slide-position-mode', choices=('auto','on','off'), default='auto',
                    help='auto applies the legacy slide-position correction only to clip names containing reload; on/off force it.')
    a=ap.parse_args()
    edited=json.loads(Path(a.edited_json).read_text(encoding='utf-8'))
    baseline=json.loads(Path(a.baseline_json).read_text(encoding='utf-8'))
    template_path = Path(a.template_anim)
    template=template_path.read_bytes()
    srcdb=Path(a.source_db).read_bytes()
    if Path(a.output_db).resolve()==Path(a.source_db).resolve(): raise SystemExit('Refusing to overwrite original DB.')
    # Prevent accidentally pairing (for example) a 10-frame reload JSON with a
    # 61-frame movement template, even if frame counts happen to be equal someday.
    json_clip = str(baseline.get('clip', '')).strip()
    if json_clip and Path(json_clip).name.casefold() != template_path.name.casefold():
        raise ValueError(
            f'Clip mismatch: template is "{template_path.name}", but JSON says "{Path(json_clip).name}". '
            'Create baseline/edited JSON from the selected template clip; do not reuse JSON from another animation.'
        )
    anim,info=patch_anim(template,baseline,edited,slide_position_mode=a.slide_position_mode)
    c4,c4end,starts,idx,name=locate_anim_clip(srcdb,template)
    clip_start,clip_end=starts[idx],starts[idx+1]
    if clip_end-clip_start != len(template): raise ValueError('Original DB clip boundary differs from template length.')
    abs_start=c4+clip_start; abs_end=c4+clip_end
    outdb=bytearray(srcdb); outdb[abs_start:abs_end]=anim; outdb=bytes(outdb)
    if len(outdb)!=len(srcdb): raise ValueError('DB size changed unexpectedly.')
    if outdb[:abs_start]!=srcdb[:abs_start] or outdb[abs_end:]!=srcdb[abs_end:]: raise ValueError('Bytes outside target clip changed.')
    if outdb[c4+clip_start:c4+clip_end]!=anim: raise ValueError('Patched DB read-back did not match animation output.')
    Path(a.output_anim).write_bytes(anim)
    Path(a.output_db).write_bytes(outdb)
    Path(a.report).write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(f'Original DB: {len(srcdb):,} bytes; patched DB: {len(outdb):,} bytes; delta: {len(outdb)-len(srcdb):+d}')
    print(f'Clip: {name}, index {idx}, DB bytes {abs_start:#x}..{abs_end:#x}')
    print(f'Frames preserved: {info["frames"]}; animation bytes: {info["original_size"]:,} -> {info["output_size"]:,}')
    print(f'Patched segments: {info["changed_segment_count"]}; skipped segments: {len(info["skipped_segments"])}')
    print('Patched segments by bone:')
    by_bone={}
    for row in info['patched_segments']:
        by_bone.setdefault(row['bone'],[]).append(f'{row["kind"]}:ch{row["channel"]}@{row["frames"][0]}-{row["frames"][1]}')
    for bone, rows in by_bone.items(): print(f'  {bone}: ' + ', '.join(rows))
    print('\nSkipped segments (kept original bytes):')
    for row in info['skipped_segments']:
        err = '' if row['error'] is None else f'; fit/encode error={row["error"]:.4f}'
        print(f'  {row["bone"]} {row["kind"]} ch{row["channel"]} frames {row["frames"][0]}-{row["frames"][1]}: {row["reason"]}{err}')
    print(f'Only clip payload bytes changed; all outer DB offsets unchanged.')
    print(f'Animation SHA-256: {info["original_sha256"]} -> {info["output_sha256"]}')
    print(f'Wrote: {a.output_db}, {a.output_anim}, {a.report}')
    print('WARNING: decoder and exact-size checks passed; game-engine compatibility is not guaranteed. Test in a separate game copy.')

if __name__=='__main__': main()
