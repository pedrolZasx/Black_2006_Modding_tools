#!/usr/bin/env python3
"""
black_fps_anim_tool.py - extract / label / rebuild the first-person animation chunk (chunk 4) of a Black (2006) weapon .db

  python black_fps_anim_tool.py extract  bg1_shg.db [out_dir]      # one NN_<ID>_<label>.anim per clip + anim_manifest.json
  python black_fps_anim_tool.py labels   bg1_shg.db [--write]      # show (and with --write add) label suggestions for this weapon in anim_labels.json
  python black_fps_anim_tool.py rebuild  <dir_with_manifest> <out.db or chunk.bin>

WHAT THE NAMES MEAN
  The .db stores only a clip ID per clip (FP_SG_S_050).  What a clip DOES depends on the weapon: the walk cycle is 110 on the pistol,
  111 on the AK, 102 on bns, 104 on heavy; the reload is 105 / 105 / 110 / 111.  So labels come from a per-weapon table,
  anim_labels.json (next to this script):

      { "bg1_pst": {"105": "reload", "110": "walk"},  "bg1_shg": {"050": "reload"} }      key = .db file name without extension, id = number in the clip ID

  Lookup order for the label in a file name:   1) anim_labels.json   2) a guess from the events inside the clip (shown as guess-<what>)   3) nothing.
  Edit anim_labels.json by hand whenever you find out what a clip really is.  Run `labels <db>` to get a starting list for a new weapon.
  Re-extracting renames files: re-bake JSON from the new file names (the patcher checks that the JSON clip name matches the template file name).

FIXES vs the old tool
  * unused table slots (filler 0x5f5f5f5f instead of an offset) and clips stored in a different order than the table no longer produce empty / wrong files
  * 52-byte stub clips are written but labelled 'stub' (there is nothing to bake in them)
  * rebuild keeps unused slots untouched, keeps storage order, and refuses to patch a .db when the chunk size would change
"""
import argparse, json, os, re, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS_FILE = os.path.join(HERE, 'anim_labels.json')
BASE = 0x100
COUNT_OFF, DESC_OFF, DATA_OFF = 0x124, 0x140, 0x190


# ------------------------------------------------------------------ labels
def load_labels(path=None):
    path = path or LABELS_FILE
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    return {}

def clip_id(name):
    m = re.search(r'_S_(\d+)', name) or re.search(r'(\d+)\s*$', name)
    return m.group(1) if m else name

def safe(s): return re.sub(r'[^A-Za-z0-9_\-]+', '_', s).strip('_')

def clip_events(payload):
    """event names stored in the clip (uses the baker's parser when it is next to this script), else []"""
    try:
        sys.path.insert(0, HERE); import black_anim_bake_v5_3 as B
        fr = struct.unpack_from('<I', payload, 0x14)[0]
        bones, tail = B.parse_bones(payload, fr)
        return [e for e in B.trailer_events(payload, tail) if set(e) - set('_')]
    except Exception:
        return [x.decode() for x in re.findall(rb'(?:animtrigger:[a-z]+|[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)', payload) if x.startswith(b'animtrigger')]

def guess_from_events(ev):
    s = ' '.join(ev).lower()
    if 'reload' in s or re.search(r'(ld\d|unld|mgnld|mgnuld)', s): return 'reload'
    if 'walk1' in s or 'ft__' in s: return 'walk'
    if 'silencer' in s: return 'silencer'
    if 'grenade' in s: return 'grenade'
    if 'melee' in s: return 'melee'
    if 'chngwpn' in s: return 'equip_or_holster'
    if 'fireselect' in s: return 'fire_select'
    if 'cock' in s: return 'cock'
    if 'shell' in s or 'eject' in s: return 'fire'
    if 'stumble' in s or 'flip' in s: return 'stumble_flip'
    return ''


# ------------------------------------------------------------------ chunk 4 access
def get_chunk4(path):
    data = open(path, 'rb').read(); c4_off = 0
    if path.lower().endswith('.db'):
        n = struct.unpack_from('<I', data, 0x18)[0]
        c4_off = struct.unpack_from('<I', data, 0x1C + 16)[0]
        offs = sorted(set([struct.unpack_from('<I', data, 0x1C + 4 * i)[0] for i in range(n + 1)] + [len(data)]))
        data = data[c4_off:offs[offs.index(c4_off) + 1]]
    return data, c4_off

def read_table(chunk):
    n = struct.unpack_from('<I', chunk, COUNT_OFF)[0]
    raw = [struct.unpack_from('<I', chunk, DATA_OFF + 4 * i)[0] + BASE for i in range(n)]
    desc = [struct.unpack_from('<I', chunk, DESC_OFF + 4 * i)[0] + BASE for i in range(n)]
    valid = [BASE <= x < len(chunk) for x in raw]
    # a clip ends where the NEXT STORED clip starts (table order need not be storage order; unused slots hold filler)
    starts = sorted({x for x, v in zip(raw, valid) if v}) + [len(chunk)]
    nxt = {s: starts[k + 1] for k, s in enumerate(starts[:-1])}
    return n, raw, desc, valid, nxt

def clip_name(chunk, doff, i):
    if not (0 <= doff < len(chunk) - 0x28): return f'clip_{i:02d}'
    return chunk[doff + 0x10:doff + 0x28].split(b'\0')[0].decode('ascii', 'ignore') or f'clip_{i:02d}'

def is_stub(payload):
    if len(payload) < 0x100: return True
    return struct.unpack_from('<I', payload, 0x14)[0] > 5000


# ------------------------------------------------------------------ extract
def extract_animations(input_path, out_dir=None, weapon=None, labels_path=None):
    if not os.path.isfile(input_path): raise FileNotFoundError(f'Input file not found: {input_path}')
    chunk, c4_off = get_chunk4(input_path)
    if c4_off: print(f'[*] Chunk 4 of the .db at 0x{c4_off:x}, {len(chunk):,} bytes')
    weapon = (weapon or os.path.splitext(os.path.basename(input_path))[0]).lower()
    table = load_labels(labels_path).get(weapon, {})
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(input_path)), os.path.splitext(os.path.basename(input_path))[0] + '_anims')
    os.makedirs(out_dir, exist_ok=True)
    n, raw, desc, valid, nxt = read_table(chunk)
    print(f'[*] {n} table slots; labels for "{weapon}": {len(table)} in anim_labels.json')
    clips = []
    for i in range(n):
        name = clip_name(chunk, desc[i], i); cid = clip_id(name)
        meta = dict(index=i, name=name, id=cid, desc_offset_rel=desc[i] - BASE, raw_table_value=raw[i] - BASE)
        if not valid[i]:
            meta.update(unused=True, filename=None, label='', label_source='unused', data_offset=None, data_size=0)
            clips.append(meta); print(f'    [{i:02d}] {name:16s} | UNUSED slot (no clip data)'); continue
        payload = chunk[raw[i]:nxt[raw[i]]]
        if is_stub(payload): label, src = 'stub', 'stub'
        elif cid in table: label, src = table[cid], 'table'
        else:
            g = guess_from_events(clip_events(payload)); label, src = ('guess-' + g, 'guess') if g else ('', 'none')
        filename = f'{i:02d}_{name}' + (f'_{safe(label)}' if label else '') + '.anim'
        open(os.path.join(out_dir, filename), 'wb').write(payload)
        meta.update(unused=False, filename=filename, label=label, label_source=src, description=label, data_offset=raw[i], data_size=len(payload))
        clips.append(meta); print(f'    [{i:02d}] {name:16s} | {len(payload):7,d} bytes | {label or "-":20s} ({src})')
    manifest = dict(format='Criterion_Black_FPGun_Animations', version='2.0', weapon=weapon, archive_size=len(chunk),
                    header_hex=chunk[:min(x for x, v in zip(raw, valid) if v)].hex(), num_clips=n, clips=clips)
    mp = os.path.join(out_dir, 'anim_manifest.json'); json.dump(manifest, open(mp, 'w', encoding='utf-8'), indent=2)
    nreal = sum(1 for c in clips if c['filename'] and c['label_source'] != 'stub'); nstub = sum(1 for c in clips if c['label_source'] == 'stub'); nun = sum(1 for c in clips if c['unused'])
    print(f'[+] {n} table slots = {nreal} REAL clips + {nstub} stubs (placeholders, nothing to bake) + {nun} unused slots (no file written)')
    print(f'[+] {nreal + nstub} files written to {out_dir}\n[+] Manifest: {mp}')
    return out_dir


# ------------------------------------------------------------------ labels command
def suggest_labels(input_path, write=False, labels_path=None):
    chunk, _ = get_chunk4(input_path); weapon = os.path.splitext(os.path.basename(input_path))[0].lower()
    labels_path = labels_path or LABELS_FILE; allv = load_labels(labels_path); cur = allv.get(weapon, {})
    n, raw, desc, valid, nxt = read_table(chunk); new = {}
    print(f'{weapon}: suggestions from the events inside each clip (edit {os.path.basename(labels_path)} to correct them)')
    for i in range(n):
        name = clip_name(chunk, desc[i], i); cid = clip_id(name)
        if not valid[i]: print(f'  {cid:>4} {name:14s} unused slot'); continue
        p = chunk[raw[i]:nxt[raw[i]]]
        if is_stub(p): print(f'  {cid:>4} {name:14s} stub - nothing to bake'); continue
        g = guess_from_events(clip_events(p)); fr = struct.unpack_from('<I', p, 0x14)[0]
        print(f'  {cid:>4} {name:14s} {len(p):7,d} B {fr:4d} fr  {("already: " + cur[cid]) if cid in cur else (g or "(no events - unknown)")}')
        if g and cid not in cur: new[cid] = g
    if write and new:
        allv.setdefault(weapon, {}).update(new); json.dump(allv, open(labels_path, 'w', encoding='utf-8'), indent=2, sort_keys=True)
        print(f'[+] added {len(new)} entries to {labels_path}')
    elif write: print('[=] nothing to add')


# ------------------------------------------------------------------ rebuild
def rebuild_animations(manifest_dir, out_path):
    mp = os.path.join(manifest_dir, 'anim_manifest.json')
    if not os.path.isfile(mp): raise FileNotFoundError(f'Manifest not found: {mp}')
    m = json.load(open(mp, encoding='utf-8')); header = bytearray.fromhex(m['header_hex'])
    print(f'[*] Rebuilding chunk 4 from {manifest_dir}')
    real = [c for c in m['clips'] if not c.get('unused') and c.get('filename')]
    by_start = {}
    for c in real: by_start.setdefault(c['data_offset'], c)          # slots that share data share one payload
    order = sorted(by_start)                                          # original STORAGE order
    pos = len(header); new_off = {}; bufs = []
    for s in order:
        c = by_start[s]; p = os.path.join(manifest_dir, c['filename'])
        if not os.path.isfile(p): raise FileNotFoundError(f'Missing animation clip file: {c["filename"]}')
        d = open(p, 'rb').read(); new_off[s] = pos; pos += len(d); bufs.append(d)
    for c in real: struct.pack_into('<I', header, DATA_OFF + 4 * c['index'], new_off[c['data_offset']] - BASE)
    chunk = bytes(header) + b''.join(bufs)                            # unused slots keep their original filler in the header
    if out_path.lower().endswith('.db') and os.path.isfile(out_path):
        db = bytearray(open(out_path, 'rb').read()); n = struct.unpack_from('<I', db, 0x18)[0]
        c4 = struct.unpack_from('<I', db, 0x1C + 16)[0]
        offs = sorted(set([struct.unpack_from('<I', db, 0x1C + 4 * i)[0] for i in range(n + 1)] + [len(db)])); c4e = offs[offs.index(c4) + 1]
        if len(chunk) != c4e - c4:
            raise SystemExit(f'[!] rebuilt chunk is {len(chunk):,} bytes but the chunk in the .db is {c4e - c4:,}: the outer chunk offsets of the .db would be wrong. '
                             'Keep every clip the same size (the patcher does this), or write a chunk .bin instead.')
        db[c4:c4e] = chunk; open(out_path, 'wb').write(db); print(f'[+] Updated .db: {out_path}')
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True); open(out_path, 'wb').write(chunk)
        print(f'[+] Wrote chunk: {out_path} ({len(chunk):,} bytes)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract'); e.add_argument('input_path'); e.add_argument('out_dir', nargs='?'); e.add_argument('--weapon', help='table key if the input is not named bg1_xxx.db'); e.add_argument('--labels', help='labels json (default anim_labels.json next to the script)')
    l = sub.add_parser('labels'); l.add_argument('input_path'); l.add_argument('--write', action='store_true'); l.add_argument('--labels')
    r = sub.add_parser('rebuild'); r.add_argument('manifest_dir'); r.add_argument('out_chunk')
    a = ap.parse_args()
    if a.command == 'extract': extract_animations(a.input_path, a.out_dir, a.weapon, a.labels)
    elif a.command == 'labels': suggest_labels(a.input_path, a.write, a.labels)
    else: rebuild_animations(a.manifest_dir, a.out_chunk)

if __name__ == '__main__':
    main()
