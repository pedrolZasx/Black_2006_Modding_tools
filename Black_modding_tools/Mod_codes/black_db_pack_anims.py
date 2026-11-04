#!/usr/bin/env python3
"""BLACK (2006) - pack ALL modded .anim files of a folder into a copy of the original DB, in one run.

  python black_db_pack_anims.py --source-db  <original>.db
                                --mod-anim-dir <Modding_project/pfguns_anim_mod/levels/level_00/bg1_pst>
                                --original-anim-dir <Modding_project/org_pfguns_anim_extracted/levels/level_00/bg1_pst>
                                --output-db <modded>.db

Every .anim in --mod-anim-dir replaces the clip with the same file name in the DB. The build always starts
from the ORIGINAL db, so the result is exactly "original + whatever is in the mod folder" (delete an anim from
the mod folder and that clip goes back to original on the next build).

The DB format is a fixed-size archive, so a modded anim must have exactly the size of the original clip
(the in-place patcher guarantees that). Nothing is written unless every anim passes the checks.
"""
import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

DB_CHUNK_COUNT_OFF = 0x18
DB_CHUNK_OFFS_OFF = 0x1C
DB_ANIM_CHUNK_INDEX = 4
BASE = 0x100
ANIM_COUNT_OFF = 0x124
ANIM_DESC_TABLE_OFF = 0x140
ANIM_DATA_TABLE_OFF = 0x190


def u32(data, off):
    return struct.unpack_from('<I', data, off)[0]


def read_clip_table(db):
    """Read descriptor-indexed clips while tolerating unused/sentinel offset entries.

    Some DBs (including bg1_shg.db) contain placeholder records whose animation
    offset is an uninitialized 0x5f5f605f sentinel. Clip descriptors remain in
    logical order, but valid animation payloads are laid out by increasing byte
    offset. Build a per-record (start, end) span from the sorted valid offsets.
    """
    chunk_count = u32(db, DB_CHUNK_COUNT_OFF)
    outer = [u32(db, DB_CHUNK_OFFS_OFF + 4 * i) for i in range(chunk_count + 1)]
    c4 = outer[DB_ANIM_CHUNK_INDEX]
    greater = [x for x in outer if x > c4]
    if not greater:
        raise ValueError('Cannot locate end of the DB animation chunk.')
    c4end = min(greater)
    chunk = db[c4:c4end]
    n = u32(chunk, ANIM_COUNT_OFF)

    desc = [BASE + u32(chunk, ANIM_DESC_TABLE_OFF + 4 * i) for i in range(n)]
    names = [chunk[o + 0x10:o + 0x28].split(b'\0', 1)[0].decode('ascii', 'replace') for o in desc]

    # A payload starts at BASE + relative_offset. Invalid/uninitialized offsets
    # must not be treated as real clips or used as a boundary for neighboring clips.
    valid = []
    invalid = []
    for i in range(n):
        rel = u32(chunk, ANIM_DATA_TABLE_OFF + 4 * i)
        start = BASE + rel
        if BASE <= start < len(chunk):
            valid.append((start, i))
        else:
            invalid.append((i, names[i], rel))

    if not valid:
        raise ValueError('No valid animation payload offsets found in DB animation chunk.')

    # Payloads are packed in byte order, not necessarily descriptor-table order.
    valid.sort()
    spans = [None] * n
    for j, (start, idx) in enumerate(valid):
        # For duplicate offsets, a zero-length span is ambiguous. Give same-start
        # records the next strictly greater offset as their shared end; clip matching
        # remains protected by filename hints and exact template bytes.
        end = len(chunk)
        for k in range(j + 1, len(valid)):
            if valid[k][0] > start:
                end = valid[k][0]
                break
        spans[idx] = (start, end)

    if invalid:
        print('NOTE: ignored %d invalid/uninitialized animation offset(s): %s' % (
            len(invalid), ', '.join('%s=0x%x' % (name, rel) for _, name, rel in invalid)))
    return c4, chunk, spans, names


def find_clip(chunk, spans, names, file_name, mod_size, template):
    """index of the clip that `file_name` (a modded anim) replaces"""
    n = len(names)
    hint = set(re.findall(r'FP(?:_[A-Za-z]){2}_\d+', file_name))
    by_hint = [i for i in range(n) if names[i] in hint and spans[i] is not None]
    if template is not None:
        exact = [i for i, span in enumerate(spans)
                 if span is not None and chunk[span[0]:span[1]] == template]
        if len(exact) > 1:
            exact = [i for i in exact if i in by_hint]
        if len(exact) == 1:
            return exact[0]
        head = template[:0x30]
        same = [i for i in by_hint
                if spans[i][1] - spans[i][0] == len(template)
                and chunk[spans[i][0]:spans[i][0] + 0x30] == head]
    else:
        same = [i for i in by_hint if spans[i][1] - spans[i][0] == mod_size]
    if len(same) == 1:
        return same[0]
    raise ValueError('cannot identify the clip in the DB (name hint %s, candidates %s)' % (sorted(hint) or 'none', same))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source-db', required=True, help='the ORIGINAL db')
    ap.add_argument('--mod-anim-dir', required=True, help='folder with the modded .anim files')
    ap.add_argument('--original-anim-dir', default='', help='folder with the originally extracted anims (same file names) - used to match clips exactly')
    ap.add_argument('--output-db', required=True)
    ap.add_argument('--report', default='')
    a = ap.parse_args()

    src_path, out_path, mod_dir = Path(a.source_db), Path(a.output_db), Path(a.mod_anim_dir)
    if src_path.resolve() == out_path.resolve():
        raise SystemExit('Refusing to write over the source DB.')
    if not src_path.is_file():
        raise SystemExit('Source DB not found: %s' % src_path)
    anims = sorted(mod_dir.rglob('*.anim')) if mod_dir.is_dir() else []
    if not anims:
        raise SystemExit('No .anim files in %s - patch at least one clip first.' % mod_dir)

    src = src_path.read_bytes()
    c4, chunk, spans, names = read_clip_table(src)
    orig_dir = Path(a.original_anim_dir) if a.original_anim_dir else None
    out = bytearray(src)
    used, rows, errors = {}, [], []
    for f in anims:
        rel = f.relative_to(mod_dir)
        mod = f.read_bytes()
        template = None
        if orig_dir is not None and (orig_dir / rel).is_file():
            template = (orig_dir / rel).read_bytes()
        elif orig_dir is not None:
            print('NOTE: no original for %s in %s - matching by clip name and size only' % (rel, orig_dir))
        try:
            idx = find_clip(chunk, spans, names, f.name, len(mod), template)
        except ValueError as exc:
            errors.append('%s: %s' % (rel, exc))
            continue
        start, end = spans[idx]
        size = end - start
        if len(mod) != size:
            errors.append('%s: size %d differs from the DB clip %s (%d). The archive is fixed-size, so the anim must be patched in place.'
                          % (rel, len(mod), names[idx], size))
            continue
        if idx in used:
            errors.append('%s: same DB clip %s as %s' % (rel, names[idx], used[idx]))
            continue
        used[idx] = str(rel)
        a0 = c4 + start
        changed = bytes(out[a0:a0 + size]) != mod
        out[a0:a0 + size] = mod
        rows.append({'file': str(rel), 'clip': names[idx], 'index': idx, 'db_offset': a0, 'size': size, 'differs_from_original': changed,
                     'sha256': hashlib.sha256(mod).hexdigest()})
    if errors:
        print('NOTHING WRITTEN. Problems:')
        for e in errors:
            print('  -', e)
        sys.exit(1)
    out = bytes(out)
    if len(out) != len(src):
        raise SystemExit('DB size changed unexpectedly.')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_path) + '.tmp')
    tmp.write_bytes(out)
    tmp.replace(out_path)
    print('Packed %d modded clip(s) into %s' % (len(rows), out_path))
    for r in rows:
        print('  %-14s <- %s%s' % (r['clip'], r['file'], '' if r['differs_from_original'] else '   (identical to original)'))
    print('DB size unchanged: %s bytes; only clip payload bytes were replaced.' % format(len(out), ','))
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('WARNING: size/structure checks passed; game compatibility is not guaranteed. Test in a separate game copy.')


if __name__ == '__main__':
    main()
