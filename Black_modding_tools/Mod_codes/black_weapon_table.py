#!/usr/bin/env python3
"""black_weapon_table.py - infer the CLIP rig of any Black (2006, Xbox) weapon: bone parents + the .anim record order.

  python black_weapon_table.py bg1_bns.db                 # print the table, write bg1_bns.table.json next to it
  python black_weapon_table.py bg1_hvy.db --out hvy.json
  python black_weapon_table.py bg1_pst.db --check         # compare with the verified pistol / AK tables

Why: every clip of a weapon stores its bones in ONE fixed order (level order of the clip hierarchy), but the order differs per weapon
(pistol 44 records, AK 42, bns 43, heavy 41 ...). The old tools only had tables for pistol and AK and guessed the rest per clip, which puts
gun parts on the wrong bones. Here the order is derived from ALL clips of the .db at once: record i's median translation must equal the
bind-pose offset of the bone assigned to it (relative to that bone's clip parent); parents of non-hand bones (gun parts) are searched the
same way; level order (parent before child, depth never decreasing) breaks ties.

The result is stored in the baked JSON ("records" + per-bone "parent"), and black_anim_build.py / the patcher read it from there.
"""
import argparse, json, os, struct, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import black_anim_bake_v5_3 as B


def db_clips(db):
    """[(name, anim_bytes)] of the animation chunk (chunk 4) of a .db"""
    u = lambda o: struct.unpack_from('<I', db, o)[0]
    cc = u(0x18); offs = [u(0x1C + 4 * i) for i in range(cc + 1)]
    c4 = offs[4]; end = min(x for x in offs if x > c4); ch = db[c4:end]
    n = struct.unpack_from('<I', ch, 0x124)[0]
    raw = [0x100 + struct.unpack_from('<I', ch, 0x190 + 4 * i)[0] for i in range(n)]
    ds = [0x100 + struct.unpack_from('<I', ch, 0x140 + 4 * i)[0] for i in range(n)]
    valid = [i for i in range(n) if 0x100 <= raw[i] < len(ch)]               # unused slots hold filler (0x5f5f5f5f); table order need not be storage order
    starts = sorted({raw[i] for i in valid}) + [len(ch)]
    nxt = {s: starts[k + 1] for k, s in enumerate(starts[:-1])}
    return [(ch[ds[i] + 0x10:ds[i] + 0x28].split(b'\0', 1)[0].decode('ascii', 'replace'), ch[raw[i]:nxt[raw[i]]]) for i in valid]


def clip_records(db):
    """decoded record lists of every clip with the most common record count"""
    out = []
    for nm, d in db_clips(db):
        if len(d) < 0x20: continue
        try:
            fr = struct.unpack_from('<I', d, 0x14)[0]; bones, _ = B.parse_bones(d, fr)
        except Exception: continue
        if bones: out.append((nm, bones))
    if not out: raise SystemExit('[!] no decodable clips in this .db')
    counts = {}
    for _, b in out: counts[len(b)] = counts.get(len(b), 0) + 1
    N = max(counts, key=counts.get)
    return N, [(nm, b) for nm, b in out if len(b) == N]


HAND = B.FINGER
FIXED_PARENT = {'Extra': 'World', 'Root': 'Extra', 'Shoulder_lft': 'Root', 'Shoulder_rt': 'Root', 'Elbow_lft': 'Shoulder_lft', 'Elbow_rt': 'Shoulder_rt',
                'Wrist_lft': 'Elbow_lft', 'Joint_wrist_twist': 'Elbow_lft', 'Wrist_rt': 'Elbow_rt'}


def fixed_parent(n, names):
    """parent that follows from the name alone (arm / hand chain, weapon root); None = must be searched (gun parts)"""
    nm = set(names)
    if n in FIXED_PARENT: return FIXED_PARENT[n]
    m = HAND.match(n)
    if m:
        k, part, side = m.groups(); return f'Wrist_{side}' if part == 'Top' else f'{k}_{"Top" if part == "Mid" else "Mid"}_{side}'
    if n.endswith('_Root') and n != 'Root': return 'Wrist_rt'
    return None


def infer(db_path, verbose=True):
    names, world = B.skeleton_from_chunk(db_path)
    idx = {n: i for i, n in enumerate(names)}
    db = open(db_path, 'rb').read()
    N, clips = clip_records(db)
    pos = []                                                         # per record: first-frame translations (cm) of every clip that has a translation track
    for i in range(N):
        ps = [b[i]['pos'][0] * 100 for _, b in clips if b[i]['pos'] is not None]
        pos.append(np.array(ps) if ps else None)

    def rest_off(b, q):
        return (np.linalg.inv(world[idx[q]]) @ world[idx[b]])[:3, 3] if q else world[idx[b]][:3, 3]

    cache = {}
    def cost(i, b, q):
        k = (i, b, q)
        if k not in cache:
            if pos[i] is None: cache[k] = 0.3                         # no translation track in any clip: neutral
            else:                                                     # bones that move still sit at their rest offset in some clips -> lower quartile
                cache[k] = float(np.percentile(np.linalg.norm(pos[i] - rest_off(b, q), axis=1), 25))
        return cache[k]

    weapon_root = next((n for n in names if n.endswith('_Root') and n != 'Root'), None)
    # arm chain: same order in every weapon (verified AK / pistol / bns / heavy); its clip offsets differ from the FBX bind pose, so no search here
    prefix = [n for n in ('World', 'Extra', 'Root', 'Shoulder_lft', 'Shoulder_rt', 'Elbow_lft', 'Elbow_rt', 'Wrist_lft', 'Joint_wrist_twist', 'Wrist_rt') if n in idx]
    st0 = dict(rec=list(prefix), par={n: fixed_parent(n, names) for n in prefix}, dep={}, cost=0.0)
    for n in prefix: st0['dep'][n] = 0 if st0['par'][n] is None else st0['dep'][st0['par'][n]] + 1
    def static_depth(b):                                               # depth of arm/hand bones follows from the names; gun parts: unknown (None)
        fp = fixed_parent(b, names)
        if b == names[0]: return 0
        if fp is None or fp not in idx: return None
        d = static_depth(fp)
        return None if d is None else d + 1
    sdep = {b: static_depth(b) for b in names}
    weapon_kids = {n for n in names if fixed_parent(n, names) is None and n not in FIXED_PARENT and n != names[0]}
    BEAM = 400; beam = [st0]
    for i in range(len(prefix), N):
        nxt = {}
        for stt in beam:
            last = stt['dep'][stt['rec'][-1]]
            for b in names:
                if b in stt['par']: continue
                fp = fixed_parent(b, names)
                cps = [fp] if fp else [q for q in stt['par'] if q == weapon_root or fixed_parent(q, names) is None and q in weapon_kids]
                for q in cps:
                    if q not in stt['par']: continue
                    d = stt['dep'][q] + 1
                    if d < last: continue                             # level order: depth never decreases
                    c = stt['cost'] + cost(i, b, q) + (0.0 if (fp or q == weapon_root) else 0.05)
                    ns = dict(rec=stt['rec'] + [b], par={**stt['par'], b: q}, dep={**stt['dep'], b: d}, cost=c)
                    # prune: enough bones must remain that can still sit at depth >= d (else the sequence dead-ends later)
                    if sum(1 for x in names if x not in ns['par'] and ((sdep[x] >= d) if sdep[x] is not None else (weapon_root in ns['par'] or (weapon_root and sdep[weapon_root] >= d)))) < N - len(ns['rec']): continue
                    key = (frozenset(ns['par'].items()), d)
                    old = nxt.get(key)
                    if old is None or (round(c, 3), [idx[x] for x in ns['rec']]) < (round(old['cost'], 3), [idx[x] for x in old['rec']]): nxt[key] = ns
        if not nxt: raise SystemExit(f'[!] record {i}: no candidate bone (clip rig differs from the skeleton)')
        beam = sorted(nxt.values(), key=lambda t: (round(t['cost'], 3), [idx[x] for x in t['rec']]))[:BEAM]
    best = beam[0]; rec, parent = best['rec'], best['par']
    if verbose:
        for i, b in enumerate(rec):
            if i >= len(prefix) and pos[i] is not None and cost(i, b, parent[b]) > 3:
                print(f'   note: record {i} -> {b} (parent {parent[b]}) offset error {cost(i, b, parent[b]):.1f} cm (animated bone or uncertain)')
    free = [n for n in names if n not in parent]
    return dict(weapon=os.path.splitext(os.path.basename(db_path))[0], records=rec, parents={n: parent.get(n) for n in rec}, unrecorded=free,
                n_records=N, n_clips=len(clips))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('db'); ap.add_argument('--out'); ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    t = infer(a.db)
    print(f"{t['weapon']}: {t['n_records']} records, {t['n_clips']} clips used, bones without a record (follow parent): {t['unrecorded'] or '-'}")
    for i, n in enumerate(t['records']): print(f'  {i:2d} {n:24s} parent {t["parents"][n]}')
    if a.check:
        ref = {len(B.AK_RECS): B.AK_RECS, len(B.PISTOL_RECS): B.PISTOL_RECS}.get(t['n_records'])
        if ref is None: print('no verified table for this record count')
        else:
            bad = [(i, x, y) for i, (x, y) in enumerate(zip(ref, t['records'])) if x != y]
            print('MATCHES the verified table' if not bad else f'DIFFERS from the verified table at {len(bad)} records: {bad}')
    out = a.out or os.path.splitext(a.db)[0] + '.table.json'
    json.dump(t, open(out, 'w'), indent=1); print('->', out)

if __name__ == '__main__':
    main()
