#!/usr/bin/env python3
"""
black_fps_model_replace_v6.py - Replace the first-person weapon mesh (ANY vertex/triangle count - chunk grows automatically) inside Black (2006, Xbox)
chunk01_skeleton.bin (or the full bg1_xxx.db) with a completely different model (.obj).

Unlike black_fps_model_rebuild.py (which only moves existing vertices), this script:
  * splits the OBJ into game vertices (unique position + uv + normal combos, via the FACE indices)
  * triangulates, checks the vertex / index budget of the target group
  * builds ONE triangle strip (degenerate joins, correct odd-triangle winding)
  * writes new vertices into the weapon group's vertex slots (bone bytes copied from the template)
  * writes the new index buffer (padded with degenerates), recalculates the group bounding box
  * leaves the arms group and everything else byte-identical
  * re-reads the written file and verifies it

Usage:
  python black_fps_model_replace_v6.py <template_chunk01_or_db> <new.obj> <output_chunk01_or_db>
         [--group 0] [--scale 1.0] [--offset X Y Z] [--rotate RX RY RZ]
         [--bone-mode nearest|common] [--bone-hex 24HEXCHARS] [--flip-winding]
         [--dry-run] [--no-verify]

Transform applied to every OBJ position:   p' = Rot(p) * scale + offset
  (Rot = Rz * Ry * Rx, angles in degrees; normals are rotated only)
If the OBJ has no 'vn' normals, smooth normals are computed from the geometry.
"""

import argparse
import heapq
import math
import os
import re
import struct
import sys
import time
import random
from collections import Counter, defaultdict

STRIDE = 36
BONE_OFF = 0x0C
NRM_OFF = 0x18
UV_OFF = 0x1C


# --------------------------------------------------------------------------------------
# Shared format helpers (same as extract / rebuild scripts)
# --------------------------------------------------------------------------------------
def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def pack_normal_nv2a(nx, ny, nz):
    """Pack normal vector into NV2A 11-11-10 format."""
    ix = int(round(max(min(nx, 1.0), -1.0) * 1023.0))
    iy = int(round(max(min(ny, 1.0), -1.0) * 1023.0))
    iz = int(round(max(min(nz, 1.0), -1.0) * 511.0))
    if ix < 0: ix += 0x800
    if iy < 0: iy += 0x800
    if iz < 0: iz += 0x400
    return (ix & 0x7FF) | ((iy & 0x7FF) << 11) | ((iz & 0x3FF) << 22)


def strip_to_tris(indices, num_verts):
    """Decode a u16 triangle strip (odd triangles flipped, degenerates skipped)."""
    tris = []
    for k in range(len(indices) - 2):
        a, b, c = indices[k], indices[k + 1], indices[k + 2]
        if a == b or b == c or a == c or max(a, b, c) >= num_verts:
            continue
        tris.append((a, b, c) if k % 2 == 0 else (b, a, c))
    return tris


def locate_chunk(data, is_db):
    """Return (start, end) of chunk 1 inside data."""
    if not is_db:
        return 0, len(data)
    chunk_count = struct.unpack_from("<I", data, 0x18)[0]
    c1_start = struct.unpack_from("<I", data, 0x1c + 4)[0]
    all_offs = sorted(set([struct.unpack_from("<I", data, 0x1c + i * 4)[0]
                           for i in range(chunk_count + 1)] + [len(data)]))
    idx = all_offs.index(c1_start)
    return c1_start, all_offs[idx + 1]


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def insert_block(c1, at, n, beyond=0, relptrs=None):
    """Insert n zero bytes at offset 'at' of chunk c1 (in place). Fixes the group records'
    relative index offsets and every pointer-like word (raw chunk offsets and 0x91xxxx
    pointers >= 'at') in the header area and in the tail after the last index buffer.
    'beyond' = bytes of the file that follow chunk 1 (later chunks): header words may point there too, and
    those targets move by the FULL growth, so words up to len(c1)+beyond are treated as pointers.
    Returns the list of changed pointer words [(pos, old, new)]."""
    info = parse_chunk(c1)
    L = len(c1)
    vstart = info["vstart"]
    last_end = max(g["index_abs_off"] + 2 * g["index_count"] for g in info["groups"])
    protect = set()
    # object-relative pointers (target = object_base + word) are handled separately below
    for rp in (relptrs or []):
        protect.add(rp[0])
    for g in info["groups"]:
        for k in range(0x20, 0x2C, 4):
            protect.add(g["record_off"] + k)
    changes = []
    for a, b in ((0, vstart), ((last_end + 3) & ~3, L)):
        for o in range((a + 3) & ~3, b - 3, 4):
            if o in protect:
                continue
            w = u32(c1, o)
            if w % 4:
                continue
            if at <= w <= L + beyond:
                changes.append((o, w, w + n))
            elif 0x910000 + at <= w <= 0x910000 + L + beyond:
                changes.append((o, w, w + n))
    for o, w, new in changes:
        struct.pack_into("<I", c1, o, new)
    # object-relative pointers: the target moves by n if it lies at/after the insertion point, but the
    # object itself may move too. The stored word must change by (target shift - object shift).
    for rp in (relptrs or []):
        pos_, base_ = rp
        w = u32(c1, pos_)
        d = (n if base_ + w >= at else 0) - (n if base_ >= at else 0)
        if d:
            struct.pack_into("<I", c1, pos_, w + d)
            changes.append((pos_, w, w + d))
    new_offs = [(g, g["index_abs_off"] + (n if g["index_abs_off"] >= at else 0)) for g in info["groups"]]
    c1[at:at] = bytes(n)
    for rp in (relptrs or []):                       # keep our bookkeeping in step with the data that moved
        if rp[0] >= at: rp[0] += n
        if rp[1] >= at: rp[1] += n
    for g, na in new_offs:
        struct.pack_into("<I", c1, g["record_off"] + 0x28, na - g["record_off"])
    return changes


def find_object_base(c1):
    """Locate the weapon object whose header holds object-relative pointers, straight from the chunk data.
    Signature (seen in bg1_pst.db, object at file 0x12A80): fields +0x38 and +0x64 hold the same pointer,
    all of +0x38/+0x40/+0x48/+0x4C/+0x64 are 4-aligned and land inside the chunk, and the block at +0x4C
    starts with two words that point at block+8 and block+16 (object-relative).
    Returns the list of chunk offsets that match (normally exactly one)."""
    L = len(c1)
    hits = []
    for B in range(0, L - 0x70, 4):
        a = u32(c1, B + 0x38)
        if a == 0 or a % 4 or u32(c1, B + 0x64) != a or not (B < B + a < L):
            continue
        if any((lambda w: w % 4 or not (B < B + w < L))(u32(c1, B + f)) for f in (0x40, 0x48, 0x4C)):
            continue
        blk = B + u32(c1, B + 0x4C)
        if blk + 8 > L:
            continue
        if B + u32(c1, blk) == blk + 8 and B + u32(c1, blk + 4) == blk + 16:
            hits.append(B)
    return hits


def collect_object_pointers(c1, base, fields, nested_field, nested_count):
    """Find the object-relative pointer words of the weapon object at c1-offset 'base'.
    Returns a list of [word_position, base]. Aborts if a word does not look like a valid pointer."""
    L = len(c1)
    out = []

    def ok(w):
        return w % 4 == 0 and 0 < base + w < L

    for f in fields:
        w = u32(c1, base + f)
        if not ok(w):
            raise SystemExit(f"[!] object field +0x{f:x} = 0x{w:x} is not a valid object-relative pointer "
                             f"(wrong --obj-base / --obj-fields?)")
        out.append([base + f, base])
    if nested_field is not None and nested_count:
        blk = base + u32(c1, base + nested_field)
        for k in range(nested_count):
            w = u32(c1, blk + 4 * k)
            if not ok(w):
                raise SystemExit(f"[!] nested pointer word at 0x{blk + 4 * k:x} = 0x{w:x} looks invalid")
            out.append([blk + 4 * k, base])
    return out


def patch_db_table(full, old_c1_end, shift):
    """Shift the .db chunk offset table entries that lie after chunk 1."""
    n = u32(full, 0x18)
    for i in range(n + 1):
        o = 0x1c + 4 * i
        w = u32(full, o)
        if w >= old_c1_end:
            struct.pack_into("<I", full, o, w + shift)


def parse_chunk(c1):
    """Locate geometry groups, vertex buffer start and each group's own vertex slots."""
    container_base = struct.unpack_from("<I", c1, 0x98)[0] if len(c1) > 0xa0 else 0x2480
    if container_base == 0 or container_base >= len(c1):
        container_base = 0x2480

    def _scan(lo, hi):
        found = []
        for i in range(lo, min(hi, len(c1) - 0x2C), 16):
            typ, cnt, off = struct.unpack_from("<3I", c1, i + 0x20)
            if typ == 6 and 100 < cnt < 100000 and 0x1000 < off < len(c1) \
                    and i + off + 2 * cnt <= len(c1):
                found.append({"record_off": i, "index_count": cnt, "index_abs_off": i + off})
        return found

    # 1) the classic window (pistol-style chunks)
    groups = _scan(container_base + 0x1000, container_base + 0x2000)

    # 2) fallback: some chunks (e.g. shotgun) have no container base at 0x98, so the group
    #    records live elsewhere. Scan the whole header area and keep the longest run of
    #    records spaced 0x30 apart (the real group table), ignoring stray false positives.
    if not groups:
        cand = _scan(0, len(c1))
        runs, cur = [], []
        for g in cand:
            if cur and g["record_off"] - cur[-1]["record_off"] == 0x30:
                cur.append(g)
            else:
                if cur: runs.append(cur)
                cur = [g]
        if cur: runs.append(cur)
        if runs:
            groups = max(runs, key=lambda r: (len(r), r[0]["index_count"]))
            container_base = groups[0]["record_off"] - 0x1000 if groups[0]["record_off"] >= 0x1000 else 0
    if not groups:
        raise ValueError("Could not locate geometry group records in chunk")

    vstart = (groups[-1]["record_off"] + 0x30 + 15) & ~15
    total_verts = (groups[0]["index_abs_off"] - vstart) // STRIDE

    # Each later group owns an identical full copy of the vertex buffer that ends exactly where
    # that group's index buffer starts (layout: [vbufA][idx0][pad][vbufB][idx1][pad][vbufC][idx2]...).
    a_size = total_verts * STRIDE
    a_blob = bytes(c1[vstart:vstart + a_size])
    # The vertex buffer may be followed by a few padding bytes before idx0 (gk1: 12 bytes), and the copy
    # is followed by the same padding before its own index buffer, so try that spacing first.
    pad = (groups[0]["index_abs_off"] - vstart) - a_size
    for g_idx, g in enumerate(groups):
        g["vcopy_off"] = None
        if g_idx > 0:
            for b in (g["index_abs_off"] - a_size - pad, g["index_abs_off"] - a_size):
                if b >= vstart + a_size and bytes(c1[b:b + a_size]) == a_blob:
                    g["vcopy_off"] = b
                    break

    names = [s.decode() for s in re.findall(rb"bg1_\w+", bytes(c1[-0x1000:]))]
    for g_idx, g in enumerate(groups):
        ind = struct.unpack_from(f"<{g['index_count']}H", c1, g["index_abs_off"])
        g["indices"] = ind
        g["slots"] = sorted(set(ind))
        g["name"] = names[g_idx] if len(names) >= len(groups) else f"submesh_{g_idx:02d}"
    return {"container_base": container_base, "groups": groups,
            "vstart": vstart, "total_verts": total_verts}


# --------------------------------------------------------------------------------------
# OBJ parsing (FACE indices, not list order)
# --------------------------------------------------------------------------------------
def parse_obj_faces(path):
    """Return list of triangles; each corner = (pos_xyz, uv or None, normal or None) by VALUE."""
    P, T, N = [], [], []
    tris = []
    n_polys = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            if line.startswith("v "):
                p = line.split()
                P.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("vt "):
                p = line.split()
                T.append((float(p[1]), float(p[2])))
            elif line.startswith("vn "):
                p = line.split()
                N.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                corners = []
                for tok in line.split()[1:]:
                    s = tok.split("/")
                    vi = int(s[0]); vi = vi - 1 if vi > 0 else len(P) + vi
                    ti = ni = None
                    if len(s) > 1 and s[1]:
                        ti = int(s[1]); ti = ti - 1 if ti > 0 else len(T) + ti
                    if len(s) > 2 and s[2]:
                        ni = int(s[2]); ni = ni - 1 if ni > 0 else len(N) + ni
                    corners.append((P[vi], T[ti] if ti is not None else None,
                                    N[ni] if ni is not None else None))
                if len(corners) < 3:
                    continue
                n_polys += 1
                for k in range(1, len(corners) - 1):          # fan triangulation
                    tris.append((corners[0], corners[k], corners[k + 1]))
    return tris, n_polys, len(P), len(T), len(N)


def rot_matrix(rx, ry, rz):
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = ((1, 0, 0), (0, cx, -sx), (0, sx, cx))
    Ry = ((cy, 0, sy), (0, 1, 0), (-sy, 0, cy))
    Rz = ((cz, -sz, 0), (sz, cz, 0), (0, 0, 1))

    def mul(A, B):
        return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)) for i in range(3))
    return mul(Rz, mul(Ry, Rx))


def mat_vec(M, v):
    return (M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
            M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
            M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2])


def normalize(v):
    l = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if l < 1e-12:
        return None
    return (v[0] / l, v[1] / l, v[2] / l)


def build_vertices(obj_tris, scale, offset, rot, flip_winding):
    """Split into game vertices: one per unique (position, uv, normal) value combo."""
    R = rot_matrix(*rot)
    key_to_id = {}
    pos, uv, nrm = [], [], []
    tris = []
    dropped = 0

    def vid(corner):
        if corner in key_to_id:
            return key_to_id[corner]
        p, t, n = corner
        q = mat_vec(R, p)
        q = (q[0] * scale + offset[0], q[1] * scale + offset[1], q[2] * scale + offset[2])
        i = len(pos)
        key_to_id[corner] = i
        pos.append(q)
        uv.append(t if t is not None else (0.0, 0.0))
        nrm.append(normalize(mat_vec(R, n)) if n is not None else None)
        return i

    for c0, c1_, c2 in obj_tris:
        a, b, c = vid(c0), vid(c1_), vid(c2)
        if a == b or b == c or a == c:
            dropped += 1                      # would vanish in the strip decoder anyway
            continue
        tris.append((a, c, b) if flip_winding else (a, b, c))

    # smooth normals for vertices that had no vn in the OBJ (welded by position)
    missing = [i for i in range(len(pos)) if nrm[i] is None]
    if missing:
        acc = defaultdict(lambda: [0.0, 0.0, 0.0])
        weld = lambda p: (round(p[0], 5), round(p[1], 5), round(p[2], 5))
        for a, b, c in tris:
            pa, pb, pc = pos[a], pos[b], pos[c]
            e1 = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
            e2 = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
            fn = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
            for v in (a, b, c):
                s = acc[weld(pos[v])]
                s[0] += fn[0]; s[1] += fn[1]; s[2] += fn[2]
        for i in missing:
            nrm[i] = normalize(acc[weld(pos[i])]) or (0.0, 1.0, 0.0)
    return pos, uv, nrm, tris, dropped, len(missing)


# --------------------------------------------------------------------------------------
# Triangle strip builder (greedy, degenerate joins, correct winding for strip_to_tris)
# --------------------------------------------------------------------------------------
def _strip_pass(tris, dir_edges, nb, seed, rnd):
    """One complete stripification pass. Uses 1-index 'swaps' (repeat a vertex to change the turning
    direction) and builds each strip for the exact index parity where it will be placed, so every
    join costs only 2 extra indices."""
    n = len(tris)
    rng = random.Random(seed)
    deg = [len(x) for x in nb]
    vis = [False] * n

    def find_next(s, par, seen):
        u, v = s[-2], s[-1]
        key = (u, v) if (len(s) - 2 + par) % 2 == 0 else (v, u)
        for j in dir_edges.get(key, ()):
            if not vis[j] and j not in seen:
                x, y, z = tris[j]
                w = [q for q in (x, y, z) if q != key[0] and q != key[1]][0]
                return j, w
        return None

    def grow(t, rot, par):
        a, b, c = tris[t]
        x, y, z = (a, b, c) if rot == 0 else ((b, c, a) if rot == 1 else (c, a, b))
        s = [x, y, z] if par == 0 else [y, x, z]
        taken, seen = [t], {t}
        while True:
            r = find_next(s, par, seen)
            if r is None:
                break
            j, w = r
            seen.add(j)
            sn, ss = s + [w], s + [s[-2], w]
            rn, rs = find_next(sn, par, seen), find_next(ss, par, seen)
            if rn is not None and (rs is None or deg[rn[0]] <= deg[rs[0]]):
                s = sn
            elif rs is not None:
                s = ss
            else:
                s = sn
            taken.append(j)
        return s, taken

    heap = [(deg[t] + rng.random() * rnd, t) for t in range(n)]
    heapq.heapify(heap)
    out, nstrips, remaining = [], 0, n
    while remaining:
        while True:
            _, t = heapq.heappop(heap)
            if not vis[t]:
                break
        par = len(out) % 2
        best = None
        for rot in range(3):
            s, tk = grow(t, rot, par)
            sc = len(tk) / (len(s) + 2.0)
            if best is None or sc > best[0]:
                best = (sc, s, tk)
        _, s, tk = best
        for j in tk:
            vis[j] = True
            remaining -= 1
        for j in tk:
            for k in nb[j]:
                if not vis[k]:
                    deg[k] -= 1
                    heapq.heappush(heap, (deg[k] + rng.random() * rnd, k))
        if out:
            out.append(out[-1])          # join: repeat last + first index; strip starts at parity 'par'
            out.append(s[0])
        out += s
        nstrips += 1
    return out, nstrips


def build_strip(tris, max_idx=None, time_budget=10.0):
    """Triangle strip with swaps. Tries several deterministic seeds until it fits max_idx (or time runs out)."""
    n = len(tris)
    dir_edges = defaultdict(list)
    und = defaultdict(list)
    for i, (a, b, c) in enumerate(tris):
        for e in ((a, b), (b, c), (c, a)):
            dir_edges[e].append(i)
            und[(e[0], e[1]) if e[0] < e[1] else (e[1], e[0])].append(i)
    nb = [set() for _ in range(n)]
    for lst in und.values():
        for i in lst:
            for j in lst:
                if i != j:
                    nb[i].add(j)
    t0 = time.time()
    best = None
    seed = 0
    while True:
        out, ns = _strip_pass(tris, dir_edges, nb, seed, 0.0 if seed == 0 else 1.5)
        if best is None or len(out) < len(best[0]):
            best = (out, ns)
        seed += 1
        if max_idx is not None and len(best[0]) <= max_idx:
            break
        if time.time() - t0 > time_budget or seed >= 400:
            break
    return best


# --------------------------------------------------------------------------------------
# Bone data
# --------------------------------------------------------------------------------------
def nearest_bones(src_pts, src_bones, queries):
    """For every query position return the bone bytes of the nearest source vertex (grid search)."""
    n = len(src_pts)
    mn = [min(p[k] for p in src_pts) for k in range(3)]
    mx = [max(p[k] for p in src_pts) for k in range(3)]
    diag = math.dist(mn, mx) or 1.0
    cell = diag / 24.0
    grid = defaultdict(list)
    cof = lambda p: (int(math.floor((p[0] - mn[0]) / cell)), int(math.floor((p[1] - mn[1]) / cell)),
                     int(math.floor((p[2] - mn[2]) / cell)))
    for i, p in enumerate(src_pts):
        grid[cof(p)].append(i)

    res, dsum = [], 0.0
    for q in queries:
        cx, cy, cz = cof(q)
        best, bd = -1, float("inf")
        r = 0
        while r <= 6:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != r:
                            continue
                        for i in grid.get((cx + dx, cy + dy, cz + dz), ()):
                            p = src_pts[i]
                            d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2
                            if d < bd:
                                bd, best = d, i
            if best >= 0 and bd <= (r * cell) ** 2:
                break
            r += 1
        if best < 0 or bd > (6 * cell) ** 2:            # far away from the template -> brute force
            for i, p in enumerate(src_pts):
                d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2
                if d < bd:
                    bd, best = d, i
        dsum += math.sqrt(bd)
        res.append(src_bones[best])
    return res, dsum / max(len(queries), 1), diag


# --------------------------------------------------------------------------------------
# Main replace
# --------------------------------------------------------------------------------------
def replace_model(template_path, obj_path, out_path, group=0, scale=1.0, offset=(0, 0, 0),
                  rotate=(0, 0, 0), bone_mode="nearest", bone_hex=None, flip_winding=False,
                  dry_run=False, verify=True, no_grow=False, obj_base=None,
                  obj_fields=(0x38, 0x40, 0x48, 0x4C, 0x64), obj_nested=2):
    if not os.path.isfile(template_path):
        raise SystemExit(f"[!] Template not found: {template_path}")
    if not os.path.isfile(obj_path):
        raise SystemExit(f"[!] OBJ not found: {obj_path}")
    if scale <= 0:
        raise SystemExit("[!] --scale must be > 0 (use --rotate / --flip-winding to mirror)")

    with open(template_path, "rb") as f:
        full = bytearray(f.read())
    is_db = template_path.lower().endswith(".db")
    c1_start, c1_end = locate_chunk(full, is_db)
    if is_db:
        print(f"[*] Template is a .db archive, Chunk 1 at 0x{c1_start:x} ({c1_end - c1_start:,} bytes)")
    c1 = bytearray(full[c1_start:c1_end])
    template_c1 = bytes(c1)

    info = parse_chunk(c1)
    groups, vstart, total = info["groups"], info["vstart"], info["total_verts"]
    print(f"[*] Chunk: container_base=0x{info['container_base']:x}, {total:,} vertices in buffer")
    for gi, g in enumerate(groups):
        print(f"    Group {gi} ({g['name']}): {len(g['slots']):,} verts (slots {g['slots'][0]}..{g['slots'][-1]}), "
              f"{g['index_count']:,} indices, {len(strip_to_tris(g['indices'], total)):,} tris")
    if not (0 <= group < len(groups)):
        raise SystemExit(f"[!] --group {group} invalid (chunk has {len(groups)} groups)")
    tg = groups[group]
    if "arm" in tg["name"].lower():
        print(f"[!] WARNING: group {group} is named '{tg['name']}' - that looks like the ARMS, not the weapon!")
    for gi, g in enumerate(groups):
        if gi != group and set(g["slots"]) & set(tg["slots"]):
            raise SystemExit(f"[!] Group {group} shares vertex slots with group {gi}; refusing (would corrupt it).")
    slots = tg["slots"]
    max_verts, max_idx = len(slots), tg["index_count"]
    print(f"[+] Target: group {group} '{tg['name']}'  budget = {max_verts:,} vertices, {max_idx:,} indices")

    # ---- 1. parse + split OBJ
    print(f"[*] Loading OBJ: {obj_path}")
    obj_tris, n_polys, nP, nT, nN = parse_obj_faces(obj_path)
    print(f"    {nP:,} v / {nT:,} vt / {nN:,} vn, {n_polys:,} faces -> {len(obj_tris):,} triangles")
    pos, uv, nrm, tris, dropped, n_gen_nrm = build_vertices(obj_tris, scale, offset, rotate, flip_winding)
    if not tris:
        raise SystemExit("[!] OBJ contains no usable triangles.")
    if dropped:
        print(f"[!] Dropped {dropped} degenerate triangles (repeated vertex)")
    if n_gen_nrm:
        print(f"[!] {n_gen_nrm:,} vertices had no vn in the OBJ -> computed smooth normals from geometry")
    if any(t[0] < 0 or t[0] > 1 or t[1] < 0 or t[1] > 1 for t in uv):
        print("[!] Note: some UVs are outside 0..1 (tiling) - stored as-is (V = -vt_v)")
    print(f"[+] Game vertices after splitting (pos+uv+normal): {len(pos):,}   triangles: {len(tris):,}")

    # ---- 2/3. strip + budget
    strip, n_strips = build_strip(tris, max_idx)
    print(f"[+] Triangle strip: {len(strip):,} indices ({n_strips:,} strips joined with degenerates)")
    over_v, over_i = len(pos) - max_verts, len(strip) - max_idx
    tpl_slots = list(slots)
    copies = [(gi, g) for gi, g in enumerate(groups) if g.get("vcopy_off") is not None]
    if copies:
        print("[*] Vertex-buffer copies: " + ", ".join(
            f"group {gi} ({g['name']}) uses its own copy at 0x{g['vcopy_off']:x}" for gi, g in copies) +
            "  -> they are kept identical to the main buffer")
    grown, shift_total, ptr_changes = False, 0, []
    relptrs = []
    base_c1 = None
    if obj_base == "auto":
        hits = find_object_base(c1)
        if len(hits) == 1:
            base_c1 = hits[0]
            print(f"[+] --obj-base auto: found weapon object at chunk offset 0x{base_c1:x} "
                  f"(file offset 0x{base_c1 + (c1_start if is_db else 0):x})")
        else:
            print(f"[!] WARNING: --obj-base auto found {len(hits)} candidates "
                  f"({', '.join(hex(h + (c1_start if is_db else 0)) for h in hits[:5])}); object-relative pointers will NOT be fixed "
                  f"if the chunk grows. Pass --obj-base 0x... explicitly.")
    elif obj_base is not None:
        base_c1 = obj_base - c1_start if is_db else obj_base
    if base_c1 is not None:
        if not (0 <= base_c1 < len(c1)):
            raise SystemExit(f"[!] --obj-base 0x{obj_base:x} is not inside chunk 1")
        relptrs = collect_object_pointers(c1, base_c1, obj_fields, 0x4C if 0x4C in obj_fields else None, obj_nested)
        print(f"[*] Object at chunk offset 0x{base_c1:x}: tracking {len(relptrs)} object-relative pointers")
        rel_orig = [(p, b, u32(c1, p)) for p, b in relptrs]
    # The header holds, for every vertex buffer, a word RELATIVE to the word before it (target = pos - 4 + word).
    # Buffer A is referenced like that at 0x11B4 (0xA0 -> 0x1250) and the arm's copy B at 0x11D0 in the shotgun chunk.
    # v2-v5 never moved the B reference when the chunk grew, so the arm read its vertices from the wrong place.
    vref = []
    for gi, g in copies:
        hits = [p for p in range(4, vstart, 4) if p - 4 + u32(c1, p) == g["vcopy_off"]]
        if hits:
            vref += [(gi, p) for p in hits]
        else:
            print(f"[!] WARNING: no header reference to the vertex copy of group {gi} found - arm may glitch if the chunk grows")
    if vref:
        print("[*] Header vertex-copy references: " + ", ".join(f"group {gi} at 0x{p:x}" for gi, p in vref))
    if (over_v > 0 or over_i > 0) and not no_grow:
        extra_v = (max(0, over_v) + 3) & ~3          # original vertex counts are multiples of 4 (36*V stays 16-aligned)
        if total + extra_v > 65535:
            raise SystemExit("[!] Too many vertices for 16-bit indices (max 65535 in the chunk).")
        dv = extra_v * STRIDE
        n_bufs = (1 + len(copies)) if extra_v else 0
        dv_total = dv * n_bufs
        need = 2 * max(0, over_i)
        n2 = (need + 15) & ~15                       # keep the next vertex copy 16-byte aligned
        while (dv_total + n2) % 0x80:                # chunk size stays a multiple of 0x80
            n2 += 16
        old_cnt = tg["index_count"]
        new_cnt = old_cnt + n2 // 2
        tot = dv_total + n2
        print(f"[+] GROWING chunk: +{extra_v:,} vertices in each of {1 + len(copies)} vertex buffer(s) "
              f"(+{dv_total:,} bytes), index buffer {old_cnt:,} -> {new_cnt:,} (+{n2:,} bytes), "
              f"total growth {tot:,} bytes (0x{tot:x})")
        # insertion points in TEMPLATE coordinates, applied from the back so earlier ones stay valid
        points = []
        if extra_v:
            points.append((groups[0]["index_abs_off"], dv))                 # end of main vertex buffer
            for gi, g in copies:
                points.append((g["index_abs_off"], dv))                     # end of that group's copy
        if n2:
            points.append((tg["index_abs_off"] + 2 * old_cnt, n2))          # end of target index buffer
        for at_, n_ in sorted(points, reverse=True):
            ptr_changes += insert_block(c1, at_, n_, beyond=(len(full) - c1_end) if is_db else 0, relptrs=relptrs)
        struct.pack_into("<I", c1, tg["record_off"] + 0x24, new_cnt)
        ninfo = parse_chunk(c1)
        tg = ninfo["groups"][group]
        groups = ninfo["groups"]
        for gi, p in vref:                            # point the references at the moved copies
            struct.pack_into("<I", c1, p, groups[gi]["vcopy_off"] - (p - 4))
        if vref:
            print("[+] Updated header vertex-copy references: " + ", ".join(
                f"0x{p:x} -> 0x{groups[gi]['vcopy_off'] - (p - 4):x}" for gi, p in vref))
        if relptrs:
            bad = 0
            for (p0, b0, w0), (p1, b1) in zip(rel_orig, relptrs):
                w1 = u32(c1, p1)
                t0, t1 = b0 + w0, b1 + w1
                same = bytes(template_c1[t0:t0 + 8]) == bytes(c1[t1:t1 + 8])
                if not same and p0 < 0x7fffffff:     # the two nested words legitimately differ; only warn about others
                    bad += 1
            print(f"[+] Object-relative pointers rechecked: {len(relptrs) - bad}/{len(relptrs)} point at byte-identical data"
                  f" (a few differ by design if they contain pointers themselves)")
        slots = list(tpl_slots) + list(range(total, total + extra_v))
        max_verts, max_idx = len(slots), new_cnt
        total = ninfo["total_verts"]
        grown, shift_total = True, tot
        over_v = over_i = 0
        print(f"[+] Fixed {len(ptr_changes)} pointer words: " +
              ", ".join(f"0x{o:x}:{w:#x}->{n:#x}" for o, w, n in ptr_changes[:12]) + (" ..." if len(ptr_changes) > 12 else ""))
    if over_v > 0 or over_i > 0:
        print("\n[!] MODEL DOES NOT FIT - nothing written.")
        print(f"    vertices : {len(pos):,} needed, {max_verts:,} available  "
              f"({'OVER by %d (%.0f%%)' % (over_v, 100.0 * over_v / max_verts) if over_v > 0 else 'ok'})")
        print(f"    indices  : {len(strip):,} needed, {max_idx:,} available  "
              f"({'OVER by %d (%.0f%%)' % (over_i, 100.0 * over_i / max_idx) if over_i > 0 else 'ok'})")
        print(f"    triangles: {len(tris):,} (the original weapon has "
              f"{len(strip_to_tris(tg['indices'], total)):,})")
        print("    Reduce the model (Maya: Mesh > Reduce, then re-check UV seams / hard edges, "
              "which each add split vertices) and run again.")
        raise SystemExit(2)

    # ---- 5. bone data (taken from the TEMPLATE, before anything is overwritten)
    tpl_pos = [struct.unpack_from("<3f", c1, vstart + s * STRIDE + 0) for s in tpl_slots]
    tpl_bone = [bytes(c1[vstart + s * STRIDE + BONE_OFF: vstart + s * STRIDE + BONE_OFF + 12]) for s in tpl_slots]
    if bone_hex:
        bh = bytes.fromhex(bone_hex)
        if len(bh) != 12:
            raise SystemExit("[!] --bone-hex needs exactly 12 bytes (24 hex chars)")
        bones = [bh] * len(pos)
        print(f"[+] Bone mode: fixed {bh.hex()}")
    elif bone_mode == "common":
        val, cnt = Counter(tpl_bone).most_common(1)[0]
        bones = [val] * len(pos)
        print(f"[+] Bone mode: common - {val.hex()} (used by {cnt:,}/{len(tpl_bone):,} template weapon verts)")
    else:
        bones, mean_d, diag = nearest_bones(tpl_pos, tpl_bone, pos)
        print(f"[+] Bone mode: nearest template vertex (mean distance {mean_d:.3f}, template weapon size {diag:.3f})")
        if mean_d > 0.25 * diag:
            print("[!] WARNING: new model is far from the original weapon - check --scale/--offset/--rotate, "
                  "or use --bone-mode common")

    # ---- expected (float32/packed) vertex data, used for writing AND verification
    exp_pos = [tuple(f32(c) for c in p) for p in pos]
    exp_uv = [(f32(u), f32(-v)) for (u, v) in uv]
    exp_nw = [pack_normal_nv2a(*n) for n in nrm]

    # ---- 4. write vertices
    for j in range(len(pos)):
        vo = vstart + slots[j] * STRIDE
        struct.pack_into("<3f", c1, vo, *pos[j])
        c1[vo + BONE_OFF: vo + BONE_OFF + 12] = bones[j]
        struct.pack_into("<I", c1, vo + NRM_OFF, exp_nw[j])
        struct.pack_into("<2f", c1, vo + UV_OFF, uv[j][0], -uv[j][1])
    v0 = vstart + slots[0] * STRIDE
    filler = bytes(c1[v0: v0 + STRIDE])
    for s in slots[len(pos):]:                       # unused slots: harmless copy of a valid vertex
        vo = vstart + s * STRIDE
        c1[vo: vo + STRIDE] = filler

    # keep every vertex-buffer copy identical to the main buffer (as in the original file)
    for gi, g in enumerate(groups):
        bo = g.get("vcopy_off")
        if bo is not None:
            c1[bo: bo + total * STRIDE] = c1[vstart: vstart + total * STRIDE]

    # ---- 7. index buffer (+ degenerate padding) and bounding box
    idx = [slots[i] for i in strip]
    idx += [idx[-1]] * (max_idx - len(idx))
    struct.pack_into(f"<{max_idx}H", c1, tg["index_abs_off"], *idx)
    bmax = tuple(max(p[k] for p in exp_pos) for k in range(3))
    bmin = tuple(min(p[k] for p in exp_pos) for k in range(3))
    struct.pack_into("<3f", c1, tg["record_off"], *bmax)
    struct.pack_into("<3f", c1, tg["record_off"] + 0x10, *bmin)
    print(f"[+] Bounding box group {group}: Min=({bmin[0]:.3f}, {bmin[1]:.3f}, {bmin[2]:.3f}) "
          f"Max=({bmax[0]:.3f}, {bmax[1]:.3f}, {bmax[2]:.3f})")

    if dry_run:
        print("[*] --dry-run: model fits, nothing written.")
        return True

    # ---- 9. output
    full[c1_start:c1_end] = c1
    if grown and is_db:
        patch_db_table(full, c1_end, shift_total)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(full)
    print(f"[+] Saved: {out_path} ({len(full):,} bytes)")

    if not verify:
        return True
    expected = {"tris": tris, "pos": exp_pos, "uv": exp_uv, "nw": exp_nw, "bones": bones,
                "bbox": (bmin, bmax)}
    if grown:
        return verify_grown(out_path, is_db, template_path, group, expected, shift_total, len(ptr_changes))
    return verify_output(out_path, is_db, template_path, group, expected)


# --------------------------------------------------------------------------------------
# Verification (re-reads the written file from disk)
# --------------------------------------------------------------------------------------
def canon(tri_keys):
    r = [tuple(tri_keys[i:] + tri_keys[:i]) for i in range(3)]
    return min(r)


def verify_output(out_path, is_db, template_path, group, exp):
    print("\n=== VERIFY (re-reading output) ===")
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
        ok = ok and cond

    with open(template_path, "rb") as f:
        t_full = f.read()
    with open(out_path, "rb") as f:
        o_full = f.read()
    check(len(t_full) == len(o_full), f"file size unchanged ({len(o_full):,} bytes)")
    ts, te = locate_chunk(t_full, is_db)
    os_, oe = locate_chunk(o_full, is_db)
    check((ts, te) == (os_, oe), "chunk location unchanged")
    t1, o1 = t_full[ts:te], o_full[os_:oe]
    check(t_full[:ts] == o_full[:ts] and t_full[te:] == o_full[oe:], "bytes outside chunk 1 identical")

    ti, oi = parse_chunk(bytearray(t1)), parse_chunk(bytearray(o1))
    vstart, total = oi["vstart"], oi["total_verts"]
    check(len(ti["groups"]) == len(oi["groups"]) and vstart == ti["vstart"],
          "group records / vertex buffer layout unchanged")
    g, tg = oi["groups"][group], ti["groups"][group]

    # decode strips back to triangles
    dtris = strip_to_tris(g["indices"], total)
    slot_set = set(tg["slots"])
    check(all(a in slot_set and b in slot_set and c in slot_set for a, b, c in dtris),
          "all decoded triangles reference weapon vertex slots only")
    check(len(dtris) == len(exp["tris"]),
          f"triangle count: decoded {len(dtris):,} vs OBJ {len(exp['tris']):,}")

    def out_corner(slot):
        vo = vstart + slot * STRIDE
        p = struct.unpack_from("<3f", o1, vo)
        nw = struct.unpack_from("<I", o1, vo + NRM_OFF)[0]
        u, v = struct.unpack_from("<2f", o1, vo + UV_OFF)
        return (p, (u, v), nw)

    cache = {}
    def oc(s):
        if s not in cache:
            cache[s] = out_corner(s)
        return cache[s]

    slots = tg["slots"]
    def ec(j):
        return (exp["pos"][j], exp["uv"][j], exp["nw"][j])

    got = Counter(canon([oc(a), oc(b), oc(c)]) for a, b, c in dtris)
    want = Counter(canon([ec(a), ec(b), ec(c)]) for a, b, c in exp["tris"])
    check(got == want, "triangles match OBJ exactly (positions, winding, UV, packed normal)")
    if got != want:
        print(f"      missing {sum((want - got).values())}, unexpected {sum((got - want).values())}")

    pos_got = Counter(tuple(c[0] for c in t) for t in (canon([oc(a), oc(b), oc(c)]) for a, b, c in dtris))
    pos_want = Counter(tuple(c[0] for c in t) for t in (canon([ec(a), ec(b), ec(c)]) for a, b, c in exp["tris"]))
    check(pos_got == pos_want, "vertex positions of all triangles match the OBJ (after transform)")

    # bone data
    used = sorted(set(i for t in dtris for i in t))
    bone_ok = True
    j_of = {s: j for j, s in enumerate(slots)}
    for s in used:
        vo = vstart + s * STRIDE + BONE_OFF
        if bytes(o1[vo:vo + 12]) != exp["bones"][j_of[s]]:
            bone_ok = False
            break
    check(bone_ok, f"bone data written on all {len(used):,} used vertices")

    # bbox
    gm = struct.unpack_from("<3f", o1, g["record_off"])
    gn = struct.unpack_from("<3f", o1, g["record_off"] + 0x10)
    ps = [struct.unpack_from("<3f", o1, vstart + s * STRIDE) for s in used]
    bm = tuple(max(p[k] for p in ps) for k in range(3))
    bn = tuple(min(p[k] for p in ps) for k in range(3))
    check(gm == bm and gn == bn, "bounding box matches vertex data")

    # everything else byte-identical
    allowed = bytearray(len(t1))
    def allow(a, b):
        allowed[a:b] = b"\x01" * (b - a)
    for s in slots:
        allow(vstart + s * STRIDE, vstart + (s + 1) * STRIDE)
    allow(tg["index_abs_off"], tg["index_abs_off"] + 2 * tg["index_count"])
    allow(tg["record_off"], tg["record_off"] + 0x0C)
    allow(tg["record_off"] + 0x10, tg["record_off"] + 0x1C)
    for gg in ti["groups"]:
        if gg.get("vcopy_off") is not None:
            for s in slots:
                allow(gg["vcopy_off"] + s * STRIDE, gg["vcopy_off"] + (s + 1) * STRIDE)
    bad = [i for i in range(len(t1)) if t1[i] != o1[i] and not allowed[i]]
    check(not bad, "no byte changed outside weapon vertex slots / weapon index buffer / weapon bbox"
          + (f" ({len(bad)} stray bytes, first at 0x{bad[0]:x})" if bad else ""))
    for gi, (a, b) in enumerate(zip(ti["groups"], oi["groups"])):
        if gi == group:
            continue
        same = all(t1[vstart + s * STRIDE: vstart + (s + 1) * STRIDE] == o1[vstart + s * STRIDE: vstart + (s + 1) * STRIDE]
                   for s in a["slots"])
        same = same and t1[a["index_abs_off"]: a["index_abs_off"] + 2 * a["index_count"]] == \
            o1[b["index_abs_off"]: b["index_abs_off"] + 2 * b["index_count"]]
        same = same and t1[a["record_off"]: a["record_off"] + 0x30] == o1[b["record_off"]: b["record_off"] + 0x30]
        check(same, f"group {gi} ({a['name']}) vertices, indices and record are byte-identical")

    for gi, (a, b) in enumerate(zip(ti["groups"], oi["groups"])):
        if a.get("vcopy_off") is not None:
            check(b.get("vcopy_off") is not None, f"group {gi}: own vertex copy present and identical to main buffer")
            if gi != group and b.get("vcopy_off") is not None:
                check(all(t1[a["vcopy_off"] + s * STRIDE: a["vcopy_off"] + (s + 1) * STRIDE] ==
                          o1[b["vcopy_off"] + s * STRIDE: b["vcopy_off"] + (s + 1) * STRIDE] for s in a["slots"]),
                      f"group {gi} ({a['name']}) own vertex copy unchanged")
    print(f"\n=== REPORT: {'ALL CHECKS PASSED' if ok else 'VERIFICATION FAILED'} ===")
    print(f"  weapon: {len(dtris):,} triangles, {len(used):,} vertices used of {len(slots):,} slots, "
          f"{g['index_count'] - len(dtris) - 2:,} degenerate/padding index slots")
    return ok


def verify_grown(out_path, is_db, template_path, group, exp, shift, n_ptr):
    print("\n=== VERIFY GROWN FILE (re-reading output) ===")
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
        ok = ok and cond

    t_full = open(template_path, "rb").read()
    o_full = open(out_path, "rb").read()
    ts, te = locate_chunk(t_full, is_db)
    os_, oe = locate_chunk(o_full, is_db)
    check(len(o_full) == len(t_full) + shift, f"file grew by {shift:,} bytes")
    check((oe - os_) == (te - ts) + shift and os_ == ts, "chunk 1 grew by exactly that")
    if is_db:
        n = u32(t_full, 0x18)
        tab_ok = all(u32(o_full, 0x1c + 4 * i) == u32(t_full, 0x1c + 4 * i) + (shift if u32(t_full, 0x1c + 4 * i) >= te else 0)
                     for i in range(n + 1))
        check(tab_ok, "chunk table shifted correctly")
        tail_start = 0x1c + 4 * (n + 1)
        check(t_full[:0x1c] == o_full[:0x1c] and t_full[tail_start:ts] == o_full[tail_start:ts],
              "db header / chunks before chunk 1 identical")
        check(t_full[te:] == o_full[oe:], "all chunks after chunk 1 byte-identical (only moved)")
        check(all(o_full[u32(o_full, 0x1c + 4 * i) % 0x80:][:0] == b"" and u32(o_full, 0x1c + 4 * i) % 0x80 == 0 for i in range(n + 1)),
              "all chunk offsets still 0x80-aligned")
    t1, o1 = t_full[ts:te], o_full[os_:oe]
    ti, oi = parse_chunk(bytearray(t1)), parse_chunk(bytearray(o1))
    check(len(ti["groups"]) == len(oi["groups"]) and ti["vstart"] == oi["vstart"], "group records / vertex start unchanged")
    vstart, total = oi["vstart"], oi["total_verts"]
    g = oi["groups"][group]
    dtris = strip_to_tris(g["indices"], total)
    check(len(dtris) == len(exp["tris"]), f"triangle count: decoded {len(dtris):,} vs OBJ {len(exp['tris']):,}")

    def oc(slot):
        vo = vstart + slot * STRIDE
        return (struct.unpack_from("<3f", o1, vo), tuple(struct.unpack_from("<2f", o1, vo + UV_OFF)),
                struct.unpack_from("<I", o1, vo + NRM_OFF)[0])

    def ec(j):
        return (exp["pos"][j], exp["uv"][j], exp["nw"][j])
    got = Counter(canon([oc(a), oc(b), oc(c)]) for a, b, c in dtris)
    want = Counter(canon([ec(a), ec(b), ec(c)]) for a, b, c in exp["tris"])
    check(got == want, "triangles match OBJ exactly (positions, winding, UV, packed normal)")
    for gi, (a, b) in enumerate(zip(ti["groups"], oi["groups"])):
        if gi == group:
            continue
        same = all(t1[ti["vstart"] + s * STRIDE: ti["vstart"] + (s + 1) * STRIDE] == o1[vstart + s * STRIDE: vstart + (s + 1) * STRIDE]
                   for s in a["slots"]) and a["indices"] == b["indices"]
        check(same, f"group {gi} ({a['name']}) vertices + indices unchanged")
    for gi, (a, b) in enumerate(zip(ti["groups"], oi["groups"])):
        if a.get("vcopy_off") is not None:
            check(b.get("vcopy_off") is not None, f"group {gi}: own vertex copy present and identical to main buffer")
            if gi != group and b.get("vcopy_off") is not None:
                check(all(t1[a["vcopy_off"] + s * STRIDE: a["vcopy_off"] + (s + 1) * STRIDE] ==
                          o1[b["vcopy_off"] + s * STRIDE: b["vcopy_off"] + (s + 1) * STRIDE] for s in a["slots"]),
                      f"group {gi} ({a['name']}) own vertex copy unchanged")
    for gi, a in enumerate(ti["groups"]):
        if a.get("vcopy_off") is None:
            continue
        for p in range(4, ti["vstart"], 4):
            if p - 4 + u32(t1, p) == a["vcopy_off"]:
                b = oi["groups"][gi].get("vcopy_off")
                check(b is not None and p - 4 + u32(o1, p) == b,
                      f"header word 0x{p:x} still points at group {gi}'s vertex copy (now 0x{b:x})" if b else "vertex copy lost")
    t_last = max(x["index_abs_off"] + 2 * x["index_count"] for x in ti["groups"])
    o_last = max(x["index_abs_off"] + 2 * x["index_count"] for x in oi["groups"])
    tt, ot = t1[t_last:], o1[o_last:]
    nd = sum(1 for i in range(0, min(len(tt), len(ot)) - 3, 4) if tt[i:i + 4] != ot[i:i + 4])
    check(len(tt) == len(ot), f"tail after index buffers same length ({len(tt)} bytes), {nd} words changed (pointer fixes)")
    print(f"\n=== REPORT: {'ALL CHECKS PASSED' if ok else 'VERIFICATION FAILED'} ===")
    return ok


def report_weights_json(path):
    """Read the Maya weights file and say what is in it. The game's per-vertex bone bytes are NOT built from it yet:
    the file layout and the 12-byte bone encoding have to be confirmed first, so the build keeps using --bone-mode."""
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[!] --weights-json {path}: cannot read ({e}); continuing without it")
        return
    if isinstance(data, dict):
        desc = ", ".join(f"{k}({type(v).__name__}{'[%d]' % len(v) if hasattr(v, '__len__') else ''})"
                         for k, v in list(data.items())[:8])
    elif isinstance(data, list):
        desc = f"list[{len(data)}]"
    else:
        desc = type(data).__name__
    print(f"[*] Weights file read: {desc}")
    print("[!] Skin weights are NOT applied yet: bone bytes still come from --bone-mode "
          "(nearest template vertex by default). Send me this JSON and the 12-byte bone layout to enable them.")


def main():
    ap = argparse.ArgumentParser(description="Replace the first-person weapon mesh in Black (2006) chunk01_skeleton.bin / .db")
    ap.add_argument("template_chunk", help="Original chunk01_skeleton.bin or bg1_xxx.db")
    ap.add_argument("new_obj", help="New weapon model (.obj, e.g. from Maya)")
    ap.add_argument("out_chunk", help="Output chunk01_skeleton.bin or .db")
    ap.add_argument("--group", type=int, default=0, help="Group to replace (default 0 = weapon)")
    ap.add_argument("--scale", type=float, default=1.0, help="Scale applied to OBJ positions (default 1.0)")
    ap.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                    help="Translation added after scale")
    ap.add_argument("--rotate", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("RX", "RY", "RZ"),
                    help="Rotation in degrees (applied before scale, order X then Y then Z)")
    ap.add_argument("--bone-mode", choices=("nearest", "common"), default="nearest",
                    help="nearest: copy bone bytes from nearest original weapon vertex; "
                         "common: most frequent original bone value on every vertex")
    ap.add_argument("--bone-hex", default=None, help="Explicit 12 bone bytes as 24 hex chars (overrides --bone-mode)")
    ap.add_argument("--flip-winding", action="store_true", help="Reverse triangle winding")
    ap.add_argument("--no-grow", action="store_true", help="Old behaviour: refuse models that exceed the original budget")
    ap.add_argument("--dry-run", action="store_true", help="Only check that the model fits; write nothing")
    ap.add_argument("--no-verify", action="store_true", help="Skip post-write verification")
    ap.add_argument("--obj-base", type=lambda x: None if x.lower() == "none" else ("auto" if x.lower() == "auto" else int(x, 0)),
                    default="auto",
                    help="File offset of the weapon object whose header holds OBJECT-RELATIVE pointers "
                         "(e.g. 0x12A80 for bg1_pst.db), or 'auto' (default: detect it from the db), or 'none'. Without it those pointers are NOT fixed when the chunk grows.")
    ap.add_argument("--obj-fields", type=lambda x: [int(v, 0) for v in x.split(",")],
                    default=[0x38, 0x40, 0x48, 0x4C, 0x64], help="Pointer field offsets inside the object")
    ap.add_argument("--obj-nested", type=int, default=2,
                    help="Number of pointer words at the start of the block pointed to by field +0x4C")
    ap.add_argument("--weights-json", default=None,
                    help="Skin weights exported by the Maya tool. Accepted so the Maya tool can call this script; "
                         "currently only inspected - bone bytes still come from --bone-mode (see log)")
    a = ap.parse_args()
    if a.weights_json:
        report_weights_json(a.weights_json)
    ok = replace_model(a.template_chunk, a.new_obj, a.out_chunk, a.group, a.scale, tuple(a.offset),
                       tuple(a.rotate), a.bone_mode, a.bone_hex, a.flip_winding, a.dry_run, not a.no_verify, a.no_grow,
                       a.obj_base, tuple(a.obj_fields), a.obj_nested)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()