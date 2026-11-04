#!/usr/bin/env python3
"""Convert a BLACK fpguns .db model into the armory viewer's GMSH v1 format.

The converter detects the primary model descriptor and DXT5 dimensions for the
known weapon DB layout. Explicit offset options are available for other variants.

Example (known pistol):
  python tools/db_to_gmesh.py bg1_pst.db bg1_pst.gmesh --known-pistol

Example (another weapon; offsets are hexadecimal):
  python tools/db_to_gmesh.py bg1_snr.db bg1_snr.gmesh \
    --vertex-offset 0x... --vertex-count ... \
    --index-offset 0x... --index-count ... \
    --texture-offset 0x200 --width ... --height ...

Vertices are 36 bytes: float3 position at +0, packed 11:11:10 normal at +24,
and float2 UV at +28. Indices are 16-bit triangle-strip indices. The texture
payload is Xbox DXT5 (BC3), already linear/unswizzled, starting at the given
texture offset. Output GMSH v2 preserves the decoded source normals.



COMANDS ALL PATCH FOLDER 

cd "C:\Users\workhard\Desktop\mod_gbt\REPO"
Get-ChildItem "R:\black\game\levels\level_00\fpguns" -Filter "bg1_*.db" | ForEach-Object {
    $output = Join-Path $_.DirectoryName ($_.BaseName + ".gmesh")
    py -3 tools\db_to_gmesh.py $_.FullName $output --auto --force
}




single 


python tools\db_to_gmesh.py "R:\black\game\levels\level_00\fpguns\bg1_ak1.db" "R:\black\work\run\armory\models\bg1_ak1.gmesh" --auto --force














"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path


MAX_VERTICES = 65535
MAX_INDICES = 300000


def u16s(data: bytes, offset: int, count: int) -> list[int]:
    end = offset + count * 2
    if offset < 0 or end > len(data):
        raise ValueError(f"index range {offset:#x}..{end:#x} is outside the DB")
    return list(struct.unpack_from(f"<{count}H", data, offset))


def read_vertices(data: bytes, offset: int, count: int, position_field: int = 0) -> tuple[list[tuple[float, float, float, float, float]], tuple[float, float, float, float, float, float]]:
    end = offset + count * 36
    if offset < 0 or end > len(data):
        raise ValueError(f"vertex range {offset:#x}..{end:#x} is outside the DB")
    verts: list[tuple[float, float, float, float, float]] = []
    mins = [math.inf] * 3
    maxs = [-math.inf] * 3
    for i in range(count):
        p = offset + i * 36
        x, y, z = struct.unpack_from("<3f", data, p + position_field)
        u, v = struct.unpack_from("<2f", data, p + 28)
        if not all(math.isfinite(q) for q in (x, y, z, u, v)):
            raise ValueError(f"non-finite position/UV at vertex {i} (DB offset {p:#x})")
        if max(abs(x), abs(y), abs(z), abs(u), abs(v)) > 10000:
            raise ValueError(f"implausible position/UV at vertex {i} (DB offset {p:#x})")
        verts.append((x, y, z, u, v))
        for k, q in enumerate((x, y, z)):
            mins[k] = min(mins[k], q)
            maxs[k] = max(maxs[k], q)
    return verts, (mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2])


def strip_to_triangles(strip: list[int], vertex_count: int) -> list[int]:
    out: list[int] = []
    run: list[int] = []
    for idx in strip:
        # Xbox strip buffers commonly use 0xffff to restart a strip.
        if idx == 0xffff:
            run.clear()
            continue
        if idx >= vertex_count:
            raise ValueError(f"index {idx} exceeds vertex count {vertex_count}")
        run.append(idx)
        if len(run) < 3:
            continue
        a, b, c = run[-3:]
        if a == b or b == c or a == c:  # degenerate connector triangle
            continue
        # Triangle-strip winding alternates on every triangle.
        if (len(run) - 3) & 1:
            a, b = b, a
        out.extend((a, b, c))
    if not out:
        raise ValueError("no non-degenerate triangles found in index strip")
    return out


def rgb565(v: int) -> tuple[int, int, int]:
    r = (v >> 11) & 31
    g = (v >> 5) & 63
    b = v & 31
    return ((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2))


def decode_dxt5(data: bytes, offset: int, width: int, height: int) -> bytes:
    if width <= 0 or height <= 0 or width > 1024 or height > 1024:
        raise ValueError("texture dimensions must be between 1 and 1024")
    blocks_x, blocks_y = (width + 3) // 4, (height + 3) // 4
    size = blocks_x * blocks_y * 16
    if offset < 0 or offset + size > len(data):
        raise ValueError(f"DXT5 range {offset:#x}..{offset + size:#x} is outside the DB")
    rgba = bytearray(width * height * 4)
    p = offset
    for by in range(blocks_y):
        for bx in range(blocks_x):
            a0, a1 = data[p], data[p + 1]
            alpha_bits = int.from_bytes(data[p + 2:p + 8], "little")
            c0, c1, color_bits = struct.unpack_from("<HHI", data, p + 8)
            p += 16
            alphas = [a0, a1]
            if a0 > a1:
                alphas.extend(((7 - i) * a0 + i * a1) // 7 for i in range(1, 7))
            else:
                alphas.extend(((5 - i) * a0 + i * a1) // 5 for i in range(1, 5))
                alphas.extend((0, 255))
            colors = [rgb565(c0), rgb565(c1)]
            colors.extend(tuple((2 * colors[0][k] + colors[1][k]) // 3 for k in range(3)) for _ in [0])
            colors.extend(tuple((colors[0][k] + 2 * colors[1][k]) // 3 for k in range(3)) for _ in [0])
            for py in range(4):
                for px in range(4):
                    x, y = bx * 4 + px, by * 4 + py
                    if x >= width or y >= height:
                        continue
                    n = py * 4 + px
                    ci = (color_bits >> (2 * n)) & 3
                    ai = (alpha_bits >> (3 * n)) & 7
                    dst = (y * width + x) * 4
                    rgba[dst:dst + 4] = bytes((*colors[ci], alphas[ai]))
    return bytes(rgba)


def parse_num(text: str) -> int:
    return int(text, 0)


def auto_layout(data: bytes) -> tuple[int, int, list[tuple[int, int]], int, int, int]:
    """Return the primary weapon mesh stream and texture layout.

    Some weapon DBs also contain a separate first-person arms/hands stream
    that shares the vertex buffer. It is not part of the weapon preview.
    """
    if len(data) < 0x200:
        raise ValueError("input is too short for a BLACK texture/database header")
    fmt = struct.unpack_from("<I", data, 0x18c)[0]
    if (fmt & 0xff) != 0x29:
        raise ValueError(f"expected Xbox DXT5 format 0x29 at 0x18c, got {fmt:#010x}")
    log_w, log_h = (fmt >> 20) & 0xf, (fmt >> 24) & 0xf
    if not (1 <= log_w <= 10 and 1 <= log_h <= 10):
        raise ValueError(f"unsupported dimensions encoded in texture format {fmt:#010x}")
    width, height = 1 << log_w, 1 << log_h
    texture_size = ((width + 3) // 4) * ((height + 3) // 4) * 16
    if 0x200 + texture_size > len(data):
        raise ValueError("DXT5 payload runs past the end of the DB")

    model_start = struct.unpack_from("<I", data, 0x20)[0]
    model_end = struct.unpack_from("<I", data, 0x2c)[0]
    if not (0x200 <= model_start < model_end <= len(data)):
        raise ValueError(f"invalid MODELS range {model_start:#x}..{model_end:#x}")

    # The FPS extraction scripts reveal the actual per-mesh group table in
    # chunk 1. Some weapons (notably the RPG) have a separate arms group after
    # the weapon group. Descriptor scanning below can mistake that later arms
    # stream for the preview model, yielding hands only. Prefer the group table
    # whenever it is present and choose the first group not named *_arm.
    chunk_count = struct.unpack_from("<I", data, 0x18)[0]
    if 0 < chunk_count < 256 and 0x20 + chunk_count * 4 <= len(data):
        offsets = [struct.unpack_from("<I", data, 0x1c + i * 4)[0]
                   for i in range(chunk_count + 1)]
        c1_start = offsets[1] if len(offsets) > 1 else 0
        later = sorted(set(o for o in offsets if o > c1_start) | {len(data)})
        c1_end = next((o for o in later if o > c1_start), len(data))
        if 0 <= c1_start < c1_end <= len(data):
            c1 = data[c1_start:c1_end]
            base = struct.unpack_from("<I", c1, 0x98)[0] if len(c1) > 0xa0 else 0x2480
            if base == 0 or base >= len(c1):
                base = 0x2480

            def scan_groups(lo: int, hi: int) -> list[tuple[int, int, int]]:
                found = []
                for rec in range(lo, min(hi, len(c1) - 0x2c), 16):
                    marker, count, rel = struct.unpack_from("<3I", c1, rec + 0x20)
                    abs_idx = rec + rel
                    if (marker == 6 and 100 < count < MAX_INDICES and 0x1000 < rel < len(c1)
                            and abs_idx + count * 2 <= len(c1)):
                        found.append((rec, count, abs_idx))
                return found

            groups = scan_groups(base + 0x1000, base + 0x2000)
            if not groups:
                candidates = scan_groups(0, len(c1))
                runs: list[list[tuple[int, int, int]]] = []
                run: list[tuple[int, int, int]] = []
                for group in candidates:
                    if run and group[0] - run[-1][0] == 0x30:
                        run.append(group)
                    else:
                        if run:
                            runs.append(run)
                        run = [group]
                if run:
                    runs.append(run)
                if runs:
                    groups = max(runs, key=lambda r: (len(r), r[0][1]))

            if groups:
                vstart = (groups[-1][0] + 0x30 + 15) & ~15
                total_verts = (groups[0][2] - vstart) // 36
                names = [s.decode("ascii", errors="ignore")
                         for s in __import__("re").findall(rb"bg1_\w+", c1[-0x1000:])]
                selected = next((i for i, _g in enumerate(groups)
                                 if i >= len(names) or "arm" not in names[i].lower()), 0)
                rec, index_count, index_rel = groups[selected]
                source = u16s(c1, index_rel, index_count)
                used = [idx for idx in source if idx != 0xffff]
                if used and total_verts > 0 and max(used) < total_verts:
                    vertex_count = max(used) + 1
                    # Confirm the exact stream begins at the expected 36-byte
                    # vertex buffer and contains plausible positions and UVs.
                    try:
                        read_vertices(c1, vstart, vertex_count)
                        strip_to_triangles(source, vertex_count)
                    except (ValueError, struct.error):
                        pass
                    else:
                        print(f"auto: group={names[selected] if selected < len(names) else selected}, "
                              f"vertices={vertex_count}, strip indices={index_count}, "
                              f"vertex offset={c1_start + vstart:#x}")
                        print(f"auto: texture={width}x{height}, DXT5 payload={texture_size:#x} bytes")
                        return (c1_start + vstart, vertex_count,
                                [(c1_start + index_rel, index_count)], 0x200, width, height)

    candidates = []
    descriptors = []
    # Mesh descriptor: [6, strip-index-count, relative-span, 0]. The primary
    # vertex buffer begins at descriptor+0x40; each descriptor's indices begin
    # at descriptor+span-0x20.
    for desc in range(model_start, model_end - 16, 4):
        marker, index_count, span, reserved = struct.unpack_from("<4I", data, desc)
        if marker != 6 or reserved != 0 or not (3 <= index_count <= MAX_INDICES) or span < 0x60:
            continue
        index_offset = desc + span - 0x20
        vertex_offset = desc + 0x40
        if index_offset <= vertex_offset or index_offset + index_count * 2 > model_end:
            continue
        try:
            source = u16s(data, index_offset, index_count)
            used = [idx for idx in source if idx != 0xffff]
            if not used or max(used) >= MAX_VERTICES:
                continue
            strip_to_triangles(source, MAX_VERTICES)
        except (ValueError, struct.error):
            continue
        vertex_count = max(idx for idx in source if idx != 0xffff) + 1
        if vertex_count > MAX_VERTICES:
            continue
        descriptors.append((desc, index_offset, index_count, vertex_count))
        try:
            read_vertices(data, vertex_offset, vertex_count)
        except (ValueError, struct.error):
            pass
        else:
            gap = index_offset - vertex_offset
            if 0 <= gap - vertex_count * 36 < 36:
                candidates.append((desc, vertex_offset, vertex_count, index_offset, index_count))
        # AK1's main weapon descriptor is followed immediately by its vertex
        # buffer (+0x10), and that buffer ends just before the separate hands
        # index stream. Don't mistake the more common descriptor+0x40 pattern
        # for this layout; validate the complete vertex range against that
        # neighbouring stream boundary.
        alt_vertex_offset = desc + 0x10
        alt_end = alt_vertex_offset + vertex_count * 36
        if any(0 <= other_index_offset - alt_end < 36
               for other_desc, other_index_offset, _, _ in descriptors
               if other_desc != desc and other_index_offset > alt_vertex_offset):
            try:
                read_vertices(data, alt_vertex_offset, vertex_count)
            except (ValueError, struct.error):
                continue
            candidates.append((desc, alt_vertex_offset, vertex_count, index_offset, index_count))
    if not candidates:
        raise ValueError("no valid model descriptor found; use explicit offset/count options")
    desc, vertex_offset, vertex_count, index_offset, index_count = min(candidates)
    streams = [(index_offset, index_count)]
    print(f"auto: descriptor={desc:#x}, vertices={vertex_count}, strip indices={index_count}, vertex offset={vertex_offset:#x}")
    print(f"auto: texture={width}x{height}, DXT5 payload={texture_size:#x} bytes")
    if len(candidates) > 1:
        print(f"auto: found {len(candidates)} valid descriptors; selected the first entry")
    return vertex_offset, vertex_count, streams, 0x200, width, height


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="source weapon .db")
    ap.add_argument("output", type=Path, help="destination .gmesh")
    ap.add_argument("--auto", action="store_true", help="detect the primary model and texture layout")
    ap.add_argument("--known-pistol", action="store_true", help="use the verified bg1_pst layout")
    ap.add_argument("--vertex-offset", type=parse_num)
    ap.add_argument("--vertex-count", type=parse_num)
    ap.add_argument("--index-offset", type=parse_num)
    ap.add_argument("--index-count", type=parse_num, help="number of 16-bit source strip indices")
    ap.add_argument("--texture-offset", type=parse_num, default=0x200)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--position-field", type=parse_num, default=0,
                    help="byte offset of float3 position within each 36-byte vertex (default 0)")
    ap.add_argument("--normal-field", type=parse_num, default=24,
                    help="byte offset of packed Xbox normal within each vertex (default 24)")
    ap.add_argument("--force", action="store_true", help="overwrite output if it already exists")
    args = ap.parse_args()

    if args.known_pistol:
        args.vertex_offset, args.vertex_count = 0x13c30, 5152
        args.index_offset, args.index_count = 0x410b0, 5905
        args.texture_offset, args.width, args.height = 0x200, 256, 256
    auto_streams = None
    if args.auto and not args.known_pistol:
        (args.vertex_offset, args.vertex_count, auto_streams,
         args.texture_offset, args.width, args.height) = auto_layout(args.input.read_bytes())
    required = (args.vertex_offset, args.vertex_count, args.width, args.height)
    if auto_streams is None:
        required += (args.index_offset, args.index_count)
    if any(v is None for v in required):
        ap.error("provide vertex/index offsets and counts plus --width/--height (or use --known-pistol)")
    if args.output.exists() and not args.force:
        ap.error(f"output already exists: {args.output} (pass --force to replace it)")

    data = args.input.read_bytes()
    if len(data) < 0x20:
        raise ValueError("input is too short to be a weapon database")
    if args.vertex_count <= 0 or args.vertex_count > MAX_VERTICES:
        raise ValueError(f"vertex count must be 1..{MAX_VERTICES}")
    if auto_streams is None and (args.index_count < 3 or args.index_count > MAX_INDICES):
        raise ValueError(f"source index count must be 3..{MAX_INDICES}")
    vertices, bounds = read_vertices(data, args.vertex_offset, args.vertex_count, args.position_field)
    streams = auto_streams or [(args.index_offset, args.index_count)]
    triangles = []
    source_index_total = 0
    for index_offset, index_count in streams:
        strip = u16s(data, index_offset, index_count)
        triangles.extend(strip_to_triangles(strip, args.vertex_count))
        source_index_total += len(strip)
    texture = decode_dxt5(data, args.texture_offset, args.width, args.height)

    out = bytearray(b"GMSH")
    # Version 2 adds the game's packed NORMPACKED3 vertex normal as float3.
    # Layout per vertex: position3, normal3, UV2 (32 bytes total).
    out += struct.pack("<5I", 2, len(vertices), len(triangles), args.width, args.height)
    def sign_extend(value: int, bits: int) -> int:
        return (value ^ (1 << (bits - 1))) - (1 << (bits - 1))

    for i, v in enumerate(vertices):
        # Match the axis/origin/scale conversion in the existing armory pistol
        # asset: source X becomes viewer Z, source Z becomes viewer X.
        x, y, z, u, vv = v
        out += struct.pack("<3f", z * 4.166054246770952 - 0.074782129260196006,
                           y * 4.1660542385894015 + 0.0046534836895695264,
                           x * 4.166054194683941)
        packed = struct.unpack_from("<I", data, args.vertex_offset + i * 36 + args.normal_field)[0]
        # Match loader/src/d3d/draw.cpp's Xbox NORMPACKED3 (11:11:10 SNORM).
        nx = sign_extend(packed & 0x7ff, 11) / 1023.0
        ny = sign_extend((packed >> 11) & 0x7ff, 11) / 1023.0
        nz = sign_extend(packed >> 22, 10) / 511.0
        # Positions map source X->viewer Z and source Z->viewer X.
        out += struct.pack("<3f", nz, ny, nx)
        out += struct.pack("<2f", u, vv)
    for idx in triangles:
        out += struct.pack("<I", idx)
    out += texture

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(out)
    print(f"Wrote {args.output} ({len(out):,} bytes)")
    print(f"vertices={len(vertices):,}, source strip indices={source_index_total:,}, triangle-list indices={len(triangles):,}")
    print(f"texture={args.width}x{args.height}, position bounds={bounds[:3]} .. {bounds[3:]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
