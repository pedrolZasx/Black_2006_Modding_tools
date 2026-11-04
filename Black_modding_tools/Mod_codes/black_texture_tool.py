# -*- coding: utf-8 -*-
"""
Black (Xbox) .db texture replacer - standalone app.

    python black_texture_tool.py                 -> opens the UI
    python black_texture_tool.py list  X.db
    python black_texture_tool.py export X.db 0 out_prefix        (writes out_prefix_rgb.png / _alpha.png)
    python black_texture_tool.py replace X.db 0 rgb.png alpha.png OUT.db [--new-size] [--invert-alpha]

Needs: Python 3.8+, Pillow, numpy (tkinter for the UI).   pip install pillow numpy

What it knows about the .db format (found by inspecting bg1_ak1.db / bg1_gk1.db):
  * header word 0x18 = chunk count, table of chunk start offsets at 0x1C (count+1 entries, not sorted).
  * a texture chunk = 0x80 byte GUID header, a descriptor, and the pixel data at chunk+0x180:
        chunk+0x10C  Xbox format dword:  bits 8-15 format (0x0C DXT1, 0x0E DXT3, 0x0F DXT5),
                     bits 16-19 mip levels, bits 20-23 log2(width), bits 24-27 log2(height)
        chunk+0x134  format, +0x138 width, +0x13C height   (mirror of the above)
        chunk+0x144  texture name (zero terminated)
  * same size + same format  -> the new pixels are written in place, nothing else moves.
  * different size           -> the chunk grows/shrinks (kept 0x80 aligned), every chunk-table entry
                                behind it is shifted by the difference, exactly like the mesh builder does.
Every build is re-read and checked (see verify_build).
"""
import os
import struct
import sys

import numpy as np
from PIL import Image

FMT_NAMES = {0x0C: "DXT1", 0x0E: "DXT3", 0x0F: "DXT5"}
BLOCK_BYTES = {0x0C: 8, 0x0E: 16, 0x0F: 16}
DESC_OFF = 0x100          # descriptor start inside the chunk
FMT_DWORD = 0x10C
MIRROR = 0x134
NAME_OFF = 0x144
DATA_OFF = 0x180
ALIGN = 0x80


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


# --------------------------------------------------------------------------------------------- db layout
def chunk_table(d):
    n = u32(d, 0x18)
    if not (0 < n < 4096) or 0x1C + 4 * (n + 1) > len(d):
        raise ValueError("not a Black .db (bad chunk count)")
    return [u32(d, 0x1C + 4 * i) for i in range(n + 1)]


def chunk_ranges(d):
    offs = sorted(set(chunk_table(d) + [len(d)]))
    return [(offs[i], offs[i + 1]) for i in range(len(offs) - 1)]


class Texture:
    def __init__(self, index, start, end, fmt, w, h, mips, name, size):
        self.index, self.start, self.end = index, start, end
        self.fmt, self.w, self.h, self.mips, self.name = fmt, w, h, mips, name
        self.size = size                       # pixel data bytes (all mip levels)

    @property
    def fmt_name(self):
        return FMT_NAMES.get(self.fmt, "0x%02X" % self.fmt)

    def label(self):
        return "%s   %dx%d  %s  mips %d   (chunk 0x%X)" % (self.name or "?", self.w, self.h, self.fmt_name, self.mips, self.start)


def level_dims(w, h, mips):
    return [(max(1, w >> l), max(1, h >> l)) for l in range(mips)]


def level_bytes(w, h, fmt):
    return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * BLOCK_BYTES[fmt]


def total_bytes(w, h, mips, fmt):
    return sum(level_bytes(a, b, fmt) for a, b in level_dims(w, h, mips))


def find_textures(d):
    out = []
    for i, (s, e) in enumerate(chunk_ranges(d)):
        if e - s < DATA_OFF + 8:
            continue
        fd = u32(d, s + FMT_DWORD)
        fmt = (fd >> 8) & 0xFF
        if fmt not in BLOCK_BYTES:
            continue
        mips = (fd >> 16) & 0xF
        w, h = 1 << ((fd >> 20) & 0xF), 1 << ((fd >> 24) & 0xF)
        if mips < 1 or u32(d, s + MIRROR) != fmt or u32(d, s + MIRROR + 4) != w or u32(d, s + MIRROR + 8) != h:
            continue
        size = total_bytes(w, h, mips, fmt)
        spare = e - s - DATA_OFF - size
        if spare < 0 or spare >= 2 * ALIGN:
            continue
        name = bytes(d[s + NAME_OFF:s + NAME_OFF + 32]).split(b"\0")[0].decode("ascii", "replace")
        out.append(Texture(len(out), s, e, fmt, w, h, mips, name, size))
    return out


# --------------------------------------------------------------------------------------------- decode
def _c565(v):
    r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.int32)


def decode_dxt(data, w, h, fmt):
    """-> RGBA uint8 array (h, w, 4)"""
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    n = bw * bh
    bs = BLOCK_BYTES[fmt]
    blk = np.frombuffer(bytes(data[:n * bs]), np.uint8).reshape(n, bs).astype(np.uint32)
    col = blk[:, bs - 8:]
    c0 = col[:, 0] | (col[:, 1] << 8)
    c1 = col[:, 2] | (col[:, 3] << 8)
    p0, p1 = _c565(c0), _c565(c1)
    pal = np.zeros((n, 4, 4), np.int32)
    pal[:, :, 3] = 255
    four = (c0 > c1) | (fmt != 0x0C)
    pal[:, 0, :3], pal[:, 1, :3] = p0, p1
    pal[:, 2, :3] = np.where(four[:, None], (2 * p0 + p1) // 3, (p0 + p1) // 2)
    pal[:, 3, :3] = np.where(four[:, None], (p0 + 2 * p1) // 3, 0)
    pal[:, 3, 3] = np.where(four, 255, 0)
    bits = col[:, 4] | (col[:, 5] << 8) | (col[:, 6] << 16) | (col[:, 7] << 24)
    idx = ((bits[:, None] >> (2 * np.arange(16, dtype=np.uint32))) & 3).astype(np.int64)
    pix = pal[np.arange(n)[:, None], idx]                       # (n,16,4)
    if fmt == 0x0E:
        ab = blk[:, 0:8]
        a64 = np.zeros(n, np.uint64)
        for k in range(8):
            a64 |= ab[:, k].astype(np.uint64) << np.uint64(8 * k)
        nib = ((a64[:, None] >> (np.uint64(4) * np.arange(16, dtype=np.uint64))) & np.uint64(15)).astype(np.int32)
        pix[:, :, 3] = nib * 17
    elif fmt == 0x0F:
        a0, a1 = blk[:, 0].astype(np.int32), blk[:, 1].astype(np.int32)
        ap = np.zeros((n, 8), np.int32)
        ap[:, 0], ap[:, 1] = a0, a1
        eight = a0 > a1
        for k in range(1, 7):
            ap[:, 1 + k] = np.where(eight, ((7 - k) * a0 + k * a1) // 7, 0)
        for k in range(1, 5):
            ap[:, 1 + k] = np.where(eight, ap[:, 1 + k], ((5 - k) * a0 + k * a1) // 5)
        ap[:, 6] = np.where(eight, ap[:, 6], 0)
        ap[:, 7] = np.where(eight, ap[:, 7], 255)
        a48 = np.zeros(n, np.uint64)
        for k in range(6):
            a48 |= blk[:, 2 + k].astype(np.uint64) << np.uint64(8 * k)
        ai = ((a48[:, None] >> (np.uint64(3) * np.arange(16, dtype=np.uint64))) & np.uint64(7)).astype(np.int64)
        pix[:, :, 3] = ap[np.arange(n)[:, None], ai]
    img = pix.astype(np.uint8).reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return img[:h, :w]


# --------------------------------------------------------------------------------------------- encode
def _blocks(img):
    h, w = img.shape[:2]
    bh, bw = (h + 3) // 4, (w + 3) // 4
    if (bh * 4, bw * 4) != (h, w):
        img = np.pad(img, ((0, bh * 4 - h), (0, bw * 4 - w), (0, 0)), mode="edge")
    return img.reshape(bh, 4, bw, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * bw, 16, 4)


def _q565(c):
    c = np.clip(np.rint(c), 0, 255).astype(np.int32)
    r = np.clip(np.rint(c[..., 0] * 31 / 255.0), 0, 31).astype(np.int32)
    g = np.clip(np.rint(c[..., 1] * 63 / 255.0), 0, 63).astype(np.int32)
    b = np.clip(np.rint(c[..., 2] * 31 / 255.0), 0, 31).astype(np.int32)
    return (r << 11) | (g << 5) | b


def _encode_color(px, rgba_alpha, fmt):
    """px (n,16,4) -> (n,8) uint8 colour blocks. DXT1 uses 3-colour+transparent mode where alpha<128."""
    n = px.shape[0]
    rgb = px[:, :, :3].astype(np.float64)
    trans = (px[:, :, 3] < 128) if (fmt == 0x0C) else np.zeros((n, 16), bool)
    wgt = (~trans).astype(np.float64)[:, :, None]
    cnt = np.maximum(wgt.sum(1), 1)
    mean = (rgb * wgt).sum(1) / cnt
    cen = (rgb - mean[:, None, :]) * wgt
    cov = np.einsum("nki,nkj->nij", cen, cen)
    ax = np.ones((n, 3)) / np.sqrt(3.0)
    for _ in range(8):
        ax = np.einsum("nij,nj->ni", cov, ax)
        nrm = np.linalg.norm(ax, axis=1, keepdims=True)
        ax = np.where(nrm > 1e-9, ax / np.maximum(nrm, 1e-9), np.ones((n, 3)) / np.sqrt(3.0))
    proj = np.einsum("nki,ni->nk", rgb - mean[:, None, :], ax)
    big = 1e9
    tmax = np.where(trans, -big, proj).max(1)
    tmin = np.where(trans, big, proj).min(1)
    tmax = np.where(tmax < -big / 2, 0, tmax)
    tmin = np.where(tmin > big / 2, 0, tmin)
    e0 = mean + ax * tmax[:, None]
    e1 = mean + ax * tmin[:, None]
    q0, q1 = _q565(e0), _q565(e1)
    has_t = trans.any(1)
    # opaque blocks need c0 > c1 (4-colour mode); blocks with transparent texels need c0 <= c1
    swap = np.where(has_t, q0 > q1, q0 < q1)
    q0, q1 = np.where(swap, q1, q0), np.where(swap, q0, q1)
    same = q0 == q1
    p0, p1 = _c565(q0).astype(np.float64), _c565(q1).astype(np.float64)
    pal = np.stack([p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3], 1)           # 4-colour
    pal3 = np.stack([p0, p1, (p0 + p1) / 2, p1], 1)                              # 3-colour (+transparent)
    use3 = (has_t | (same & (fmt == 0x0C)))[:, None, None]
    pal = np.where(use3, pal3, pal)
    dist = ((rgb[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)              # (n,16,4)
    dist[:, :, 3] = np.where(has_t[:, None], 1e12, dist[:, :, 3])                # index 3 only for transparent texels
    idx = dist.argmin(2)
    idx = np.where(trans, 3, idx)
    idx = np.where(same[:, None] & ~has_t[:, None], 0, idx)
    bits = np.zeros(n, np.uint64)
    for k in range(16):
        bits |= idx[:, k].astype(np.uint64) << np.uint64(2 * k)
    out = np.zeros((n, 8), np.uint8)
    out[:, 0], out[:, 1] = q0 & 255, q0 >> 8
    out[:, 2], out[:, 3] = q1 & 255, q1 >> 8
    for k in range(4):
        out[:, 4 + k] = ((bits >> np.uint64(8 * k)) & np.uint64(255)).astype(np.uint8)
    return out


def _alpha_pal(a0, a1, eight):
    n = len(a0)
    pal = np.zeros((n, 8), np.float64)
    pal[:, 0], pal[:, 1] = a0, a1
    for k in range(1, 7):
        pal[:, 1 + k] = ((7 - k) * a0 + k * a1) / 7.0
    p6 = pal.copy()
    for k in range(1, 5):
        p6[:, 1 + k] = ((5 - k) * a0 + k * a1) / 5.0
    p6[:, 6], p6[:, 7] = 0, 255
    return np.where(eight[:, None], pal, p6)


def _encode_alpha(a):
    """a (n,16) int -> (n,8) uint8 DXT5 alpha blocks"""
    n = a.shape[0]
    af = a.astype(np.float64)
    hi, lo = af.max(1), af.min(1)
    palA = _alpha_pal(hi, lo, np.ones(n, bool))                                  # 8-level, a0>=a1
    # 6-level mode with exact 0 / 255
    mid = (a > 0) & (a < 255)
    mlo = np.where(mid, af, 1e9).min(1)
    mhi = np.where(mid, af, -1e9).max(1)
    none = ~mid.any(1)
    mlo, mhi = np.where(none, 0, mlo), np.where(none, 0, mhi)
    palB = _alpha_pal(mlo, mhi, np.zeros(n, bool))                               # a0<=a1
    dA = np.abs(af[:, :, None] - palA[:, None, :])
    dB = np.abs(af[:, :, None] - palB[:, None, :])
    iA, iB = dA.argmin(2), dB.argmin(2)
    eA = np.take_along_axis(dA, iA[:, :, None], 2).squeeze(2).__pow__(2).sum(1)
    eB = np.take_along_axis(dB, iB[:, :, None], 2).squeeze(2).__pow__(2).sum(1)
    useB = (eB < eA) & (mlo <= mhi)
    idx = np.where(useB[:, None], iB, iA)
    a0 = np.where(useB, mlo, hi)
    a1 = np.where(useB, mhi, lo)
    bits = np.zeros(n, np.uint64)
    for k in range(16):
        bits |= idx[:, k].astype(np.uint64) << np.uint64(3 * k)
    out = np.zeros((n, 8), np.uint8)
    out[:, 0], out[:, 1] = np.rint(a0).astype(np.uint8), np.rint(a1).astype(np.uint8)
    for k in range(6):
        out[:, 2 + k] = ((bits >> np.uint64(8 * k)) & np.uint64(255)).astype(np.uint8)
    return out


def encode_dxt(img, fmt):
    """RGBA uint8 (h,w,4) -> DXT bytes"""
    px = _blocks(img.astype(np.uint8))
    col = _encode_color(px, None, fmt)
    if fmt == 0x0C:
        return col.tobytes()
    if fmt == 0x0F:
        return np.concatenate([_encode_alpha(px[:, :, 3].astype(np.int32)), col], 1).tobytes()
    a4 = np.rint(px[:, :, 3] / 17.0).astype(np.uint64)                           # DXT3: 4-bit alpha
    bits = np.zeros(len(px), np.uint64)
    for k in range(16):
        bits |= a4[:, k] << np.uint64(4 * k)
    ab = np.zeros((len(px), 8), np.uint8)
    for k in range(8):
        ab[:, k] = ((bits >> np.uint64(8 * k)) & np.uint64(255)).astype(np.uint8)
    return np.concatenate([ab, col], 1).tobytes()


def encode_levels(img, w, h, mips, fmt):
    """Encode the base image plus box-filtered mip levels."""
    pil = Image.fromarray(img, "RGBA")
    out = []
    for lw, lh in level_dims(w, h, mips):
        lvl = np.asarray(pil if (lw, lh) == (w, h) else pil.resize((lw, lh), Image.BOX))
        out.append(encode_dxt(lvl, fmt))
    return b"".join(out)


# --------------------------------------------------------------------------------------------- texture io
def read_texture(d, t, level=0):
    off = t.start + DATA_OFF
    for lw, lh in level_dims(t.w, t.h, level):
        off += level_bytes(lw, lh, t.fmt)
    lw, lh = level_dims(t.w, t.h, level + 1)[-1]
    return decode_dxt(d[off:off + level_bytes(lw, lh, t.fmt)], lw, lh, t.fmt)


def split_rgba(rgba):
    rgb = Image.fromarray(np.ascontiguousarray(rgba[:, :, :3]), "RGB")
    alpha = Image.fromarray(np.ascontiguousarray(rgba[:, :, 3]), "L")
    return rgb, alpha


def combine(rgb_path, alpha_path, size, invert_alpha=False, alpha_from_rgb=False):
    """Load the user's images, resize to 'size' = (w,h) and return RGBA uint8 + notes."""
    notes = []
    src = Image.open(rgb_path)
    if src.size != size:
        notes.append("RGB image %dx%d resized to %dx%d" % (src.size[0], src.size[1], size[0], size[1]))
    rgb = src.convert("RGB").resize(size, Image.LANCZOS)
    if alpha_path:
        a_img = Image.open(alpha_path)
        if a_img.size != size:
            notes.append("alpha image %dx%d resized to %dx%d" % (a_img.size[0], a_img.size[1], size[0], size[1]))
        alpha = a_img.convert("L").resize(size, Image.LANCZOS)
    elif alpha_from_rgb and "A" in src.getbands():
        alpha = src.getchannel("A").resize(size, Image.LANCZOS)
    else:
        alpha = Image.new("L", size, 255)
        notes.append("no alpha image given: texture is fully opaque")
    a = np.asarray(alpha)
    if invert_alpha:
        a = 255 - a
    rgba = np.dstack([np.asarray(rgb), a]).astype(np.uint8)
    return rgba, notes


# --------------------------------------------------------------------------------------------- build
def pad_to(b, align=ALIGN):
    return b + bytes((-len(b)) % align)


def build_replacement(d, t, rgba, new_size=None, keep_mips=True):
    """Return (new_file_bytes, info dict). 'rgba' must already be (h,w,4) of the target size."""
    h, w = rgba.shape[:2]
    if (w, h) != (t.w, t.h):
        if not new_size:
            raise ValueError("image is %dx%d but the texture is %dx%d (enable 'allow new size')" % (w, h, t.w, t.h))
        for v in (w, h):
            if v < 4 or v & (v - 1) or v > 4096:
                raise ValueError("Xbox textures need power-of-two sizes between 4 and 4096 (got %dx%d)" % (w, h))
    mips = t.mips if keep_mips else 1
    mips = max(1, min(mips, min(w, h).bit_length()))
    data = encode_levels(rgba, w, h, mips, t.fmt)
    old = bytearray(d[t.start:t.end])
    head = bytearray(old[:DATA_OFF])
    fd = u32(head, FMT_DWORD)
    fd = (fd & ~(0xF << 16) & ~(0xF << 20) & ~(0xF << 24)) | (mips << 16) | ((w.bit_length() - 1) << 20) | ((h.bit_length() - 1) << 24)
    struct.pack_into("<I", head, FMT_DWORD, fd)
    struct.pack_into("<3I", head, MIRROR, t.fmt, w, h)
    new_chunk = pad_to(bytes(head) + data)
    if len(new_chunk) == len(old):
        out = bytearray(d)
        out[t.start:t.end] = new_chunk
        shift = 0
    else:
        shift = len(new_chunk) - len(old)
        out = bytearray(d[:t.start]) + new_chunk + bytearray(d[t.end:])
        n = u32(out, 0x18)
        for i in range(n + 1):                         # same rule as the mesh builder's patch_db_table
            o = 0x1C + 4 * i
            v = u32(out, o)
            if v >= t.end:
                struct.pack_into("<I", out, o, v + shift)
    return bytes(out), {"shift": shift, "w": w, "h": h, "mips": mips, "data": data}


def verify_build(orig, new, t, info, rgba):
    """Re-read the built file. Returns (ok, [lines])."""
    lines, ok = [], True

    def chk(cond, text):
        nonlocal ok
        ok &= bool(cond)
        lines.append(("[OK]   " if cond else "[FAIL] ") + text)

    shift = info["shift"]
    chk(len(new) == len(orig) + shift, "file size changed by %+d bytes" % shift)
    chk(new[:0x18] == orig[:0x18] and u32(new, 0x18) == u32(orig, 0x18), "db header / chunk count unchanged")
    to, tn = chunk_table(orig), chunk_table(new)
    exp = [v + shift if v >= t.end else v for v in to]
    chk(tn == exp, "chunk table: entries behind the texture shifted by %+d, the rest untouched" % shift)
    chk(all(v % ALIGN == 0 for v in tn), "all chunk offsets still 0x%X-aligned" % ALIGN)
    tab_a, tab_b = 0x1C, 0x1C + 4 * len(to)                   # the chunk table itself is the only thing allowed to differ
    chk(new[:tab_a] == orig[:tab_a] and new[tab_b:t.start] == orig[tab_b:t.start],
        "everything before the texture chunk is byte-identical (except the chunk table)")
    chk(new[t.end + shift:] == orig[t.end:], "everything after the texture chunk is byte-identical (only moved)" if shift else
        "everything after the texture chunk is byte-identical")
    nt = [x for x in find_textures(new) if x.start == t.start]
    chk(len(nt) == 1, "texture chunk still parses as a texture")
    if nt:
        x = nt[0]
        chk((x.w, x.h, x.fmt) == (info["w"], info["h"], t.fmt), "descriptor: %dx%d %s, mips %d" % (x.w, x.h, x.fmt_name, x.mips))
        chk(x.end - x.start == len(pad_to(new[x.start:x.start + DATA_OFF] + info["data"])), "chunk length = header + pixel data (aligned)")
        got = new[x.start + DATA_OFF:x.start + DATA_OFF + len(info["data"])]
        chk(got == info["data"], "pixel data in the file equals what was encoded")
        dec = read_texture(new, x).astype(np.int32)
        diff = ((dec - rgba.astype(np.int32)) ** 2).mean(axis=(0, 1))
        psnr = [10 * np.log10(255.0 ** 2 / max(m, 1e-9)) for m in diff]
        lines.append("       quality (PSNR dB): R %.1f  G %.1f  B %.1f  A %.1f" % tuple(psnr))
    return ok, lines


# --------------------------------------------------------------------------------------------- CLI
def _cli(argv):
    cmd = argv[0]
    d = open(argv[1], "rb").read()
    tex = find_textures(d)
    if cmd == "list":
        for t in tex:
            print(t.index, t.label())
        return 0
    t = tex[int(argv[2])]
    if cmd == "export":
        rgb, alpha = split_rgba(read_texture(d, t))
        rgb.save(argv[3] + "_rgb.png")
        alpha.save(argv[3] + "_alpha.png")
        print("wrote", argv[3] + "_rgb.png,", argv[3] + "_alpha.png")
        return 0
    if cmd == "replace":
        flags = argv[6:]
        rgb_p, al_p, out_p = argv[3], argv[4], argv[5]
        al_p = None if al_p.lower() in ("none", "-") else al_p
        src = Image.open(rgb_p)
        size = src.size if "--new-size" in flags else (t.w, t.h)
        rgba, notes = combine(rgb_p, al_p, size, "--invert-alpha" in flags, True)
        new, info = build_replacement(d, t, rgba, "--new-size" in flags)
        ok, lines = verify_build(d, new, t, info, rgba)
        for n in notes:
            print("note:", n)
        print("\n".join(lines))
        if ok:
            open(out_p, "wb").write(new)
            print("saved", out_p)
        else:
            print("verification failed - nothing written")
        return 0 if ok else 1
    return 2


# --------------------------------------------------------------------------------------------- UI
def run_ui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("Black texture replacer")
    root.geometry("1040x640")
    S = {"path": None, "data": None, "tex": [], "img": None, "photo": None}
    rgb_var, alpha_var = tk.StringVar(), tk.StringVar()
    view_var, inv_var, newsize_var = tk.StringVar(value="combined"), tk.BooleanVar(), tk.BooleanVar()
    status = tk.StringVar(value="Open a .db to begin.")

    top = ttk.Frame(root, padding=6)
    top.pack(fill="x")
    left = ttk.Frame(root, padding=6)
    left.pack(side="left", fill="y")
    right = ttk.Frame(root, padding=6)
    right.pack(side="left", fill="both", expand=True)

    lb = tk.Listbox(left, width=44, height=10, exportselection=False)
    lb.pack(fill="x")

    canvas = tk.Canvas(right, bg="#303030", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    def checker(w, h, n=8):
        yy, xx = np.mgrid[0:h, 0:w]
        c = np.where(((xx // n + yy // n) % 2)[:, :, None] == 0, 200, 140).astype(np.uint8)
        return np.repeat(c, 3, 2)

    def show(rgba):
        S["img"] = rgba
        draw()

    def draw(_e=None):
        rgba = S["img"]
        canvas.delete("all")
        if rgba is None:
            return
        mode = view_var.get()
        h, w = rgba.shape[:2]
        if mode == "rgb":
            im = Image.fromarray(rgba[:, :, :3], "RGB")
        elif mode == "alpha":
            im = Image.fromarray(rgba[:, :, 3], "L").convert("RGB")
        else:
            a = rgba[:, :, 3:4].astype(np.float32) / 255.0
            comp = (rgba[:, :, :3] * a + checker(w, h) * (1 - a)).astype(np.uint8)
            im = Image.fromarray(comp, "RGB")
        cw, ch = max(canvas.winfo_width(), 50), max(canvas.winfo_height(), 50)
        sc = min(cw / w, ch / h)
        im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.NEAREST if sc >= 2 else Image.BILINEAR)
        S["photo"] = ImageTk.PhotoImage(im)
        canvas.create_image(cw // 2, ch // 2, image=S["photo"])

    canvas.bind("<Configure>", draw)

    def cur():
        sel = lb.curselection()
        return S["tex"][sel[0]] if sel else None

    def on_select(_e=None):
        t = cur()
        if t is None:
            return
        try:
            show(read_texture(S["data"], t))
            status.set("%s   |   file %s" % (t.label(), os.path.basename(S["path"])))
        except Exception as exc:                                                    # pragma: no cover
            messagebox.showerror("Decode failed", str(exc))

    lb.bind("<<ListboxSelect>>", on_select)

    def load(path):
        d = open(path, "rb").read()
        tex = find_textures(d)
        S.update(path=path, data=d, tex=tex, img=None)
        lb.delete(0, "end")
        for t in tex:
            lb.insert("end", t.label())
        if tex:
            lb.selection_set(0)
            on_select()
        else:
            canvas.delete("all")
            status.set("No supported textures (DXT1/DXT3/DXT5) found in %s" % os.path.basename(path))

    def open_db():
        p = filedialog.askopenfilename(title="Open .db", filetypes=[("Black db", "*.db"), ("All files", "*.*")])
        if p:
            try:
                load(p)
            except Exception as exc:
                messagebox.showerror("Cannot open", str(exc))

    def pick(var, title):
        p = filedialog.askopenfilename(title=title, filetypes=[("Images", "*.png *.tga *.bmp *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")])
        if p:
            var.set(p)

    def export_maps():
        t = cur()
        if not t:
            return
        p = filedialog.asksaveasfilename(title="Save maps as (suffixes _rgb.png / _alpha.png are added)", initialfile=t.name or "texture")
        if p:
            rgb, alpha = split_rgba(read_texture(S["data"], t))
            rgb.save(p + "_rgb.png")
            alpha.save(p + "_alpha.png")
            status.set("Saved %s_rgb.png and %s_alpha.png" % (p, p))

    def replace():
        t = cur()
        if not t:
            messagebox.showinfo("Replace", "Open a db and select a texture first.")
            return
        if not rgb_var.get():
            messagebox.showinfo("Replace", "Choose the RGB image first.")
            return
        try:
            src_size = Image.open(rgb_var.get()).size
            size = src_size if newsize_var.get() else (t.w, t.h)
            rgba, notes = combine(rgb_var.get(), alpha_var.get() or None, size, inv_var.get(), True)
            new, info = build_replacement(S["data"], t, rgba, newsize_var.get())
            ok, lines = verify_build(S["data"], new, t, info, rgba)
        except Exception as exc:
            messagebox.showerror("Replace failed", str(exc))
            return
        report = "\n".join(["note: " + n for n in notes] + lines)
        if not ok:
            messagebox.showerror("Verification failed - nothing was written", report)
            return
        base, ext = os.path.splitext(os.path.basename(S["path"]))
        out = filedialog.asksaveasfilename(title="Save new db as", initialfile=base + "_newtex" + ext, defaultextension=".db")
        if not out:
            return
        if os.path.abspath(out) == os.path.abspath(S["path"]):
            if not messagebox.askyesno("Overwrite?", "This replaces the db you opened. Continue?"):
                return
        open(out, "wb").write(new)
        if messagebox.askyesno("Done", report + "\n\nSaved: " + out + "\n\nOpen the new db now?"):
            load(out)

    ttk.Button(top, text="Open .db…", command=open_db).pack(side="left")
    ttk.Button(top, text="Export maps…", command=export_maps).pack(side="left", padx=4)
    for txt, val in (("Combined", "combined"), ("RGB", "rgb"), ("Alpha", "alpha")):
        ttk.Radiobutton(top, text=txt, value=val, variable=view_var, command=draw).pack(side="left", padx=2)

    ttk.Label(left, text="RGB image").pack(anchor="w", pady=(10, 0))
    fr = ttk.Frame(left)
    fr.pack(fill="x")
    ttk.Entry(fr, textvariable=rgb_var, width=34).pack(side="left", fill="x", expand=True)
    ttk.Button(fr, text="…", width=3, command=lambda: pick(rgb_var, "RGB image")).pack(side="left")
    ttk.Label(left, text="Alpha image (grayscale, white = opaque; optional)").pack(anchor="w", pady=(8, 0))
    fa = ttk.Frame(left)
    fa.pack(fill="x")
    ttk.Entry(fa, textvariable=alpha_var, width=34).pack(side="left", fill="x", expand=True)
    ttk.Button(fa, text="…", width=3, command=lambda: pick(alpha_var, "Alpha image")).pack(side="left")
    ttk.Checkbutton(left, text="Invert alpha", variable=inv_var).pack(anchor="w", pady=(8, 0))
    ttk.Checkbutton(left, text="Allow a new size (rebuilds chunk, shifts the chunk table)", variable=newsize_var).pack(anchor="w")
    ttk.Button(left, text="Replace texture → build new .db", command=replace).pack(fill="x", pady=14)
    ttk.Label(left, textvariable=status, wraplength=300, foreground="#555").pack(anchor="w")

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        root.after(100, lambda: load(sys.argv[1]))
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] in ("list", "export", "replace"):
        sys.exit(_cli(sys.argv[1:]))
    run_ui()
