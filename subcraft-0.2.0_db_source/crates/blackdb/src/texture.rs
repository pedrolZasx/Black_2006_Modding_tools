//! DXT1/3/5 texture chunks of a Black `.db` (ported from `black_texture_tool.py`).
//!
//! A texture chunk = 0x80-byte GUID header, a descriptor and pixel data at `chunk + 0x180`.
//! `chunk + 0x10C` holds the Xbox format dword (bits 8-15 format, 16-19 mips, 20-23 log2 width,
//! 24-27 log2 height), `+0x134..` mirrors format/width/height and `+0x144` is the name.

use crate::{ALIGN, Error, chunk_ranges, chunk_table, put_u32, u32_at};

pub const DXT1: u32 = 0x0C;
pub const DXT3: u32 = 0x0E;
pub const DXT5: u32 = 0x0F;

const FMT_DWORD: usize = 0x10C;
const MIRROR: usize = 0x134;
const NAME_OFF: usize = 0x144;
const DATA_OFF: usize = 0x180;

pub fn fmt_name(fmt: u32) -> String {
    match fmt {
        DXT1 => "DXT1".into(),
        DXT3 => "DXT3".into(),
        DXT5 => "DXT5".into(),
        other => format!("0x{other:02X}"),
    }
}

fn block_bytes(fmt: u32) -> Option<usize> {
    match fmt {
        DXT1 => Some(8),
        DXT3 | DXT5 => Some(16),
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub struct Texture {
    pub index: usize,
    pub start: usize,
    pub end: usize,
    pub fmt: u32,
    pub w: usize,
    pub h: usize,
    pub mips: usize,
    pub name: String,
    /// Pixel bytes of all mip levels.
    pub size: usize,
}

impl Texture {
    pub fn label(&self) -> String {
        format!("{}  {}x{}  {}  mips {}  (chunk 0x{:X})", if self.name.is_empty() { "?" } else { &self.name }, self.w, self.h, fmt_name(self.fmt), self.mips, self.start)
    }
}

fn level_dims(w: usize, h: usize, mips: usize) -> Vec<(usize, usize)> {
    (0..mips).map(|l| ((w >> l).max(1), (h >> l).max(1))).collect()
}

fn level_bytes(w: usize, h: usize, fmt: u32) -> usize {
    w.div_ceil(4).max(1) * h.div_ceil(4).max(1) * block_bytes(fmt).unwrap_or(16)
}

fn total_bytes(w: usize, h: usize, mips: usize, fmt: u32) -> usize {
    level_dims(w, h, mips).iter().map(|&(a, b)| level_bytes(a, b, fmt)).sum()
}

pub fn find_textures(d: &[u8]) -> Result<Vec<Texture>, Error> {
    let mut out = Vec::new();
    for (s, e) in chunk_ranges(d)? {
        if e - s < DATA_OFF + 8 {
            continue;
        }
        let Some(fd) = u32_at(d, s + FMT_DWORD) else { continue };
        let fmt = (fd >> 8) & 0xFF;
        if block_bytes(fmt).is_none() {
            continue;
        }
        let mips = ((fd >> 16) & 0xF) as usize;
        let (lw, lh) = ((fd >> 20) & 0xF, (fd >> 24) & 0xF);
        let (w, h) = (1usize << lw, 1usize << lh);
        if mips < 1 || u32_at(d, s + MIRROR) != Some(fmt) || u32_at(d, s + MIRROR + 4) != Some(w as u32) || u32_at(d, s + MIRROR + 8) != Some(h as u32) {
            continue;
        }
        let size = total_bytes(w, h, mips, fmt);
        let Some(spare) = (e - s).checked_sub(DATA_OFF + size) else { continue };
        if spare >= 2 * ALIGN {
            continue;
        }
        let raw = &d[s + NAME_OFF..(s + NAME_OFF + 32).min(d.len())];
        let name = String::from_utf8_lossy(raw.split(|&b| b == 0).next().unwrap_or(&[])).into_owned();
        out.push(Texture { index: out.len(), start: s, end: e, fmt, w, h, mips, name, size });
    }
    Ok(out)
}

/// Pick the texture that belongs to a mesh group named `group` (e.g. `bg1_pst_gun`): exact name,
/// then one name containing the other, else the largest texture. Returns the index and a note.
pub fn pick_for_group(textures: &[Texture], group: &str) -> Option<(usize, String)> {
    let g = group.to_ascii_lowercase();
    let by = |f: &dyn Fn(&str) -> bool| textures.iter().position(|t| !t.name.is_empty() && f(&t.name.to_ascii_lowercase()));
    if let Some(i) = by(&|n| n == g) {
        return Some((i, format!("texture '{}' matches group name", textures[i].name)));
    }
    if let Some(i) = by(&|n| n.contains(&g) || g.contains(n)) {
        return Some((i, format!("texture '{}' partly matches group '{group}'", textures[i].name)));
    }
    let i = textures.iter().enumerate().max_by_key(|(_, t)| t.w * t.h).map(|(i, _)| i)?;
    Some((i, format!("no texture name matches '{group}'; using the largest ('{}')", textures[i].name)))
}

// ------------------------------------------------------------------------------------------ decode
fn c565(v: u16) -> [i32; 3] {
    let (r, g, b) = (((v >> 11) & 31) as i32, ((v >> 5) & 63) as i32, (v & 31) as i32);
    [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)]
}

/// Decode one mip level to RGBA8 (`w * h * 4`).
pub fn decode_dxt(data: &[u8], w: usize, h: usize, fmt: u32) -> Result<Vec<u8>, Error> {
    let bs = block_bytes(fmt).ok_or_else(|| Error::Texture(format!("unsupported format {}", fmt_name(fmt))))?;
    let (bw, bh) = (w.div_ceil(4).max(1), h.div_ceil(4).max(1));
    if data.len() < bw * bh * bs {
        return Err(Error::Texture("texture data is truncated".into()));
    }
    let mut out = vec![0u8; w * h * 4];
    for by in 0..bh {
        for bx in 0..bw {
            let blk = &data[(by * bw + bx) * bs..(by * bw + bx + 1) * bs];
            let col = &blk[bs - 8..];
            let c0 = u16::from_le_bytes([col[0], col[1]]);
            let c1 = u16::from_le_bytes([col[2], col[3]]);
            let (p0, p1) = (c565(c0), c565(c1));
            let four = c0 > c1 || fmt != DXT1;
            let mut pal = [[0i32; 4]; 4];
            for k in 0..3 {
                pal[0][k] = p0[k];
                pal[1][k] = p1[k];
                pal[2][k] = if four { (2 * p0[k] + p1[k]) / 3 } else { (p0[k] + p1[k]) / 2 };
                pal[3][k] = if four { (p0[k] + 2 * p1[k]) / 3 } else { 0 };
            }
            for p in &mut pal {
                p[3] = 255;
            }
            pal[3][3] = if four { 255 } else { 0 };
            let bits = u32::from_le_bytes([col[4], col[5], col[6], col[7]]);
            let mut alpha = [255i32; 16];
            match fmt {
                DXT3 => {
                    let a64 = u64::from_le_bytes([blk[0], blk[1], blk[2], blk[3], blk[4], blk[5], blk[6], blk[7]]);
                    for (k, a) in alpha.iter_mut().enumerate() {
                        *a = ((a64 >> (4 * k)) & 15) as i32 * 17;
                    }
                }
                DXT5 => {
                    let (a0, a1) = (blk[0] as i32, blk[1] as i32);
                    let mut ap = [0i32; 8];
                    ap[0] = a0;
                    ap[1] = a1;
                    if a0 > a1 {
                        for k in 1..7 {
                            ap[1 + k] = ((7 - k as i32) * a0 + k as i32 * a1) / 7;
                        }
                    } else {
                        for k in 1..5 {
                            ap[1 + k] = ((5 - k as i32) * a0 + k as i32 * a1) / 5;
                        }
                        ap[6] = 0;
                        ap[7] = 255;
                    }
                    let mut a48 = 0u64;
                    for k in 0..6 {
                        a48 |= (blk[2 + k] as u64) << (8 * k);
                    }
                    for (k, a) in alpha.iter_mut().enumerate() {
                        *a = ap[((a48 >> (3 * k)) & 7) as usize];
                    }
                }
                _ => {}
            }
            for k in 0..16 {
                let (px, py) = (bx * 4 + k % 4, by * 4 + k / 4);
                if px >= w || py >= h {
                    continue;
                }
                let c = pal[((bits >> (2 * k)) & 3) as usize];
                let o = (py * w + px) * 4;
                out[o] = c[0] as u8;
                out[o + 1] = c[1] as u8;
                out[o + 2] = c[2] as u8;
                out[o + 3] = if fmt == DXT1 { c[3] as u8 } else { alpha[k] as u8 };
            }
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------ encode
fn q565(c: [f64; 3]) -> u16 {
    let q = |v: f64, max: f64| (v.clamp(0.0, 255.0).round() * max / 255.0).round().clamp(0.0, max) as u16;
    (q(c[0], 31.0) << 11) | (q(c[1], 63.0) << 5) | q(c[2], 31.0)
}

fn encode_color(px: &[[u8; 4]; 16], fmt: u32) -> [u8; 8] {
    let rgb: Vec<[f64; 3]> = px.iter().map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect();
    let trans: Vec<bool> = px.iter().map(|p| fmt == DXT1 && p[3] < 128).collect();
    let n_opaque = trans.iter().filter(|t| !**t).count().max(1) as f64;
    let mut mean = [0.0; 3];
    for (c, _) in rgb.iter().zip(&trans).filter(|(_, t)| !**t) {
        for k in 0..3 {
            mean[k] += c[k];
        }
    }
    for m in &mut mean {
        *m /= n_opaque;
    }
    let mut cov = [[0.0f64; 3]; 3];
    for (c, _) in rgb.iter().zip(&trans).filter(|(_, t)| !**t) {
        let d = [c[0] - mean[0], c[1] - mean[1], c[2] - mean[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    let init = 1.0 / 3f64.sqrt();
    let mut ax = [init; 3];
    for _ in 0..8 {
        let v = [0, 1, 2].map(|i| cov[i][0] * ax[0] + cov[i][1] * ax[1] + cov[i][2] * ax[2]);
        let nrm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        ax = if nrm > 1e-9 { [v[0] / nrm, v[1] / nrm, v[2] / nrm] } else { [init; 3] };
    }
    let (mut tmax, mut tmin, mut any) = (f64::NEG_INFINITY, f64::INFINITY, false);
    for (c, _) in rgb.iter().zip(&trans).filter(|(_, t)| !**t) {
        let t = (c[0] - mean[0]) * ax[0] + (c[1] - mean[1]) * ax[1] + (c[2] - mean[2]) * ax[2];
        tmax = tmax.max(t);
        tmin = tmin.min(t);
        any = true;
    }
    if !any {
        tmax = 0.0;
        tmin = 0.0;
    }
    let e0 = [0, 1, 2].map(|k| mean[k] + ax[k] * tmax);
    let e1 = [0, 1, 2].map(|k| mean[k] + ax[k] * tmin);
    let (mut q0, mut q1) = (q565(e0), q565(e1));
    let has_t = trans.iter().any(|t| *t);
    // opaque blocks need c0 > c1 (4-colour mode); blocks with transparent texels need c0 <= c1
    if (has_t && q0 > q1) || (!has_t && q0 < q1) {
        std::mem::swap(&mut q0, &mut q1);
    }
    let same = q0 == q1;
    let (p0, p1) = (c565(q0).map(f64::from), c565(q1).map(f64::from));
    let use3 = has_t || (same && fmt == DXT1);
    let mut pal = [[0.0f64; 3]; 4];
    for k in 0..3 {
        pal[0][k] = p0[k];
        pal[1][k] = p1[k];
        if use3 {
            pal[2][k] = (p0[k] + p1[k]) / 2.0;
            pal[3][k] = p1[k];
        } else {
            pal[2][k] = (2.0 * p0[k] + p1[k]) / 3.0;
            pal[3][k] = (p0[k] + 2.0 * p1[k]) / 3.0;
        }
    }
    let mut bits = 0u32;
    for k in 0..16 {
        let idx = if trans[k] {
            3
        } else if same && !has_t {
            0
        } else {
            let mut best = (0usize, f64::INFINITY);
            for (i, p) in pal.iter().enumerate() {
                if i == 3 && has_t {
                    continue; // index 3 is reserved for transparent texels
                }
                let d = (0..3).map(|c| (rgb[k][c] - p[c]).powi(2)).sum::<f64>();
                if d < best.1 {
                    best = (i, d);
                }
            }
            best.0
        };
        bits |= (idx as u32) << (2 * k);
    }
    let mut out = [0u8; 8];
    out[0..2].copy_from_slice(&q0.to_le_bytes());
    out[2..4].copy_from_slice(&q1.to_le_bytes());
    out[4..8].copy_from_slice(&bits.to_le_bytes());
    out
}

fn alpha_pal(a0: f64, a1: f64, eight: bool) -> [f64; 8] {
    let mut p = [0.0; 8];
    p[0] = a0;
    p[1] = a1;
    if eight {
        for k in 1..7 {
            p[1 + k] = ((7 - k) as f64 * a0 + k as f64 * a1) / 7.0;
        }
    } else {
        for k in 1..5 {
            p[1 + k] = ((5 - k) as f64 * a0 + k as f64 * a1) / 5.0;
        }
        p[6] = 0.0;
        p[7] = 255.0;
    }
    p
}

fn encode_alpha5(a: &[u8; 16]) -> [u8; 8] {
    let af: Vec<f64> = a.iter().map(|&v| v as f64).collect();
    let hi = af.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lo = af.iter().cloned().fold(f64::INFINITY, f64::min);
    let pal_a = alpha_pal(hi, lo, true);
    let mid: Vec<f64> = a.iter().filter(|&&v| v > 0 && v < 255).map(|&v| v as f64).collect();
    let (mlo, mhi) = if mid.is_empty() { (0.0, 0.0) } else { (mid.iter().cloned().fold(f64::INFINITY, f64::min), mid.iter().cloned().fold(f64::NEG_INFINITY, f64::max)) };
    let pal_b = alpha_pal(mlo, mhi, false);
    let nearest = |pal: &[f64; 8], v: f64| {
        let mut best = (0usize, f64::INFINITY);
        for (i, p) in pal.iter().enumerate() {
            let d = (v - p).abs();
            if d < best.1 {
                best = (i, d);
            }
        }
        best
    };
    let (mut ea, mut eb) = (0.0, 0.0);
    let (mut ia, mut ib) = ([0usize; 16], [0usize; 16]);
    for k in 0..16 {
        let (i, d) = nearest(&pal_a, af[k]);
        ia[k] = i;
        ea += d * d;
        let (i, d) = nearest(&pal_b, af[k]);
        ib[k] = i;
        eb += d * d;
    }
    let use_b = eb < ea && mlo <= mhi;
    let (idx, a0, a1) = if use_b { (ib, mlo, mhi) } else { (ia, hi, lo) };
    let mut bits = 0u64;
    for (k, &i) in idx.iter().enumerate() {
        bits |= (i as u64) << (3 * k);
    }
    let mut out = [0u8; 8];
    out[0] = a0.round() as u8;
    out[1] = a1.round() as u8;
    out[2..8].copy_from_slice(&bits.to_le_bytes()[..6]);
    out
}

/// RGBA8 (`w * h * 4`) -> DXT bytes.
pub fn encode_dxt(rgba: &[u8], w: usize, h: usize, fmt: u32) -> Vec<u8> {
    let (bw, bh) = (w.div_ceil(4).max(1), h.div_ceil(4).max(1));
    let mut out = Vec::with_capacity(bw * bh * block_bytes(fmt).unwrap_or(16));
    for by in 0..bh {
        for bx in 0..bw {
            let mut px = [[0u8; 4]; 16];
            for (k, p) in px.iter_mut().enumerate() {
                let (x, y) = ((bx * 4 + k % 4).min(w - 1), (by * 4 + k / 4).min(h - 1));
                p.copy_from_slice(&rgba[(y * w + x) * 4..(y * w + x) * 4 + 4]);
            }
            match fmt {
                DXT1 => out.extend_from_slice(&encode_color(&px, fmt)),
                DXT5 => {
                    out.extend_from_slice(&encode_alpha5(&px.map(|p| p[3])));
                    out.extend_from_slice(&encode_color(&px, fmt));
                }
                _ => {
                    let mut bits = 0u64;
                    for (k, p) in px.iter().enumerate() {
                        bits |= (((p[3] as f64 / 17.0).round() as u64).min(15)) << (4 * k);
                    }
                    out.extend_from_slice(&bits.to_le_bytes());
                    out.extend_from_slice(&encode_color(&px, fmt));
                }
            }
        }
    }
    out
}

fn box_down(rgba: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u8> {
    let (fx, fy) = ((w / nw).max(1), (h / nh).max(1));
    let mut out = vec![0u8; nw * nh * 4];
    for y in 0..nh {
        for x in 0..nw {
            let mut acc = [0u32; 4];
            for sy in 0..fy {
                for sx in 0..fx {
                    let o = ((y * fy + sy).min(h - 1) * w + (x * fx + sx).min(w - 1)) * 4;
                    for c in 0..4 {
                        acc[c] += rgba[o + c] as u32;
                    }
                }
            }
            let n = (fx * fy) as u32;
            for c in 0..4 {
                out[(y * nw + x) * 4 + c] = ((acc[c] + n / 2) / n) as u8;
            }
        }
    }
    out
}

/// Base image plus box-filtered mip levels, each DXT encoded.
pub fn encode_levels(rgba: &[u8], w: usize, h: usize, mips: usize, fmt: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for (lw, lh) in level_dims(w, h, mips) {
        if (lw, lh) == (w, h) {
            out.extend(encode_dxt(rgba, w, h, fmt));
        } else {
            out.extend(encode_dxt(&box_down(rgba, w, h, lw, lh), lw, lh, fmt));
        }
    }
    out
}

// ------------------------------------------------------------------------------------------ io
/// Decode mip `level` (0 = full size) of a texture to RGBA8, with its dimensions.
pub fn read_texture(d: &[u8], t: &Texture, level: usize) -> Result<(Vec<u8>, usize, usize), Error> {
    if level >= t.mips {
        return Err(Error::Texture("mip level out of range".into()));
    }
    let dims = level_dims(t.w, t.h, level + 1);
    let off: usize = t.start + DATA_OFF + dims[..level].iter().map(|&(a, b)| level_bytes(a, b, t.fmt)).sum::<usize>();
    let (lw, lh) = dims[level];
    let n = level_bytes(lw, lh, t.fmt);
    let data = d.get(off..off + n).ok_or_else(|| Error::Texture("texture data is outside the file".into()))?;
    Ok((decode_dxt(data, lw, lh, t.fmt)?, lw, lh))
}

/// Nearest-neighbour / box resize of RGBA8 to `nw x nh` (area average when shrinking).
pub fn resize_rgba(rgba: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u8> {
    if (w, h) == (nw, nh) {
        return rgba.to_vec();
    }
    let mut out = vec![0u8; nw * nh * 4];
    for y in 0..nh {
        let (y0, y1) = (y * h / nh, ((y + 1) * h).div_ceil(nh).clamp(y * h / nh + 1, h));
        for x in 0..nw {
            let (x0, x1) = (x * w / nw, ((x + 1) * w).div_ceil(nw).clamp(x * w / nw + 1, w));
            let mut acc = [0u64; 4];
            for sy in y0..y1 {
                for sx in x0..x1 {
                    for c in 0..4 {
                        acc[c] += rgba[(sy * w + sx) * 4 + c] as u64;
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as u64;
            for c in 0..4 {
                out[(y * nw + x) * 4 + c] = ((acc[c] + n / 2) / n) as u8;
            }
        }
    }
    out
}

fn pad_to(mut b: Vec<u8>) -> Vec<u8> {
    let pad = (ALIGN - b.len() % ALIGN) % ALIGN;
    b.resize(b.len() + pad, 0);
    b
}

#[derive(Clone, Debug)]
pub struct BuildInfo {
    /// Bytes the chunk grew (+) or shrank (-); 0 = in place, nothing else moved.
    pub shift: isize,
    pub w: usize,
    pub h: usize,
    pub mips: usize,
    pub data: Vec<u8>,
}

/// Replace texture `t` with `rgba` (`w * h * 4`). Same size: written in place. New size
/// (`allow_new_size`): the chunk is rebuilt and every chunk-table entry behind it is shifted.
pub fn build_replacement(d: &[u8], t: &Texture, rgba: &[u8], w: usize, h: usize, allow_new_size: bool) -> Result<(Vec<u8>, BuildInfo), Error> {
    if rgba.len() != w * h * 4 {
        return Err(Error::Texture("pixel buffer does not match the given size".into()));
    }
    if (w, h) != (t.w, t.h) {
        if !allow_new_size {
            return Err(Error::Texture(format!("image is {w}x{h} but the texture is {}x{}", t.w, t.h)));
        }
        for v in [w, h] {
            if v < 4 || !v.is_power_of_two() || v > 4096 {
                return Err(Error::Texture(format!("Xbox textures need power-of-two sizes between 4 and 4096 (got {w}x{h})")));
            }
        }
    }
    let max_mips = (usize::BITS - w.min(h).leading_zeros()) as usize;
    let mips = t.mips.clamp(1, max_mips);
    let data = encode_levels(rgba, w, h, mips, t.fmt);
    let old = &d[t.start..t.end];
    let mut head = old[..DATA_OFF].to_vec();
    let mut fd = u32_at(&head, FMT_DWORD).ok_or_else(|| Error::Texture("descriptor missing".into()))?;
    fd = (fd & !(0xF << 16) & !(0xF << 20) & !(0xF << 24)) | ((mips as u32) << 16) | ((w.trailing_zeros()) << 20) | ((h.trailing_zeros()) << 24);
    put_u32(&mut head, FMT_DWORD, fd);
    put_u32(&mut head, MIRROR, t.fmt);
    put_u32(&mut head, MIRROR + 4, w as u32);
    put_u32(&mut head, MIRROR + 8, h as u32);
    let mut chunk = head;
    chunk.extend_from_slice(&data);
    let new_chunk = pad_to(chunk);
    let mut out;
    let shift = new_chunk.len() as isize - old.len() as isize;
    if shift == 0 {
        out = d.to_vec();
        out[t.start..t.end].copy_from_slice(&new_chunk);
    } else {
        out = Vec::with_capacity((d.len() as isize + shift) as usize);
        out.extend_from_slice(&d[..t.start]);
        out.extend_from_slice(&new_chunk);
        out.extend_from_slice(&d[t.end..]);
        let n = u32_at(&out, 0x18).unwrap_or(0) as usize;
        for i in 0..=n {
            let o = 0x1C + 4 * i;
            if let Some(v) = u32_at(&out, o) {
                if v as usize >= t.end {
                    put_u32(&mut out, o, (v as isize + shift) as u32);
                }
            }
        }
    }
    Ok((out, BuildInfo { shift, w, h, mips, data }))
}

/// Re-read a built file and check it. Returns `(ok, report lines)`.
pub fn verify_build(orig: &[u8], new: &[u8], t: &Texture, info: &BuildInfo, rgba: &[u8]) -> (bool, Vec<String>) {
    let mut ok = true;
    let mut lines = Vec::new();
    let mut chk = |cond: bool, text: String| {
        ok &= cond;
        lines.push(format!("{} {text}", if cond { "[OK]  " } else { "[FAIL]" }));
    };
    let shift = info.shift;
    chk(new.len() as isize == orig.len() as isize + shift, format!("file size changed by {shift:+} bytes"));
    chk(new.get(..0x18) == orig.get(..0x18) && u32_at(new, 0x18) == u32_at(orig, 0x18), "db header / chunk count unchanged".into());
    if let (Ok(to), Ok(tn)) = (chunk_table(orig), chunk_table(new)) {
        let exp: Vec<u32> = to.iter().map(|&v| if v as usize >= t.end { (v as isize + shift) as u32 } else { v }).collect();
        chk(tn == exp, format!("chunk table: entries behind the texture shifted by {shift:+}, the rest untouched"));
        chk(tn.iter().all(|v| *v as usize % ALIGN == 0), format!("all chunk offsets still 0x{ALIGN:X}-aligned"));
        let (a, b) = (0x1C, 0x1C + 4 * to.len());
        chk(new[..a] == orig[..a] && new[b..t.start] == orig[b..t.start], "everything before the texture chunk is byte-identical (except the chunk table)".into());
    }
    let tail_ok = new.get((t.end as isize + shift) as usize..) == orig.get(t.end..);
    chk(tail_ok, "everything after the texture chunk is byte-identical".into());
    match find_textures(new) {
        Ok(found) => match found.iter().find(|x| x.start == t.start) {
            Some(x) => {
                chk((x.w, x.h, x.fmt) == (info.w, info.h, t.fmt), format!("descriptor: {}x{} {}, mips {}", x.w, x.h, fmt_name(x.fmt), x.mips));
                let got = new.get(x.start + DATA_OFF..x.start + DATA_OFF + info.data.len());
                chk(got == Some(&info.data[..]), "pixel data in the file equals what was encoded".into());
                if let Ok((dec, _, _)) = read_texture(new, x, 0) {
                    let mut mse = [0f64; 4];
                    for (a, b) in dec.chunks_exact(4).zip(rgba.chunks_exact(4)) {
                        for c in 0..4 {
                            mse[c] += (a[c] as f64 - b[c] as f64).powi(2);
                        }
                    }
                    let n = (rgba.len() / 4).max(1) as f64;
                    let psnr = mse.map(|m| 10.0 * (255.0f64 * 255.0 / (m / n).max(1e-9)).log10());
                    lines.push(format!("       quality (PSNR dB): R {:.1}  G {:.1}  B {:.1}  A {:.1}", psnr[0], psnr[1], psnr[2], psnr[3]));
                }
            }
            None => chk(false, "texture chunk still parses as a texture".into()),
        },
        Err(e) => chk(false, format!("rebuilt file does not parse: {e}")),
    }
    (ok, lines)
}
