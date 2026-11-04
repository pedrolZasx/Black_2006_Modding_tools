//! Weapon / arm geometry of chunk 1 (ported from `black_fps_model_replace_v6.parse_chunk` and
//! `black_fps_extract_fbx.group_skinned_mesh`).
//!
//! Layout: group records (type 6, index count, relative index offset) sit 0x30 apart, followed by one
//! vertex buffer (36-byte vertices: position at 0, skin bytes at 0x0C, NV2A 11-11-10 normal at 0x18,
//! UV at 0x1C) and one `u16` triangle strip per group. UV `v` is stored negated.

use crate::{Error, f32_at, u32_at};

pub const STRIDE: usize = 36;
pub const BONE_OFF: usize = 0x0C;
pub const NRM_OFF: usize = 0x18;
pub const UV_OFF: usize = 0x1C;

#[derive(Clone, Debug)]
pub struct Group {
    pub record_off: usize,
    pub index_count: usize,
    pub index_abs_off: usize,
    pub indices: Vec<u16>,
    /// Sorted distinct vertex slots the group's strip uses.
    pub slots: Vec<u16>,
    pub name: String,
}

impl Group {
    /// The first-person arms group (everything else is the weapon).
    pub fn is_arm(&self) -> bool {
        self.name.to_ascii_lowercase().contains("arm")
    }
}

#[derive(Clone, Debug)]
pub struct ChunkInfo {
    pub container_base: usize,
    pub groups: Vec<Group>,
    pub vstart: usize,
    pub total_verts: usize,
}

/// Which groups to load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Part {
    /// Weapon and arms.
    #[default]
    Both,
    /// Weapon only (`--part gun`).
    Gun,
    Arm,
}

impl Part {
    pub fn parse(s: &str) -> Option<Part> {
        match s.to_ascii_lowercase().as_str() {
            "both" | "all" => Some(Part::Both),
            "gun" | "weapon" => Some(Part::Gun),
            "arm" | "arms" => Some(Part::Arm),
            _ => None,
        }
    }

    pub fn includes(self, g: &Group) -> bool {
        match self {
            Part::Both => true,
            Part::Gun => !g.is_arm(),
            Part::Arm => g.is_arm(),
        }
    }
}

fn scan(c1: &[u8], lo: usize, hi: usize) -> Vec<Group> {
    let mut found = Vec::new();
    let limit = hi.min(c1.len().saturating_sub(0x2C));
    let mut i = lo;
    while i < limit {
        if let (Some(typ), Some(cnt), Some(off)) = (u32_at(c1, i + 0x20), u32_at(c1, i + 0x24), u32_at(c1, i + 0x28)) {
            let (cnt, off) = (cnt as usize, off as usize);
            if typ == 6 && 100 < cnt && cnt < 100_000 && 0x1000 < off && off < c1.len() && i + off + 2 * cnt <= c1.len() {
                found.push(Group { record_off: i, index_count: cnt, index_abs_off: i + off, indices: Vec::new(), slots: Vec::new(), name: String::new() });
            }
        }
        i += 16;
    }
    found
}

/// Names like `bg1_pst_gun` stored near the end of the chunk, in order of the groups.
fn group_names(c1: &[u8]) -> Vec<String> {
    let tail = &c1[c1.len().saturating_sub(0x1000)..];
    let is_w = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= tail.len() {
        if &tail[i..i + 4] == b"bg1_" {
            let mut j = i + 4;
            while j < tail.len() && is_w(tail[j]) {
                j += 1;
            }
            out.push(String::from_utf8_lossy(&tail[i..j]).into_owned());
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// Locate the geometry groups, vertex buffer start and each group's vertex slots.
pub fn parse_chunk(c1: &[u8]) -> Result<ChunkInfo, Error> {
    let mut container_base = if c1.len() > 0xA0 { u32_at(c1, 0x98).unwrap_or(0) as usize } else { 0x2480 };
    if container_base == 0 || container_base >= c1.len() {
        container_base = 0x2480;
    }
    // 1) the classic window (pistol-style chunks)
    let mut groups = scan(c1, container_base + 0x1000, container_base + 0x2000);
    // 2) fallback (shotgun etc.): longest run of records spaced 0x30 apart anywhere in the chunk
    if groups.is_empty() {
        let cand = scan(c1, 0, c1.len());
        let mut runs: Vec<Vec<Group>> = Vec::new();
        let mut cur: Vec<Group> = Vec::new();
        for g in cand {
            if cur.last().is_some_and(|l| g.record_off - l.record_off == 0x30) {
                cur.push(g);
            } else {
                if !cur.is_empty() {
                    runs.push(std::mem::take(&mut cur));
                }
                cur.push(g);
            }
        }
        if !cur.is_empty() {
            runs.push(cur);
        }
        if let Some(best) = runs.into_iter().max_by_key(|r| (r.len(), r[0].index_count)) {
            container_base = best[0].record_off.saturating_sub(0x1000);
            groups = best;
        }
    }
    if groups.is_empty() {
        return Err(Error::Mesh("could not locate geometry group records in the chunk".into()));
    }
    let vstart = (groups[groups.len() - 1].record_off + 0x30 + 15) & !15;
    let first_idx = groups[0].index_abs_off;
    if first_idx < vstart {
        return Err(Error::Mesh("index buffer starts before the vertex buffer".into()));
    }
    let total_verts = (first_idx - vstart) / STRIDE;
    let names = group_names(c1);
    let n_groups = groups.len();
    for (gi, g) in groups.iter_mut().enumerate() {
        g.indices = (0..g.index_count).map(|k| c1.get(g.index_abs_off + 2 * k..g.index_abs_off + 2 * k + 2).map(|s| u16::from_le_bytes([s[0], s[1]])).unwrap_or(0)).collect();
        let mut slots = g.indices.clone();
        slots.sort_unstable();
        slots.dedup();
        g.slots = slots;
        g.name = if names.len() >= n_groups { names[gi].clone() } else { format!("submesh_{gi:02}") };
    }
    Ok(ChunkInfo { container_base, groups, vstart, total_verts })
}

/// Decode a `u16` triangle strip (odd triangles flipped, degenerates skipped).
pub fn strip_to_tris(indices: &[u16], num_verts: usize) -> Vec<[u16; 3]> {
    let mut tris = Vec::new();
    for k in 0..indices.len().saturating_sub(2) {
        let (a, b, c) = (indices[k], indices[k + 1], indices[k + 2]);
        if a == b || b == c || a == c || a.max(b).max(c) as usize >= num_verts {
            continue;
        }
        tris.push(if k % 2 == 0 { [a, b, c] } else { [b, a, c] });
    }
    tris
}

/// NV2A 11-11-10 packed normal.
pub fn unpack_normal(w: u32) -> [f32; 3] {
    let (mut ix, mut iy, mut iz) = ((w & 0x7FF) as i32, ((w >> 11) & 0x7FF) as i32, ((w >> 22) & 0x3FF) as i32);
    if ix >= 0x400 {
        ix -= 0x800;
    }
    if iy >= 0x400 {
        iy -= 0x800;
    }
    if iz >= 0x200 {
        iz -= 0x400;
    }
    let n = [ix as f32 / 1023.0, iy as f32 / 1023.0, iz as f32 / 511.0];
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l > 0.0 { [n[0] / l, n[1] / l, n[2] / l] } else { n }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// UV exactly as stored (`v` is negated relative to OBJ).
    pub uv: [f32; 2],
}

pub fn read_vertex(c1: &[u8], info: &ChunkInfo, slot: u16) -> Option<Vertex> {
    let vo = info.vstart + slot as usize * STRIDE;
    Some(Vertex {
        position: [f32_at(c1, vo)?, f32_at(c1, vo + 4)?, f32_at(c1, vo + 8)?],
        normal: unpack_normal(u32_at(c1, vo + NRM_OFF)?),
        uv: [f32_at(c1, vo + UV_OFF)?, f32_at(c1, vo + UV_OFF + 4)?],
    })
}

/// One loaded group as plain triangles.
#[derive(Clone, Debug)]
pub struct GroupMesh {
    pub name: String,
    pub triangles: Vec<[Vertex; 3]>,
}

/// A parsed `.db` / chunk with the weapon geometry.
#[derive(Clone, Debug)]
pub struct BlackModel {
    pub chunk_start: usize,
    pub chunk_end: usize,
    pub info: ChunkInfo,
}

impl BlackModel {
    pub fn parse(data: &[u8], is_db: bool) -> Result<Self, Error> {
        let (s, e) = crate::locate_chunk1(data, is_db)?;
        let info = parse_chunk(&data[s..e])?;
        Ok(Self { chunk_start: s, chunk_end: e, info })
    }

    /// Triangles of the groups selected by `part`. Non-finite or out-of-range vertices drop their triangle.
    pub fn groups(&self, data: &[u8], part: Part) -> Vec<GroupMesh> {
        let c1 = &data[self.chunk_start..self.chunk_end];
        self.info
            .groups
            .iter()
            .filter(|g| part.includes(g))
            .map(|g| {
                let triangles = strip_to_tris(&g.indices, self.info.total_verts)
                    .into_iter()
                    .filter_map(|t| {
                        let v = [read_vertex(c1, &self.info, t[0])?, read_vertex(c1, &self.info, t[1])?, read_vertex(c1, &self.info, t[2])?];
                        v.iter().all(|x| x.position.iter().all(|c| c.is_finite())).then_some(v)
                    })
                    .collect();
                GroupMesh { name: g.name.clone(), triangles }
            })
            .collect()
    }
}
