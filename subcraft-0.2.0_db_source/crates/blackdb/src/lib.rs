//! Black (2006, Xbox) `.db` archives: the first-person weapon chunk (mesh) and its DXT textures.
//!
//! * [`mesh`] reads the skinned weapon/arm groups of chunk 1 (ported from `black_fps_extract_fbx`).
//!   Skin data is ignored here; SubCraft only needs positions, normals and UVs to paint.
//! * [`texture`] finds, decodes, encodes and rebuilds DXT1/3/5 texture chunks (ported from
//!   `black_texture_tool`). The rebuild never touches the mesh chunk. When the new texture has the
//!   same size and format the pixels are written in place and no pointer or chunk offset moves.

pub mod mesh;
pub mod texture;

/// Chunks inside a `.db` start on multiples of this.
pub const ALIGN: usize = 0x80;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("not a Black .db: {0}")]
    NotDb(String),
    #[error("cannot read the weapon chunk: {0}")]
    Mesh(String),
    #[error("{0}")]
    Texture(String),
}

pub(crate) fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    b.get(o..o.checked_add(4)?).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

pub(crate) fn f32_at(b: &[u8], o: usize) -> Option<f32> {
    u32_at(b, o).map(f32::from_bits)
}

pub(crate) fn put_u32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

/// Chunk start offsets: word 0x18 = count, table at 0x1C with `count + 1` entries (not sorted).
pub fn chunk_table(d: &[u8]) -> Result<Vec<u32>, Error> {
    let n = u32_at(d, 0x18).ok_or_else(|| Error::NotDb("file too short".into()))? as usize;
    if !(0 < n && n < 4096) || 0x1C + 4 * (n + 1) > d.len() {
        return Err(Error::NotDb("bad chunk count".into()));
    }
    Ok((0..=n).filter_map(|i| u32_at(d, 0x1C + 4 * i)).collect())
}

/// Every chunk as `(start, end)`, sorted by offset.
pub fn chunk_ranges(d: &[u8]) -> Result<Vec<(usize, usize)>, Error> {
    let mut offs: Vec<usize> = chunk_table(d)?.into_iter().map(|v| v as usize).collect();
    offs.push(d.len());
    offs.sort_unstable();
    offs.dedup();
    Ok(offs.windows(2).map(|w| (w[0], w[1])).collect())
}

/// Chunk 1 (the weapon chunk) of a `.db`; a raw `chunk01_skeleton.bin` is the chunk itself.
pub fn locate_chunk1(d: &[u8], is_db: bool) -> Result<(usize, usize), Error> {
    if !is_db {
        return Ok((0, d.len()));
    }
    let start = chunk_table(d)?.get(1).copied().ok_or_else(|| Error::NotDb("no chunk 1".into()))? as usize;
    let ranges = chunk_ranges(d)?;
    ranges.iter().find(|r| r.0 == start).map(|r| (r.0, r.1)).ok_or_else(|| Error::NotDb("chunk 1 offset is outside the file".into()))
}

/// True when the bytes look like a `.db` archive (chunk table plausible).
pub fn looks_like_db(d: &[u8]) -> bool {
    chunk_table(d).is_ok()
}

#[cfg(test)]
mod tests;
