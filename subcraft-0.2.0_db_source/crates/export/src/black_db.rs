//! Rebuild a Black (2006 Xbox) `.db` with the textures painted in SubCraft, following
//! `black_texture_tool.py replace`:
//!
//! * the texture set is exported as two maps, `BaseColor` (RGB) and `Alpha` (the Metallic channel),
//! * both are resized to the texture's size (or kept at their size with `allow_new_size`),
//! * alpha may be inverted, no alpha map means fully opaque,
//! * the pixels are DXT encoded with box-filtered mips and written into the texture chunk
//!   (same size: in place; new size: chunk rebuilt, chunk table shifted),
//! * the written file is re-read and checked; a failed check writes nothing.
//!
//! The mesh chunk is never touched, so every mesh pointer stays valid. Saving over the original
//! first copies it to `name.db.bak`, so the rebuilt file can be dropped straight into the game.

use std::path::{Path, PathBuf};

use subcraft_blackdb::texture;
use subcraft_mesh::Mesh;

use crate::{ExportError, ExportSettings, ExportTemplate, ExportTextureSet, Padding};

fn err(path: &Path, msg: impl ToString) -> ExportError {
    ExportError::Write(PathBuf::from(path), msg.to_string())
}

/// Replace texture `texture_index` of `db` with the `BaseColor` and `Alpha` PNGs. Returns the new
/// file and the verification report.
pub fn replace_texture_maps(db: &[u8], texture_index: usize, color_png: &[u8], alpha_png: Option<&[u8]>, invert_alpha: bool, allow_new_size: bool) -> Result<(Vec<u8>, Vec<String>), String> {
    let color = image::load_from_memory(color_png).map_err(|e| format!("cannot read exported BaseColor PNG: {e}"))?.to_rgba8();
    let (cw, ch) = (color.width() as usize, color.height() as usize);
    let textures = texture::find_textures(db).map_err(|e| e.to_string())?;
    let t = textures.get(texture_index).ok_or_else(|| format!("the .db has no texture #{texture_index} ({} found)", textures.len()))?;
    let (w, h) = if allow_new_size { (cw, ch) } else { (t.w, t.h) };
    let mut rgba = texture::resize_rgba(color.as_raw(), cw, ch, w, h);
    let mut notes = Vec::new();
    if (cw, ch) != (w, h) {
        notes.push(format!("note: BaseColor {cw}x{ch} resized to {w}x{h}"));
    }
    match alpha_png {
        Some(png) => {
            let a = image::load_from_memory(png).map_err(|e| format!("cannot read exported Alpha PNG: {e}"))?.to_luma8();
            let (aw, ah) = (a.width() as usize, a.height() as usize);
            let gray: Vec<u8> = a.as_raw().iter().flat_map(|&v| [v, v, v, 255]).collect();
            let gray = texture::resize_rgba(&gray, aw, ah, w, h);
            for (px, g) in rgba.chunks_exact_mut(4).zip(gray.chunks_exact(4)) {
                px[3] = if invert_alpha { 255 - g[0] } else { g[0] };
            }
            if (aw, ah) != (w, h) {
                notes.push(format!("note: Alpha {aw}x{ah} resized to {w}x{h}"));
            }
        }
        None => {
            rgba.chunks_exact_mut(4).for_each(|p| p[3] = 255);
            notes.push("note: no Alpha map (the set has no Metallic channel): texture is fully opaque".into());
        }
    }
    if t.fmt == texture::DXT1 {
        // DXT1 alpha is 1-bit transparency, not a gloss map: keep the texture fully opaque.
        if alpha_png.is_some() {
            notes.push("note: this texture is DXT1 (1-bit alpha), so the Alpha map was not stored".into());
        }
        rgba.chunks_exact_mut(4).for_each(|p| p[3] = 255);
    }
    let (new, info) = texture::build_replacement(db, t, &rgba, w, h, allow_new_size).map_err(|e| e.to_string())?;
    let (ok, mut lines) = texture::verify_build(db, &new, t, &info, &rgba);
    notes.append(&mut lines);
    if !ok {
        notes.insert(0, "verification failed, nothing written".into());
        return Err(notes.join("\n"));
    }
    Ok((new, notes))
}

fn same_file(a: &Path, b: &Path) -> bool {
    a == b || matches!((std::fs::canonicalize(a), std::fs::canonicalize(b)), (Ok(x), Ok(y)) if x == y)
}

/// `name.db.bak`, or `name.db.bak2`, `.bak3`... when earlier backups exist.
fn free_backup_path(db: &Path) -> PathBuf {
    let name = db.file_name().and_then(|n| n.to_str()).unwrap_or("texture.db");
    let mut n = 1;
    loop {
        let candidate = db.with_file_name(if n == 1 { format!("{name}.bak") } else { format!("{name}.bak{n}") });
        if !candidate.exists() {
            return candidate;
        }
        n += 1;
    }
}

/// Export `sets` and rebuild `db_in` into `db_out`: set `i` replaces texture `textures[i]`.
/// Returns the verification report lines.
pub fn export_black_db(mesh: &Mesh, mesh_name: &str, sets: &[ExportTextureSet], textures: &[usize], settings: &ExportSettings, db_in: &Path, db_out: &Path, allow_new_size: bool, invert_alpha: bool) -> Result<Vec<String>, ExportError> {
    if textures.len() != sets.len() {
        return Err(err(db_out, "every texture set needs a target texture in the .db"));
    }
    let original = subcraft_core::vfs::read(db_in).map_err(|e| err(db_in, e))?;
    let mut db = original.clone();
    let mut s = settings.clone();
    s.template = ExportTemplate::black_xbox();
    s.format_override = Some(subcraft_image::FileFormat::Png8);
    s.selection = crate::MapSelection::default(); // the map checklist belongs to the image export
    if s.padding == Padding::None {
        s.padding = Padding::Infinite; // DXT blocks and mips bleed across island edges
    }
    let mut report = Vec::new();
    for (i, set) in sets.iter().enumerate() {
        let (mut color, mut alpha) = (None, None);
        crate::render_maps(mesh, mesh_name, &sets[i..=i], &s, |name, bytes| {
            let file = name.rsplit('/').next().unwrap_or(name);
            if file.ends_with("_BaseColor.png") {
                color = Some(bytes.to_vec());
            } else if file.ends_with("_Alpha.png") {
                alpha = Some(bytes.to_vec());
            }
            Ok(())
        })?;
        let color = color.ok_or_else(|| err(db_out, format!("texture set '{}' has no Base Color channel to export", set.name)))?;
        let (new, lines) = replace_texture_maps(&db, textures[i], &color, alpha.as_deref(), invert_alpha, allow_new_size).map_err(|e| err(db_out, e))?;
        report.push(format!("texture set '{}' -> texture #{}", set.name, textures[i]));
        report.extend(lines);
        db = new;
    }
    if let Some(parent) = db_out.parent().filter(|p| !p.as_os_str().is_empty()) {
        subcraft_core::vfs::create_dir_all(parent).map_err(|e| err(parent, e))?;
    }
    // Writing over the original: keep a copy first (never over an earlier backup), so the first
    // backup stays the untouched original.
    if same_file(db_in, db_out) {
        let backup = free_backup_path(db_in);
        subcraft_core::vfs::write(&backup, &original).map_err(|e| err(&backup, format!("could not back up the original first: {e}")))?;
        report.push(format!("backup of the original: {}", backup.display()));
    }
    subcraft_core::vfs::write(db_out, &db).map_err(|e| err(db_out, e))?;
    report.push(format!("written: {}", db_out.display()));
    Ok(report)
}
