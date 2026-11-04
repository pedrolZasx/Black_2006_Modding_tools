//! Black (2006 Xbox) `.db` helpers for the app: read the textures inside a `.db`, turn one into
//! SubCraft maps (base color from RGB, gloss/metal map from alpha) and import it as a fill layer.

use std::path::Path;

use subcraft_blackdb::texture;
use subcraft_core::ChannelKind;
use subcraft_doc::{FillChannel, FillSource, Layer, LayerContent};
use subcraft_image::FileFormat;

use crate::command::Command;
use crate::session::Session;

/// What the export window needs to know about a chosen `.db`.
#[derive(Clone, Debug, Default)]
pub struct BlackDbInfo {
    /// One label per texture found in the `.db` (index = texture number).
    pub textures: Vec<String>,
    /// The project's texture sets, in export order.
    pub sets: Vec<String>,
    /// Suggested texture for each set (matched by name, else the largest texture).
    pub defaults: Vec<usize>,
}

/// Read the textures of `db` and suggest a target texture for each of `set_names`.
pub fn read_info(db: &Path, set_names: &[String]) -> Result<BlackDbInfo, String> {
    let bytes = subcraft_core::vfs::read(db).map_err(|e| format!("{}: {e}", db.display()))?;
    let textures = texture::find_textures(&bytes).map_err(|e| e.to_string())?;
    if textures.is_empty() {
        return Err("no DXT1/DXT3/DXT5 textures found in this .db".into());
    }
    let defaults = set_names.iter().map(|n| texture::pick_for_group(&textures, n).map_or(0, |(i, _)| i)).collect();
    Ok(BlackDbInfo { textures: textures.iter().map(|t| format!("#{}  {}", t.index, t.label())).collect(), sets: set_names.to_vec(), defaults })
}

struct DbTexture {
    name: String,
    color_png: Vec<u8>,
    /// `None` when the alpha channel is fully opaque (nothing to import as gloss).
    gloss_png: Option<Vec<u8>>,
    note: String,
    count: usize,
}

fn decode_texture(db: &Path, set_name: &str) -> Result<DbTexture, String> {
    let bytes = subcraft_core::vfs::read(db).map_err(|e| format!("{}: {e}", db.display()))?;
    let textures = texture::find_textures(&bytes).map_err(|e| e.to_string())?;
    let (idx, note) = texture::pick_for_group(&textures, set_name).ok_or("no DXT1/DXT3/DXT5 textures found in this .db")?;
    let t = &textures[idx];
    let (rgba, w, h) = texture::read_texture(&bytes, t, 0).map_err(|e| e.to_string())?;
    let rgb: Vec<f32> = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).map(|v| v as f32 / 255.0).collect();
    let color_png = subcraft_image::io::encode_image(w as u32, h as u32, 3, &rgb, FileFormat::Png8).map_err(|e| e.to_string())?;
    let gloss_png = if rgba.chunks_exact(4).all(|p| p[3] == 255) {
        None
    } else {
        let a: Vec<f32> = rgba.chunks_exact(4).map(|p| p[3] as f32 / 255.0).collect();
        Some(subcraft_image::io::encode_image(w as u32, h as u32, 1, &a, FileFormat::Png8).map_err(|e| e.to_string())?)
    };
    let name = if t.name.is_empty() { format!("texture{}", t.index) } else { t.name.clone() };
    Ok(DbTexture { name, color_png, gloss_png, note, count: textures.len() })
}

impl Session {
    /// Import the `.db` texture that belongs to the selected texture set as one fill layer:
    /// RGB -> Base Color, alpha (the grayscale gloss/metal map of the 2006 engine) -> Metallic.
    /// One undo step. Returns a status line.
    pub fn import_black_db_texture(&mut self, path: &Path) -> Result<String, String> {
        let set_name = self.selected_texture_set().name.clone();
        let tex = decode_texture(path, &set_name)?;
        let ts = self.selected_texture_set().clone();
        let color = self.import_image_bytes(&format!("{}_basecolor.png", tex.name), tex.color_png)?;
        let gloss = match tex.gloss_png {
            Some(bytes) => Some(self.import_image_bytes(&format!("{}_gloss.png", tex.name), bytes)?),
            None => None,
        };
        let mut layer = Layer::new_fill(tex.name.clone(), &ts);
        let LayerContent::Fill(fill) = &mut layer.content else { unreachable!("new_fill makes a fill layer") };
        for c in fill.channels.values_mut() {
            c.enabled = false;
        }
        fill.channels.insert(ChannelKind::BaseColor, FillChannel::new(FillSource::Bitmap { resource: color }));
        let mut parts = vec!["Base Color".to_string()];
        match gloss {
            Some(g) if ts.has_channel(ChannelKind::Metallic) => {
                fill.channels.insert(ChannelKind::Metallic, FillChannel::new(FillSource::Bitmap { resource: g }));
                parts.push("Metallic (from alpha)".into());
            }
            Some(_) => parts.push("alpha skipped: the texture set has no Metallic channel".into()),
            None => {}
        }
        self.execute(Command::InsertLayer { layer: Box::new(layer), label: format!("Import {}", tex.name) }).map_err(|e| e.to_string())?;
        log::info!("black .db import: {}", tex.note);
        Ok(format!("Imported \"{}\" from .db ({} texture(s) in file): {}", tex.name, tex.count, parts.join(", ")))
    }
}

impl Session {
    /// Import the `.db` textures for every texture set (each set takes the texture that matches its
    /// name), then go back to the set that was selected. Used when a `.db` is loaded so the gun
    /// shows its own texture at once. Returns one status line.
    pub fn import_black_db_all(&mut self, path: &Path) -> String {
        let original = self.selection().texture_set;
        let sets: Vec<(subcraft_core::TextureSetId, String)> = self.document().texture_sets.iter().map(|t| (t.id, t.name.clone())).collect();
        let (mut done, mut notes) = (0usize, Vec::new());
        for (id, name) in &sets {
            if self.execute(Command::SelectTextureSet(*id)).is_err() {
                continue;
            }
            match self.import_black_db_texture(path) {
                Ok(_) => done += 1,
                Err(e) => notes.push(format!("{name}: {e}")),
            }
        }
        let _ = self.execute(Command::SelectTextureSet(original));
        let mut line = format!("Textures from .db assigned to {done} of {} texture set(s) (Base Color, alpha as Metallic)", sets.len());
        if !notes.is_empty() {
            line.push_str(&format!("; {}", notes.join("; ")));
        }
        line
    }
}
