//! Black (2006 Xbox) `.db` support in the UI:
//! * a small window that asks which part to load (gun, arm or both) when a `.db` is opened,
//! * the Export window's "Rebuild .db" mode (pick the `.db`, pick which texture each texture set replaces).
//! All decisions live in `subcraft_app::black_db` / `Session`; this file draws and dispatches.

use std::path::{Path, PathBuf};

use egui::RichText;
use subcraft_app::black_db::BlackDbInfo;
use subcraft_mesh::BlackPart;

use crate::state::UiState;
use crate::theme::TEXT_DIM;
use crate::widgets::{caption, divider, primary_button};

pub const DB_FILES: (&str, &[&str]) = ("Black .db", &["db"]);

pub fn is_db(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("db"))
}

/// What to do once the part is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum After {
    OpenMesh,
    NewProject,
}

#[derive(Default)]
pub struct BlackDbUi {
    /// A `.db` waiting for the user to choose what to load.
    pub pending: Option<(PathBuf, After)>,
    /// Export window: rebuild a `.db` instead of writing map files.
    pub export_as_db: bool,
    pub db_in: String,
    pub db_out: String,
    pub info: Option<BlackDbInfo>,
    /// Target texture (index into `info.textures`) for each texture set.
    pub targets: Vec<usize>,
    pub allow_new_size: bool,
    pub invert_alpha: bool,
    pub error: Option<String>,
    pub report: Vec<String>,
}

impl BlackDbUi {
    fn load_info(&mut self, session: &subcraft_app::Session) {
        self.report.clear();
        match session.black_db_info(Path::new(&self.db_in)) {
            Ok(info) => {
                self.targets = info.defaults.clone();
                self.info = Some(info);
                self.error = None;
                if self.db_out.is_empty() {
                    // Same name as the original: ready to drop into the game (the original is backed up).
                    self.db_out = self.db_in.clone();
                }
            }
            Err(e) => {
                self.info = None;
                self.error = Some(e);
            }
        }
    }
}

/// Ask which part of `path` to load. Called when a `.db` is picked.
pub fn ask_part(state: &mut UiState, path: &Path, after: After) {
    state.black_db.pending = Some((path.to_path_buf(), after));
}

/// The "Load Black .db" window (draw every frame).
pub fn part_window(state: &mut UiState, ctx: &egui::Context, now: f64) {
    let Some((path, after)) = state.black_db.pending.clone() else { return };
    let mut choice = None;
    let mut open = true;
    crate::widgets::window(ctx, RichText::new("Load Black .db")).open(&mut open).collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
        ui.label(RichText::new(path.display().to_string()).color(TEXT_DIM).small());
        ui.add_space(6.0);
        ui.label("What do you want to load?");
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if primary_button(ui, "Gun only").clicked() {
                choice = Some(BlackPart::Gun);
            }
            if ui.button("Arm only").clicked() {
                choice = Some(BlackPart::Arm);
            }
            if ui.button("Both").clicked() {
                choice = Some(BlackPart::Both);
            }
        });
    });
    if !open {
        state.black_db.pending = None;
    }
    if let Some(part) = choice {
        state.black_db.pending = None;
        subcraft_mesh::set_black_part(part);
        match after {
            After::NewProject => state.new_project.load_model(&path),
            After::OpenMesh => match subcraft_app::Session::open_mesh(&path, subcraft_app::DEFAULT_RESOLUTION) {
                Ok((s, warnings)) => {
                    state.set_session(s);
                    state.welcome.remember(&path);
                    let tex = warnings.iter().find(|w| w.starts_with("Textures")).map(|w| format!(". {w}")).unwrap_or_default();
                    state.set_status(format!("Opened {} ({part:?}){tex}", path.file_name().and_then(|n| n.to_str()).unwrap_or("file")), now);
                }
                Err(e) => state.set_status(format!("Could not open .db: {e}"), now),
            },
        }
    }
}

/// Import the textures of a `.db` into the selected texture set as one fill layer.
pub fn import_textures(state: &mut UiState, path: &Path, now: f64) {
    match state.session.import_black_db_texture(path) {
        Ok(msg) => state.set_status(msg, now),
        Err(e) => state.set_status(format!("Import from .db: {e}"), now),
    }
}

/// Maps / .db switch at the top of the Export window. Returns true when `.db` mode is on.
pub fn mode_switch(ui: &mut egui::Ui, state: &mut UiState) -> bool {
    ui.horizontal(|ui| {
        ui.label(RichText::new("Export as").color(TEXT_DIM));
        ui.selectable_value(&mut state.black_db.export_as_db, false, "Maps (image files)");
        ui.selectable_value(&mut state.black_db.export_as_db, true, "Rebuild Black .db");
    });
    ui.add_space(4.0);
    state.black_db.export_as_db
}

/// The Export window body in `.db` mode.
pub fn export_panel(ui: &mut egui::Ui, state: &mut UiState, now: f64) {
    let mut pick_in = false;
    let mut pick_out = false;
    let mut load = false;
    {
        let b = &mut state.black_db;
        egui::Grid::new("black_db_grid").num_columns(2).spacing([16.0, 10.0]).show(ui, |ui| {
            ui.label(RichText::new("Original .db").color(TEXT_DIM));
            ui.horizontal(|ui| {
                if ui.add(egui::TextEdit::singleline(&mut b.db_in).desired_width(300.0)).lost_focus() && !b.db_in.is_empty() {
                    load = true;
                }
                if ui.button("Browse…").clicked() {
                    pick_in = true;
                }
            });
            ui.end_row();
            ui.label(RichText::new("Save .db as").color(TEXT_DIM));
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut b.db_out).desired_width(300.0));
                if ui.button("Browse…").clicked() {
                    pick_out = true;
                }
            });
            ui.end_row();
        });
    }
    if pick_in
        && let Some(p) = rfd::FileDialog::new().add_filter(DB_FILES.0, DB_FILES.1).pick_file()
    {
        state.black_db.db_in = p.display().to_string();
        state.black_db.db_out.clear();
        load = true;
    }
    if pick_out
        && let Some(p) = rfd::FileDialog::new().add_filter(DB_FILES.0, DB_FILES.1).save_file()
    {
        state.black_db.db_out = p.display().to_string();
    }
    if load {
        let session = &state.session;
        let mut b = std::mem::take(&mut state.black_db);
        b.load_info(session);
        state.black_db = b;
    }
    let b = &mut state.black_db;
    if let Some(e) = &b.error {
        ui.colored_label(crate::theme::WARNING, e);
    }
    ui.add_space(4.0);
    divider(ui);
    ui.add_space(4.0);
    let mut go = false;
    match &b.info {
        None => {
            ui.label(caption("Choose the original .db: the textures painted here replace its textures. Only texture chunks change; the mesh and every pointer stay as they were."));
        }
        Some(info) => {
            ui.label(caption("Each texture set replaces one texture of the .db (BaseColor map in RGB, Alpha map = the Metallic channel)."));
            ui.add_space(4.0);
            egui::Grid::new("black_db_targets").num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
                for (i, set) in info.sets.iter().enumerate() {
                    ui.label(RichText::new(set).strong());
                    let cur = b.targets.get(i).copied().unwrap_or(0).min(info.textures.len() - 1);
                    if b.targets.len() <= i {
                        b.targets.resize(i + 1, cur);
                    }
                    egui::ComboBox::from_id_salt(("black_db_target", i)).width(340.0).selected_text(&info.textures[cur]).show_ui(ui, |ui| {
                        for (n, label) in info.textures.iter().enumerate() {
                            ui.selectable_value(&mut b.targets[i], n, label);
                        }
                    });
                    ui.end_row();
                }
            });
            ui.add_space(4.0);
            ui.checkbox(&mut b.allow_new_size, "Allow a new texture size (rebuilds the chunk, shifts the chunk table)")
                .on_hover_text("Off: the image is resized to the original texture size and written in place, nothing moves.");
            ui.checkbox(&mut b.invert_alpha, "Invert alpha")
                .on_hover_text("Flip the Alpha map (Metallic channel) before it is written. Like --invert-alpha in black_texture_tool.");
            let dup = {
                let mut t = b.targets.clone();
                t.sort_unstable();
                t.windows(2).any(|w| w[0] == w[1])
            };
            if dup {
                ui.colored_label(crate::theme::WARNING, "Two texture sets target the same texture; the later one wins.");
            }
        }
    }
    ui.add_space(6.0);
    let ready = b.info.is_some() && !b.db_out.is_empty();
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        if b.db_out == b.db_in && !b.db_in.is_empty() {
            ui.label(RichText::new("Same name as the original: it is backed up to .db.bak first.").color(TEXT_DIM));
        }
        ui.add_enabled_ui(ready, |ui| go = primary_button(ui, format!("{}  Rebuild .db", crate::icons::EXPORT)).clicked());
    });
    if !b.report.is_empty() {
        ui.add_space(4.0);
        egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
            for l in &b.report {
                ui.label(RichText::new(l).monospace().small());
            }
        });
    }
    if go {
        let (db_in, db_out, targets, allow) = (PathBuf::from(&b.db_in), PathBuf::from(&b.db_out), b.targets.clone(), b.allow_new_size);
        let invert = b.invert_alpha;
        let settings = state.export.settings.clone();
        match state.session.export_black_db(&db_in, &db_out, &targets, &settings, allow, invert) {
            Ok(report) => {
                state.black_db.report = report;
                state.black_db.error = None;
                state.set_status(format!("Rebuilt {}", db_out.display()), now);
            }
            Err(e) => {
                state.black_db.error = Some(e.clone());
                state.set_status(format!("Rebuild failed: {e}"), now);
            }
        }
    }
}
