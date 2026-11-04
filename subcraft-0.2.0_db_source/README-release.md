# SubCraft – Black .db edition

SubCraft is an open-source 3D texture painting app. This release adds support for the first-person
weapon models of **Black (2006, Xbox)**: open the game's `.db`, paint the gun, and write the result
straight back into a `.db` you can run in the game.

> Unofficial fan tool for modding your own copy of the game. Not affiliated with or endorsed by the
> game's developers or publisher, or by ArtCraft. SubCraft is an independent project.

## What's new

- **Open a Black `.db`** and choose what to load: **Gun only**, **Arm only** or **Both**.
- **Textures load automatically.** The gun's own texture is read from the `.db` and assigned:
  RGB becomes **Base Color**, the alpha channel becomes **Metallic** (the grayscale shine map the
  2006 engine uses).
- **Import from .db** button (Shelf › Textures, Shelf › Materials, File menu) to bring a `.db`
  texture into the selected texture set as a new fill layer.
- **Black Xbox project template**: only Base Color and Metallic channels.
- **Export maps as `BaseColor` and `Alpha`** (Alpha = the Metallic channel).
- **Rebuild .db**: replaces the texture inside the `.db`, keeps a backup, and saves under the
  original name so you can test it in the game right away.
- **Image channel picker** for image sources: use the whole image, or only its **R**, **G** or
  **B** channel as grayscale. Any shelf texture, including normal maps, can be dropped on any
  channel slot (for example a normal map's G channel into Base Color).

## Quick start

1. **File › Open mesh** (or **File › New project**) and pick the `.db`.
2. Choose **Gun only**, **Arm only** or **Both**.
   Command line: `subcraft.exe --part gun path\to\bg1_xxx.db`
3. Paint. For a project made for this game, pick the **Black Xbox .db** template in New Project.
4. **Export**, switch to **Rebuild Black .db**, pick the original `.db`, check which texture each
   texture set replaces, tick *Invert alpha* if the shine map comes out reversed, and click
   **Rebuild .db**.
5. Copy the new `.db` into the game. The first rebuild leaves `name.db.bak` next to it.

## Rebuilding the .db

- The **mesh data is never changed**, only texture chunks, so the mesh pointers stay valid.
- Same size as the original texture: pixels are written in place and nothing else moves.
  Resized (tick *Allow a new texture size*, power of two, 4–4096): the chunk is rebuilt and the
  chunk table entries behind it are shifted.
- The result is re-read and checked. If a check fails, nothing is written.
- Saving over the original first copies it to `name.db.bak` (`.bak2`, `.bak3`… if one exists,
  so the first backup stays your untouched original).
- Saving to a different file name leaves the original alone and makes no backup.
- Textures are DXT1, DXT3 or DXT5 as in the original. **DXT1 has only 1-bit alpha**, so its Alpha
  map is not stored; the texture stays fully opaque. DXT3 and DXT5 keep the shine map.
- Mip levels are regenerated with a box filter.

## Layer blending

Each layer row in the Layers panel has a blend-mode list for the active channel (Add, Multiply,
and the rest). Select the Base Color channel first to blend color. Each channel keeps its own
blend mode and opacity.

## Requirements

- Windows 10 or later (64-bit) with a GPU that supports Vulkan or DX12.
- Black's original `.db` files from your own copy of the game.

## Known limits

- Gun and arm textures are matched to the model by name; when no name matches, the largest
  texture is used. Check the arms after loading **Both**.
- Skinning and animation data are not loaded or edited; the mesh is read for painting only.
- Replacing the **model** itself is not part of this tool.
- The viewport uses SubCraft's standard PBR shading, not the 2006 engine's look.
- Opening a `.db` from the recent list or by drag and drop uses the last chosen part.

## Build from source

```powershell
git clone https://github.com/Aalzard/subcraft.git
cd subcraft
cargo run --release
cargo test -p subcraft-blackdb
```

Needs Rust 1.95 or newer and the Visual Studio C++ Build Tools (see `docs/development.md`).

## Credits and license

Based on [SubCraft by Aalzard](https://github.com/Aalzard/subcraft), licensed MIT OR Apache-2.0
(see `LICENSE-MIT`, `LICENSE-APACHE`). The `.db` texture and mesh handling is a Rust port of the
community `black_texture_tool` and `black_fps_*` scripts.
