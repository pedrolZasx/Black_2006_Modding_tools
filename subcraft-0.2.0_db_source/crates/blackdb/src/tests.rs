use crate::mesh::{Part, strip_to_tris, unpack_normal};
use crate::texture::{self, DXT1, DXT5};
use crate::{chunk_table, put_u32};

/// db: chunk0 (dummy, 0x80), texture chunk (8x8 DXT1, 4 mips), trailing chunk (0x80).
fn synthetic_db(fmt: u32) -> Vec<u8> {
    let (w, h, mips) = (8usize, 8usize, 4usize);
    let bs = if fmt == DXT1 { 8 } else { 16 };
    let data_len = (4 + 1 + 1 + 1) * bs;
    let tex_len = (0x180 + data_len).div_ceil(0x80) * 0x80;
    let (c0, c1) = (0x80usize, 0x100usize);
    let c2 = c1 + tex_len;
    let end = c2 + 0x80;
    let mut d = vec![0u8; end];
    put_u32(&mut d, 0x18, 3);
    for (i, v) in [c0, c1, c2, end].iter().enumerate() {
        put_u32(&mut d, 0x1C + 4 * i, *v as u32);
    }
    let fd = (fmt << 8) | ((mips as u32) << 16) | (3 << 20) | (3 << 24);
    put_u32(&mut d, c1 + 0x10C, fd);
    put_u32(&mut d, c1 + 0x134, fmt);
    put_u32(&mut d, c1 + 0x138, w as u32);
    put_u32(&mut d, c1 + 0x13C, h as u32);
    d[c1 + 0x144..c1 + 0x144 + 6].copy_from_slice(b"gunTex");
    for b in &mut d[c2..end] {
        *b = 0xAB; // trailing chunk must survive untouched
    }
    d
}

fn gradient(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::new();
    for y in 0..h {
        for x in 0..w {
            v.extend_from_slice(&[(x * 255 / (w - 1)) as u8, (y * 255 / (h - 1)) as u8, 128, ((x + y) * 255 / (w + h - 2)) as u8]);
        }
    }
    v
}

#[test]
fn dxt5_roundtrip_is_close() {
    let img = gradient(16, 16);
    let dec = texture::decode_dxt(&texture::encode_dxt(&img, 16, 16, DXT5), 16, 16, DXT5).unwrap();
    let err: f64 = img.iter().zip(&dec).map(|(a, b)| (*a as f64 - *b as f64).abs()).sum::<f64>() / img.len() as f64;
    assert!(err < 6.0, "mean abs error {err}");
}

#[test]
fn finds_and_replaces_in_place() {
    for fmt in [DXT1, DXT5] {
        let db = synthetic_db(fmt);
        let tex = texture::find_textures(&db).unwrap();
        assert_eq!(tex.len(), 1);
        assert_eq!((tex[0].w, tex[0].h, tex[0].mips, tex[0].name.as_str()), (8, 8, 4, "gunTex"));
        let img = gradient(8, 8);
        let (new, info) = texture::build_replacement(&db, &tex[0], &img, 8, 8, false).unwrap();
        assert_eq!(new.len(), db.len());
        assert_eq!(info.shift, 0);
        assert_eq!(chunk_table(&new).unwrap(), chunk_table(&db).unwrap());
        let (ok, lines) = texture::verify_build(&db, &new, &tex[0], &info, &img);
        assert!(ok, "{lines:#?}");
    }
}

#[test]
fn new_size_shifts_the_chunk_table() {
    let db = synthetic_db(DXT1);
    let tex = texture::find_textures(&db).unwrap();
    let img = gradient(16, 16);
    assert!(texture::build_replacement(&db, &tex[0], &img, 16, 16, false).is_err());
    let (new, info) = texture::build_replacement(&db, &tex[0], &img, 16, 16, true).unwrap();
    assert!(info.shift > 0);
    let (ok, lines) = texture::verify_build(&db, &new, &tex[0], &info, &img);
    assert!(ok, "{lines:#?}");
    assert_eq!(texture::find_textures(&new).unwrap()[0].w, 16);
}

#[test]
fn strip_and_normal_decoding() {
    // 0 1 2 3 -> tris (0,1,2) and flipped (2,1,3); degenerate join skipped
    assert_eq!(strip_to_tris(&[0, 1, 2, 3, 3, 4], 5), vec![[0, 1, 2], [2, 1, 3]]);
    let n = unpack_normal(0x3FF | (0 << 11) | (0 << 22)); // +X
    assert!((n[0] - 1.0).abs() < 1e-4 && n[1].abs() < 1e-4);
    assert_eq!(Part::parse("gun"), Some(Part::Gun));
    assert_eq!(Part::parse("nope"), None);
}
