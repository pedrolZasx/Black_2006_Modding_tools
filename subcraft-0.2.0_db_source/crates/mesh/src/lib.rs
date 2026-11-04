//! Mesh processing: canonical mesh, importers, tangent space, BVH, topology, primitives.

mod alembic;
pub mod bvh;
pub mod ray_batch;
pub mod import;
pub mod mesh;
pub mod primitives;
pub mod regroup;
pub mod serialize;
pub mod spp;
pub mod tangent;
pub mod topology;
pub mod transfer;
pub mod uv_raster;
mod usd_import;

pub use bvh::{Bvh, ClosestPoint, FlatBvh, RayHit};
pub use import::{BlackPart, ImportError, ImportReport, SUPPORTED_EXTENSIONS, black_part, import_mesh, import_mesh_for_baking, set_black_part};
pub use mesh::{Aabb, BuildError, Corner, MaterialSlot, Mesh, MeshBuilder, MeshObject, Submesh, SurfacePoint};
