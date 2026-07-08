#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

pub mod beam;
pub mod cluster_tile;
pub mod cluster_tile_v2;
pub mod dm_to_dm;
pub mod multiuse;
pub mod slice_padding;
pub mod slice_padding_v2;
pub mod switch_hang;
pub mod tile;
pub mod tile_hbm;
pub mod typecheck_collect_kernel;
