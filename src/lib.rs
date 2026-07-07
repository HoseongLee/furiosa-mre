#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

pub mod beam;
pub mod cluster_tile;
pub mod dm_to_dm;
pub mod multiuse;
pub mod slice_padding;
pub mod slice_padding_v2;
pub mod tile;
pub mod tile_hbm;
pub mod typecheck_collect_kernel;

use furiosa_opt_std::prelude::*;

axes![
    H = 896,    // Hidden size = 14 * 64
    P = 128,    // KV projection size
    Q = 896,    // Q projection size
    W = 151936, // Word Count
    N = 2,
    G = 7,
    D = 64,
    L = 4864,
    S_decode = 128,   // sequence length (decode)
    S_prefill = 1024, // query sequence length (prefill)
    T = 1024,         // Sequence length
    Dummy16 = 16
];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
