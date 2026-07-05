use furiosa_opt_std::prelude::*;

use crate::{H, Q, S_decode as S};
use crate::{Chip, Cluster};

/*
thread 'rustc' (45277) panicked at crates/npu-compiler/crates/npu-compiler-dma/src/compile_dma_sequencer.rs:200:73:
called `Option::unwrap()` on a `None` value
*/

#[device(chip = 1)]
pub fn forward(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![Q, H]>,
) {
    let q: DmTensor<bf16, Chip, Cluster, m![S / 8, Q / 128 # 8, Q / 64 % 2], m![Q % 64]> =
        q.to_dm(&mut ctx.tdma);
}