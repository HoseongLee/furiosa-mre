use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2];
axes![Q = 896, S = 128];

/*
thread 'rustc' (3557815) panicked at crates/npu-compiler/crates/npu-compiler-dma/src/compile_dma_sequencer.rs:201:22:
called `Result::unwrap()` on an `Err` value: split (axis_index: 0, inner_size: 4) is not valid on shape([Q_64=14, S_decode_8=16])
*/

#[device(chip = 1)]
pub fn forward(ctx: &mut Context, q: &HbmTensor<bf16, Chip, m![S, Q]>) {
    let q: DmTensor<bf16, Chip, Cluster, m![Q / 128 # 8, Q / 64 % 2, S / 8], m![S % 8, Q % 64]> =
        q.to_dm(&mut ctx.tdma);
}
