use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2];
axes![D = 64, G = 7, H = 896, L = 4864, N = 2, P = 128, Q = 896, S = 128];

/*
error: Unknown primitive: DmTensor::to_dm
  --> src/dm_to_dm.rs:12:66
   |
12 | ...ter, m![S # 256], m![Q]> = q.to_dm(&mut ctx.tdma, 0);
   |                                 ^^^^^^^^^^^^^^^^^^^^^^^

error: could not compile `mre` (lib) due to 1 previous error
*/

#[device(chip = 1)]
pub fn forward(ctx: &mut Context, q: &HbmTensor<bf16, Chip, m![S, Q]>) {
    let q: DmTensor<bf16, Chip, Cluster, m![S # 256], m![Q]> = q.to_dm(&mut ctx.tdma);
    let q: DmTensor<bf16, Chip, Cluster, m![S # 256], m![Q]> = q.to_dm(&mut ctx.tdma);
}
