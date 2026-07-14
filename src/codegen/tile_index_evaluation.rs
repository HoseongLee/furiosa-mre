use furiosa_opt_std::prelude::*;

type Chip = m![1];
axes![H = 896, L = 4864];

/*

error: expected an axis view offset, i.e. usize or a scalar variable
  --> src/tile.rs:19:42
   |
19 | ...p_weight.view().tile::<m![L], 256, m![L = 256 # 4864, H]>(i << 8);
   |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

error: could not compile `mre` (lib) due to 1 previous error

*/

#[device(chip = 1)]
pub fn forward(ctx: &mut Context, up_weight: &HbmTensor<bf16, Chip, m![L, H]>) {
    for i in 0..1 {
        let up_weight = up_weight.view().tile::<m![L], 256, m![L = 256 # 4864, H]>(i << 8);
        // let up_weight: DmTensor<bf16, Chip, Cluster, m![S / 8, L = 256 / 16], m![L = 256 % 16, H]> = up_weight.to_dm(&mut ctx.tdma);
    }
}
