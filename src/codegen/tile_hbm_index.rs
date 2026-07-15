use furiosa_opt_std::prelude::*;

/*
error: furiosa-opt: codegen::tile_hbm_index::forward: visa: internal compiler error: alias index_access stores only IndexAccess and reshape ops, got CoreSymExpr { inputs: [], output: T2, expr: ConstInt(0) }
*/

/*
Conditions required to reproduce this error
1. Write to Hbm tiled view via to_hbm_view method
2. Index using some constant or constant variable (Works inside loop)
 */

axes![W1 = 1187, W2 = 128];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![W2 / 4 # 256];

#[device(chip = 1)]
pub fn forward(ctx: &mut Context, out: &mut HbmTensor<bf16, Chip, m![W1, W2]>) {
    let mut result_tile: DmTensor<bf16, Chip, Cluster, Slice, m![W2 % 4]> = unsafe { DmTensor::from_addr(0) };

    result_tile
        .view()
        .to_hbm_view(&mut ctx.tdma, out.view_mut().tile::<m![W1], 1, m![1 #{!} 1187, W2]>(0));
}
