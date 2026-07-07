use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::{D, N, S_decode as S};

/*

thread 'rustc' (661460) panicked at crates/npu-opt/src/bin/furiosa-opt-driver/codegen/rustc_plugin.rs:424:25:
failed to translate vISA to IR: TensorInfo mismatch for TensorVar(11) (TensorIndex T0): existing TensorInfo { shape: [ Chip: [] | Cluster: [N_1=2] | Partitioning: [S_decode_1=128] | InSlice: [D_1=64] ] (1), element_type: Bfloat16, memory_class: DM, address: None } vs new TensorInfo { shape: [ Chip: [] | Cluster: [[[]+1]=2] | Partitioning: [S_decode_1=128] | InSlice: [D_1=64] ] (1), element_type: Bfloat16, memory_class: DM, address: None }

*/

#[device(chip = 1)]
pub fn forward(ctx: &mut Context) {
    let clustered_result: DmTensor<bf16, Chip, m![N], m![S], m![D]> = unsafe { DmTensor::from_addr(0) };

    let mut result1: DmTensor<bf16, Chip, m![1 # 2], m![S], m![D]> = unsafe { DmTensor::from_addr(0) };
    let mut result2: DmTensor<bf16, Chip, m![1 # 2], m![S], m![N, D]> = unsafe { DmTensor::from_addr(0) };

    clustered_result
        .view()
        .cluster_tile::<m![N], 1, m![1 # 2]>(0)
        .to_dm_view(&mut ctx.tdma, result1.view_mut());

    // clustered_result
    //     .view()
    //     .cluster_tile::<m![N], 1, m![1 # 2]>(0)
    //     .to_dm_view(&mut ctx.tdma, result2.view_mut().tile::<m![N], 1, m![1 #{!} 2, D]>(0));
}
