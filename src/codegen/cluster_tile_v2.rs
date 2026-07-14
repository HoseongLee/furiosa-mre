use furiosa_opt_std::prelude::*;

type Chip = m![1];
axes![D = 64, N = 2, S = 128];

/*

thread 'rustc' (668578) panicked at crates/npu-opt/src/bin/furiosa-opt-driver/codegen/rustc_plugin.rs:424:25:
failed to translate vISA to IR: output T12 must be a view

*/

#[device(chip = 1)]
pub fn forward(ctx: &mut Context) {
    let mut clustered_result: DmTensor<bf16, Chip, m![N], m![S], m![D]> = unsafe { DmTensor::from_addr(0) };

    let result1: DmTensor<bf16, Chip, m![1 # 2], m![S], m![D]> = unsafe { DmTensor::from_addr(0) };

    result1.view().to_dm_view(
        &mut ctx.tdma,
        clustered_result.view_mut().cluster_tile::<m![N], 1, m![1 #{!} 2]>(0),
    );
}
