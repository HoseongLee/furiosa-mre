use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2];
axes![G = 7, H = 896, L = 4864, N = 2, P = 128, Q = 896, S = 128];

/*

thread 'rustc' (3394568) panicked at crates/npu-opt/src/bin/furiosa-opt-driver/codegen/rustc_plugin.rs:424:25:
failed to translate vISA to IR: BeamSearch failed

*/

#[device(chip = 1)]
pub fn forward(
    ctx: &mut Context,
    up_weight: &HbmTensor<bf16, Chip, m![L, H]>,
) {
    // let up_weight: DmTensor<bf16, Chip, Cluster, m![L / 304, 1 # 16], m![L % 304, H]> = up_weight.to_dm(&mut ctx.tdma);
    let up_weight: DmTensor<bf16, Chip, Cluster, m![L / 304, S / 8], m![L % 304, H]> = up_weight.to_dm(&mut ctx.tdma);
}