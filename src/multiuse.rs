use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2];
axes![D = 64, G = 7, H = 896, L = 4864, N = 2, P = 128, Q = 896, S = 128];

/*
thread 'rustc' (67257) panicked at crates/npu-compiler/crates/scheduler/src/common/operator_schedule_graph.rs:571:39:
HashMap::index: invalid key
*/

fn test(
    ctx: &mut Context,
    t: &DmTensor<f32, Chip, Cluster, m![N, S], m![D]>,
) {
    let _vrf: VrfTensor<f32, Chip, Cluster, m![N, S], m![D]> = ctx
        .sub
        .begin(t.view())
        .fetch::<m![1], m![D]>()
        .collect::<m![D / 8], m![D % 8]>()
        .to_vrf();
}

#[device(chip = 1)]
pub fn forward(
    ctx: &mut Context,
) {
    let t: DmTensor<f32, Chip, Cluster, m![N, S], m![D]> = unsafe { DmTensor::from_addr(0) };

    test(ctx, &t);
    test(ctx, &t);
}