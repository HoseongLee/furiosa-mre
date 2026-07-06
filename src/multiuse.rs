use furiosa_opt_std::prelude::*;

use crate::{D, N, S_decode as S};
use crate::{Chip, Cluster};

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