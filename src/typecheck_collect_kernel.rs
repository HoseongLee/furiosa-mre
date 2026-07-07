use furiosa_opt_std::prelude::*;

use crate::{H, L, S_decode as S};
use crate::{Chip, Cluster};

#[device(chip = 1)]
pub fn forward(ctx: &mut Context) {
    let mut o_proj: DmTensor<bf16, Chip, Cluster, m![S / 16, H / 28], m![S % 16, H % 28]> = unsafe { DmTensor::from_addr(0) };

    let o_proj: DmTensor<bf16, Chip, Cluster, m![S / 16, L / 152], m![S % 16, H]> = ctx
        .sub
        .begin(o_proj.view())
        .fetch::<m![L / 152, S % 16], m![H % 28]>()
        .switch::<m![S / 16, L / 152], m![S % 16, H / 28]>(SwitchConfig::InterTranspose { slice1: 32, slice0: 1, time0: 16 })
        .collect::<m![S % 16, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
}
