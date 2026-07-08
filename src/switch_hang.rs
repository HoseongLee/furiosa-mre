use furiosa_opt_std::prelude::*;


type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![S / 8, Q / 64 % 2, Q / 128 # 8];

axes![D = 64, G = 7, H = 896, L = 4864, N = 2, P = 128, Q = 896, S = 128];

#[device(chip = 1)]
pub fn forward(ctx: &mut Context) {
    let result: DmTensor<bf16, Chip, Cluster, m![S / 8, Q / 64 % 2, Q / 8 % 8], m![Q / 128, S % 8, Q % 8]> =
        unsafe { DmTensor::from_addr(0) };
    let mut result: DmTensor<bf16, Chip, Cluster, Slice, m![S % 8, Q % 64]> = ctx
        .sub
        .begin(result.view())
        .fetch::<m![Q / 128 # 8, S % 8], m![Q % 8 # 16]>()
        .switch::<Slice, m![S % 8, Q / 8 % 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 8,
        })
        .collect::<m![S % 8, Q / 8 % 8], m![Q % 8 # 16]>()
        .commit_trim::<m![Q % 8]>()
        .commit();
}
