use furiosa_opt_std::prelude::*;

/*
large_loop::forward: lir: attempt to read incomplete tensor buffer(T11, D:0x00000000, kind=Stack)
*/

/*
Conditions required to reproduce this error
1. Tiled HBM load to DM with padding in Slice
2. Filing the padded slice with switch
3. Size of the loop does not matter just that is it more than 1

It does not appear if
- Tiled HBM load to DM with no padding (and such not using switch)
- HBM is loaded to DM fully and then tiled
 */

axes![H = 896, S = 128, W1 = 2, W2 = 128];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![W2 / 2, S / 32];

#[device(chip = 1)]
pub fn forward(ctx: &mut Context, weight: &HbmTensor<bf16, Chip, m![W1, W2, H]>) {
    let mut result_tile: DmTensor<bf16, Chip, Cluster, Slice, m![W2 % 2, H]> = unsafe { DmTensor::from_addr(0) };

    for i in 0..W1::SIZE {
        let weight_dm: DmTensor<bf16, Chip, Cluster, m![W2 / 2, 1 # 4], m![W2 % 2, H]> = weight
            .view()
            .tile::<m![W1], 1, m![1 # 2, W2, H]>(i)
            .to_dm(&mut ctx.tdma);

        ctx.main
            .begin(weight_dm.view())
            .fetch::<m![W2 % 2, H / 16], m![H % 16]>()
            .switch::<Slice, m![W2 % 2, H / 16]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
            .collect::<m![W2 % 2, H / 16], m![H % 16]>()
            .commit_trim::<m![H % 16]>()
            .commit_view(result_tile.view_mut());
    }
}
