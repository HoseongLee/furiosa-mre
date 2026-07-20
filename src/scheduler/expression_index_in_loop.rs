use furiosa_opt_std::prelude::*;

axes![S = 512, H = 1024, L = 3072];

type Chip = m![1];
type Cluster = m![S / 256];

#[device(chip = 1)]
pub fn forward(ctx: &mut Context, input: &HbmTensor<bf16, Chip, m![S, H]>, weight: &HbmTensor<bf16, Chip, m![L, H]>) {
    type SliceUG = m![S / 64 % 4, L / 48];

    let input: DmTensor<bf16, Chip, Cluster, m![S % 256], m![H]> = input.to_dm(&mut ctx.tdma);

    let x: DmTensor<bf16, Chip, Cluster, SliceUG, m![S % 64, H]> = ctx
        .main
        .begin(input.view())
        .fetch::<m![1], m![H]>()
        .switch::<SliceUG, m![S % 64]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![S % 64, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    let mut result: DmTensor<bf16, Chip, Cluster, SliceUG, m![S % 64, L % 48]> = unsafe { DmTensor::from_addr(0) };

    let weight: DmTensor<bf16, Chip, Cluster, SliceUG, m![L % 48, H]> = weight.to_dm(&mut ctx.tdma);

    for i in 0..5 {
        let weight_tile = weight.view().tile::<m![L % 48], 8, m![L % 48 = 8 # 48, H]>(i << 3);
        let weight_trf: TrfTensor<bf16, Chip, Cluster, SliceUG, m![L % 48 = 8], m![H]> = ctx
            .sub
            .begin(weight_tile)
            .fetch::<m![L % 48 = 8, H / 16], m![H % 16]>()
            .collect::<m![L % 48 = 8, H / 16], m![H % 16]>()
            .to_trf();

        ctx.main
            .begin(x.view())
            .fetch::<m![S % 64, H / 16], m![H % 16]>()
            .collect::<m![S % 64, H / 16], m![H % 16]>()
            .contract_outer::<m![S % 64, H / 32], m![H % 32], _, _>(&weight_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![S % 64]>()
            .contract_lane::<m![S % 64], m![L % 48 = 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![L % 48 = 8 # 16]>()
            .commit_trim::<m![L % 48 = 8]>()
            .commit_view(
                result
                    .view_mut()
                    .tile::<m![L % 48], 8, m![S % 64, L % 48 = 8 #{!} 48]>(i << 3),
            );
    }

    let weight_tile = weight.view().tile::<m![L % 48], 8, m![L % 48 = 8 # 48, H]>(40);
    let weight_trf: TrfTensor<bf16, Chip, Cluster, SliceUG, m![L % 48 = 8], m![H]> = ctx
        .sub
        .begin(weight_tile)
        .fetch::<m![L % 48 = 8, H / 16], m![H % 16]>()
        .collect::<m![L % 48 = 8, H / 16], m![H % 16]>()
        .to_trf();

    ctx.main
        .begin(x.view())
        .fetch::<m![S % 64, H / 16], m![H % 16]>()
        .collect::<m![S % 64, H / 16], m![H % 16]>()
        .contract_outer::<m![S % 64, H / 32], m![H % 32], _, _>(&weight_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![S % 64]>()
        .contract_lane::<m![S % 64], m![L % 48 = 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![L % 48 = 8 # 16]>()
        .commit_trim::<m![L % 48 = 8]>()
        .commit_view(
            result
                .view_mut()
                .tile::<m![L % 48], 8, m![S % 64, L % 48 = 8 #{!} 48]>(40),
        );
}
