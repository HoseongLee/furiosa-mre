use furiosa_opt_std::prelude::*;

/*
error: furiosa-opt: codegen::large_loop::forward: lir: CTensorAddr::Var() count must be <= TENSOR_DYNAMIC_INDICIES_MAX, got 1187
*/

/*
Conditions required to reproduce this error
1. Large Loops with tile write(?) not certain
2. When more operations are added to the loop body the less the loop size has to be before the error is triggered
*/

axes![H = 896, W1 = 1187, W2 = 128];

type Chip = m![1];
type Cluster = m![1 # 2];

type Slice = m![W2 / 4 # 256];

#[device(chip = 1)]
pub fn forward(ctx: &mut Context, input: &HbmTensor<bf16, Chip, m![H]>, weight: &HbmTensor<bf16, Chip, m![W1, W2, H]>) {
    let input: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = input.to_dm(&mut ctx.tdma);
    let input_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![1], m![H]> = ctx
        .sub
        .begin(input.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let mut result_tile: DmTensor<bf16, Chip, Cluster, Slice, m![W1, W2 % 4]> = unsafe { DmTensor::from_addr(0) };

    for i in 0..W1::SIZE {
        let weight_dm: DmTensor<bf16, Chip, Cluster, Slice, m![W2 % 4, H]> = weight
            .view()
            .tile::<m![W1], 1, m![1 # 1187, W2, H]>(i)
            .to_dm(&mut ctx.tdma);

        ctx.main
            .begin(weight_dm.view())
            .fetch::<m![W2 % 4, H / 16], m![H % 16]>()
            .collect::<m![W2 % 4, H / 16], m![H % 16]>()
            .contract_outer::<m![W2 % 4, H / 32], m![H % 32], _, _>(&input_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![W2 % 4]>()
            .contract_lane::<m![W2 % 4], m![1 # 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![W2 % 4 # 16]>()
            .commit_trim::<m![W2 % 4]>()
            .commit_view(result_tile.view_mut().tile::<m![W1], 1, m![1 # 1187, W2 % 4]>(i));
    }
}
