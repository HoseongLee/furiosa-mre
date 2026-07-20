use furiosa_opt_std::prelude::*;

axes![S = 4, H = 8];

pub type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

#[device(chip = 1)]
pub fn forward(ctx: &mut Context, input: &HbmTensor<f32, Chip, m![S, H]>, out: &mut HbmTensor<f32, Chip, m![S, H]>) {
    let input: DmTensor<f32, Chip, Cluster, Slice, m![S, H]> = input.to_dm(&mut ctx.tdma);
    let mut result: DmTensor<f32, Chip, Cluster, Slice, m![S, H]> = unsafe { DmTensor::from_addr(1 << 12) };

    ctx.main
        .begin(input.view().tile::<m![S], 2, m![S = 2 # 4, H]>(0))
        .fetch::<m![S = 2], m![H]>()
        .collect::<m![S = 2], m![H]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, 1f32)
        .vector_final()
        .commit_trim::<m![H]>()
        .commit_view(result.view_mut().tile::<m![S], 2, m![S = 2 #{!} 4, H]>(0));

    ctx.main
        .begin(input.view().tile::<m![S], 2, m![S = 2 # 4, H]>(2))
        .fetch::<m![S = 2], m![H]>()
        .collect::<m![S = 2], m![H]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, 2f32)
        .vector_final()
        .commit_trim::<m![H]>()
        .commit_view(result.view_mut().tile::<m![S], 2, m![S = 2 #{!} 4, H]>(2));

    result.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}
