use furiosa_opt_std::prelude::*;

/*
error: furiosa-opt: codegen::padding_in_lane::forward: visa: StoTrf: kernel-declared fn_output_shape does not match trf_shape after TrfMacRows flattening.
         declared: [ Chip: [] | Cluster: [[[]+1]=2] | Partitioning: [[[]+255]=256] | InSlice: [[[S_1=9]+7]=16, T_1=128] ] (16)
         derived:  [ Chip: [] | Cluster: [[[]+1]=2] | Partitioning: [[[]+255]=256] | InSlice: [S_2=8, *{_1}S_1=2, T_1=128] ] (16), packing: [*S{_1} -> [[S_1=9]+7]=16]
  --> src/codegen/padding_in_lane.rs:24:10
   |
24 |         .to_trf();
   |          ^^^^^^^^
*/

/*
Conditions required to reproduce this error
1. S is not 8
 */

axes![S = 9, T = 128];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

#[device(chip = 1)]
pub fn forward(ctx: &mut Context) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![S # 16, T]> = unsafe { DmTensor::from_addr(0) };
    let x: TrfTensor<bf16, Chip, Cluster, Slice, m![S # 16 / 2], m![S # 16 % 2, T]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![S # 16, T / 16], m![T % 16]>()
        .collect::<m![S # 16, T / 16], m![T % 16]>()
        .to_trf();
}
