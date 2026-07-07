use furiosa_opt_std::prelude::*;

use crate::{H, L};
use crate::Chip;

axes![X = 19];

/*

failed to translate vISA to IR: verification of operator failed:
IndexAccess
axis: L_1, window_size: 1
name: 

input tensors: 2
  input T0: []|[L_1=4864:896, H_1=896:1], 8716288 B, bf16, dram
  source: unknown
  input T1: [Broadcast=1]|[Broadcast=2]|[Dummy=4]|[], 32 B, raw_i32, spm
  source: unknown
  total bytes: 8716320
output tensors: 1
  output T2: []|[L_1=256:896, H_1=896:1], 458752 B, bf16, dram
  source: unknown
  total bytes: 458752

Caused by:
    Output shape mismatch for IndexAccess

*/

#[device(chip = 1)]
pub fn forward(
    ctx: &mut Context,
    up_weight: &HbmTensor<bf16, Chip, m![L, H]>,
) {
    let up_weight = up_weight.view().tile::<m![L], 256, m![L = 256 # 4864, H]>(0);
}