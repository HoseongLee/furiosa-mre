use furiosa_opt_std::prelude::*;

/*
error: furiosa-opt: codegen::empty_loop::forward: visa: internal compiler error: attempt to subtract with overflow
*/

/*
Conditions required to reproduce this error
1. Emtpy loop body
 */

#[device(chip = 1)]
pub fn forward(ctx: &mut Context) {
    for _ in 0..1 {}
}
