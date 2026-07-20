use furiosa_opt_std::prelude::*;
use mre::behavior::tile_index::{self, *};

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();

    let input_host: HostTensor<f32, m![S, H]> = HostTensor::from_buf((0..32).map(|e| e as f32).collect::<Vec<f32>>());
    let input_hbm: HbmTensor<f32, Chip, m![S, H]> = input_host.to_hbm(&mut ctx.pdma).await;

    let mut output_hbm: HbmTensor<f32, Chip, m![S, H]> = unsafe { HbmTensor::from_addr(0) };

    launch(tile_index::forward, (&mut ctx, &input_hbm, &mut output_hbm)).await;

    let output_host: HostTensor<f32, m![S, H]> = output_hbm.to_host(&mut ctx.pdma).await;
    let output: Vec<f32> = output_host.into_buf();

    for i in 0..32 {
        println!("{} {}", i, output[i]);
    }
}
