use furiosa_opt_std::prelude::*;

use mre::typecheck_collect_kernel;

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();

    launch(typecheck_collect_kernel::forward, &mut ctx).await;
}
