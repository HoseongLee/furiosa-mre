# Minimal Reproducible Example

Every examples has the error code inside the source code

## Installation

1. Install [furiosa-opt-preview](https://github.com/furiosa-ai/furiosa-opt-preview)
2. Inside furiosa-opt-preview clone this repo
3. run
```
cargo furiosa-opt compiler build --device-function {FILENAME}::{FUNCTIONNAME} --dump-schedule transformer.json
```

## List of Examples

- [X] beam.rs (caused by allocating more than 512KB to one slice)
- [ ] tile.rs
- [ ] slice_padding.rs
- [X] dm_to_dm.rs (fixed in 0.4.0)
- [ ] slice_padding_v2.rs
- [ ] multiuse.rs
- [ ] typecheck_collect_kernel.rs (Compiles, but does not pass typecheck (can be checked via `typecheck_collect`))
- [ ] tile_hbm.rs