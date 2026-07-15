# Minimal Reproducible Example

Every examples has the error code inside the source code

## Installation

1. Install [furiosa-opt-preview](https://github.com/furiosa-ai/furiosa-opt-preview)
2. Inside furiosa-opt-preview clone this repo
3. Select which test to run by enabling them in src/codegen/mod.rs file
4. run
```
cargo furiosa-opt compile
```

## List of Examples 

### Compile Errors
tested on snapshot-6a7d3377

- [X] beam.rs (caused by allocating more than 512KB to one slice)
- [X] dm_to_dm.rs (fixed in 0.4.0)
- [ ] tile_index_evaluation.rs
- [ ] padding_in_slice.rs
- [ ] padding_in_slice_v2.rs
- [ ] multiuse.rs
- [ ] tile_hbm.rs
- [ ] cluster_tile.rs
- [ ] cluster_tile_v2.rs
- [ ] padding_in_lane.rs
- [ ] empty_loop.rs
- [ ] large_loop.rs
- [ ] tile_hbm_index.rs

### Hang
- [ ] switching_padding_into_slice.rs