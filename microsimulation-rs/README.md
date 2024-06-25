# microsimulation-rs

## Summary

This package aims to provide a mirror of the functionality of the [microsimulation](https://github.com/nismod/microsimulation) package, with seedable simulation, a revised implementation and written in Rust for improved performance.

Since the assignment algorithm and output schema aim to be the same as `microsimulation`, the package can be used as a drop-in substitute.

With improved performance, the crate aims to facilitate the generation of ensembles of populations for uncertainty and scenario modelling, for example, as an ensemble of base populations for the [Synthetic Population Catalyst](https://github.com/alan-turing-institute/uatk-spc).

## Quickstart

- Install [Rust](https://www.rust-lang.org/tools/install)
- Build `microsimulation-rs` with `cargo build --release`
- Run the tests with `cargo test`
- Generate microsimulation outputs for a given region for the population ([run_ssm.py](../scripts/run_ssm.py)) and households ([run_ssm_h.py](../scripts/run_ssm_h.py))
- Run assignment with an example region (assumes both static population and household microsimulation have already been run):

```
./RUST_LOG=info ./target/release/assignment \
    --config config/ass_current_2020.json \
    --region E06000001 \
    --seed 0
```

## Features

- [ ] Sequence of microsynthesised populations ([static.py](../microsimulation/static.py))
- [ ] Sequence of microsynthesised households ([static_h.py](../microsimulation/static.py))
- [x] Assignment of people to households ([assignment.py](../microsimulation/assignment.py))

## Methods

The implementation aims to be as close as possible to the original Python code, with the following changes:

- **Queue algorithm**: Sampling of sets of unassigned people for a given set of conditions (e.g. MSOA, age) is implemented through a [`Queues`](src/queues.rs#L63) type. This type maintains multiple queues of mapping keys such as `(MSOA, Age)` to queues of person IDs (`PID`) that can be assigned, maintained in a `Vec<PID>`. To ensure that no person is assigned more than once, a separate assigned set (type `HashSet<PID>`) is maintained. For a given `PID` a check is performed to ensure that the `PID` is not contained in the assigned set.
- **New type pattern**: We make use of the [newtype pattern](https://doc.rust-lang.org/rust-by-example/generics/new_types.html) to enable function signatures to carry documentation as well compile-time checks for correct variable usage. For example, the `struct MSOA(String)` and `struct OA(String)` types allow similar codes to be distinguished in their usage.
- **Deterministic simulation**: random seeding is included throughout the simulation and can be set from the CLI.
- **Logging**: We use [env_logger](https://crates.io/crates/env_logger) to log messages distinguishing info, warning, debug and error cases.

## Benches

Benchmarks can be run for `microsimulation-rs` using `cargo bench`.

For comparison with the Python version, we can run the assignment script:

```
# microsimulation
hyperfine -M 1 'python -W ignore scripts/run_assignment.py -c config/ass_current_2020.json E06000001'

# microsimulation-rs
hyperfine -M 1 './target/release/assignment -c config/ass_current_2020.json -r E06000001 -s 0'
```

with output:
| Package | Time (seconds) |
|---------|--------------------------|
| microsimulation | 392.5 |
| microsimulation-rs | 0.9 |

## Validation

Outputs are validated against the Python code in [microsimulation](../microsimulation/).

For a direct comparison of measures used for validation in [microsimulation](../microsimulation/assignment.py#L633), see the table below comparing a single run from `microsimulation` with 100 stochastic runs (mean and standard deviation) from `microsimulation-rs`.

| Measure                                       | microsimulation | microsimulation-rs    |
| --------------------------------------------- | --------------- | --------------------- |
| Occupied households without HRP               | 0               | 0.0 (0.0)             |
| Occupied households not filled                | 7181 of 41930   | 7185.1 (2.9) of 41930 |
| Communal residences not filled                | 14              | 13.6 (1.2)            |
| Single-occupant households not filled         | 0               | 0.0 (0.0)             |
| Single-parent one-child households not filled | 0               | 0.0 (0.0)             |
| Single-parent two-child households not filled | 0               | 0.0 (0.0)             |
| Single-parent 3+ households not filled        | 825             | 825.0 (0.0)           |
| Couple households with no children not filled | 0               | 0.0 (0.0)             |
| Couple households with one child not filled   | 4               | 8.1 (2.9)             |
| Couple households with 2+ children not filled | 6240            | 6240.0 (0.0)          |
| Mixed (2,3) households not filled             | 0               | 0.0 (0.0)             |
| Mixed (4+) households not filled              | 112             | 112.0 (0.0)           |
| Adults not assigned                           | 0 of 75530      | 0.0 (0.0) of 75530    |
| Children not assigned                         | 4 of 17984      | 1.6 (1.5) of 17984    |
