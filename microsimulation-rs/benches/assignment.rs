use std::{path::PathBuf, str::FromStr};

use lazy_static::lazy_static;
use microsimulation::{
    assignment::Assignment,
    config::{Config, Projection, Resolution, Year},
};

fn main() {
    // Run registered benchmarks.
    divan::main();
}

lazy_static! {
    static ref TEST_CONFIG: Config = Config {
        person_resolution: Resolution::MSOA11,
        household_resolution: Resolution::OA11,
        projection: Projection::PPP,
        strict: false,
        year: Year(2020),
        data_dir: PathBuf::from_str("../data/").unwrap(),
        persistent_data_dir: Some(PathBuf::from_str("../persistent_data/").unwrap()),
        profile: false,
    };
}

// Run benchmark for assignment to "E09000001" (small area, n=8_718)
#[divan::bench(args = ["E09000001"])]
fn assignment_small_size(area: &str) {
    let config = &Config {
        data_dir: PathBuf::from_str("tests/data/").unwrap(),
        ..TEST_CONFIG.clone()
    };
    let mut assignment_0_0 = Assignment::new(area, 0, config).unwrap();
    assignment_0_0.run().unwrap();
}

// Run benchmark for assignment to "E06000001" (medium area, n=93_515):
#[divan::bench(args = ["E06000001"])]
fn assignment_medium_size(area: &str) {
    let mut assignment_0_0 = Assignment::new(area, 0, &TEST_CONFIG).unwrap();
    assignment_0_0.run().unwrap();
}

// Run benchmark for assignment to "S12000046" (large area, n=631_427)
#[divan::bench(args = ["S12000046"], sample_count = 1)]
fn assignment_large_size(area: &str) {
    let mut assignment_0_0 = Assignment::new(area, 0, &TEST_CONFIG).unwrap();
    assignment_0_0.run().unwrap();
}
