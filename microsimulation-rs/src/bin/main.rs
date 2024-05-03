use clap::{arg, Command};
use log::info;
use microsimulation::{assignment::Assignment, config::Config};

fn cli() -> Command {
    Command::new("SPENSER")
        .about(format!(
            "SPENSER: assignment of people and households v{}\n\n",
            env!("CARGO_PKG_VERSION")
        ))
        .arg_required_else_help(true)
        .arg(arg!(-c --config <FILE_PATH>).required(true))
        .arg(arg!(-r --region <REGION>).required(true))
        .arg(
            arg!(-s --rng_seed <RNG_SEED>)
                .required(false)
                .default_value("0"),
        )
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let matches = cli().get_matches();
    let region = matches.get_one::<String>("region").unwrap();
    let file_path: &String = matches.get_one::<String>("config").unwrap();
    let rng_seed: u64 = matches.get_one::<String>("rng_seed").unwrap().parse()?;

    let config: Config = serde_json::from_str(&std::fs::read_to_string(file_path)?)?;

    info!("Config: {}", serde_json::to_string(&config).unwrap());

    let mut assignment = Assignment::new(region, rng_seed, &config)?;
    assignment.run()?;
    assignment.check()?;
    assignment.write(region, &config)?;
    Ok(())
}