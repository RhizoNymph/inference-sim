fn main() {
    if let Err(err) = inference_sim::cli::run_from_env() {
        if matches!(err, inference_sim::cli::CliError::Help(_)) {
            println!("{err}");
            return;
        }

        eprintln!("{err}");
        std::process::exit(2);
    }
}
