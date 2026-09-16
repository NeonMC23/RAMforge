fn main() {
    if let Err(error) = ramforge_cli::tui::run() {
        eprintln!("RAMforge TUI error: {error}");
        std::process::exit(1);
    }
}
