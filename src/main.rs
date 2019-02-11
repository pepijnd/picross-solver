#[macro_use]
extern crate error_chain;

use std::env;
use std::fs::File;
use std::io::prelude::*;

use time::Duration;

mod picross;

fn main() -> picross::error::Result<()> {
    let args: Vec<String> = env::args().collect();
    let filename = args.get(1);

    let mut file = File::open(filename.unwrap_or(&String::from("input.txt")))?;
    let mut input = String::new();
    file.read_to_string(&mut input)?;

    let puzzle = picross::Puzzle::new(input)?;
    puzzle.solve();

    Ok(())
}
