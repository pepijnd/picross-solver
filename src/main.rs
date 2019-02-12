#[macro_use]
extern crate error_chain;

use std::env;
use std::fs::File;
use std::io::prelude::*;

use std::path::Path;
use std::io::BufWriter;

use png::HasParameters;

mod picross;

fn main() -> picross::error::Result<()> {
    let input_file = String::from("input.txt");
    let output_file = String::from("output.png");

    let args: Vec<String> = env::args().collect();
    let filename = args.get(1).unwrap_or(&input_file);
    let out_file = args.get(2).unwrap_or(&output_file);

    let mut file = File::open(filename)?;
    let mut input = String::new();
    file.read_to_string(&mut input)?;

    let puzzle = picross::Puzzle::new(input)?;
    let output = puzzle.solve().unwrap();

    let image_data = output.to_rgba(32);

    let path = Path::new(out_file);
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);

    let size = output.get_size();
    let mut encoder = png::Encoder::new(w, (size.0*32) as u32, (size.1*32) as u32);
    encoder.set(png::ColorType::RGBA).set(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();

    writer.write_image_data(&image_data).unwrap();

    Ok(())
}
