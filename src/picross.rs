#![allow(dead_code)]

//use std::collections::HashMap;
use hashbrown::HashMap;

#[allow(deprecated)]
pub mod error {
    error_chain! {
        foreign_links {
            Io(::std::io::Error);
            ParseInt(::std::num::ParseIntError);
        }

        errors {
            InvalidInput
            // ParseInputError(s: String) {
            //     description("Error in parsing input file"),
            //     display("Error in parsing input file: {}", s)
            // }
        }
    }
}

use error::*;

mod parse_input {
    use super::Clue;
    use super::{ErrorKind, Result};

    pub fn parse_input_string(input: String) -> Result<(Vec<Clue>, Vec<Clue>)> {
        let input: Vec<String> = input.split_whitespace().map(|e| String::from(e)).collect();
        let n_cols = input
            .get(0)
            .ok_or::<ErrorKind>(ErrorKind::InvalidInput.into())?
            .parse::<usize>()?;
        let n_rows = input
            .get(1)
            .ok_or::<ErrorKind>(ErrorKind::InvalidInput.into())?
            .parse::<usize>()?;
        let cols = input
            .get(2)
            .ok_or::<ErrorKind>(ErrorKind::InvalidInput.into())?;
        let rows = input
            .get(3)
            .ok_or::<ErrorKind>(ErrorKind::InvalidInput.into())?;

        let cols = parse_clues_string(cols)?;
        let rows = parse_clues_string(rows)?;

        if n_cols != cols.iter().len() || n_rows != rows.iter().len() {
            return Err(ErrorKind::InvalidInput.into());
        }

        Ok((cols, rows))
    }

    pub fn parse_clues_string(clues: &String) -> Result<Vec<Clue>> {
        let mut in_clue = false;
        let mut clue_string = String::new();
        let mut clue_list = Vec::new();

        for c in clues.chars() {
            if c == '"' {
                if in_clue {
                    clue_list.push(parse_clue_string(&clue_string)?);
                    clue_string = String::new();
                    in_clue = false;
                } else {
                    in_clue = true;
                }
            } else if c == ',' {
                if in_clue {
                    clue_string.push(c);
                }
            } else {
                if in_clue {
                    clue_string.push(c);
                } else {
                    return Err(ErrorKind::InvalidInput.into());
                }
            }
        }
        if in_clue {
            return Err(ErrorKind::InvalidInput.into());
        }
        Ok(clue_list)
    }

    pub fn parse_clue_string(clue: &String) -> Result<Clue> {
        let mut current = String::new();
        let mut sets = Vec::new();
        for c in clue.chars() {
            if c == ',' {
                sets.push(current.parse::<u32>()?);
                current = String::new();
            } else {
                current.push(c);
            }
        }
        sets.push(current.parse::<u32>()?);
        Ok(Clue::new(sets))
    }
}

pub mod util {
    //use std::collections::HashMap;
    use hashbrown::HashMap;

    use super::{ClueMask, Mask, SolverCache};

    pub type SpacingCache = HashMap<(u32, u32), Vec<Vec<u32>>>;

    pub fn spacings(count: u32, max: u32, cache: &mut SolverCache) -> &Vec<Vec<u32>> {
        if cache.spacing_cache.contains_key(&(count, max)) {
            return cache.spacing_cache.get(&(count, max)).unwrap();
        }

        let mut parts = partitions(count, max);
        parts.retain(|e| {
            for i in 1..e.len() - 1 {
                if *e.get(i).unwrap() == 0 {
                    return false;
                }
            }
            true
        });
        cache.spacing_cache.insert((count, max), parts);
        cache.spacing_cache.get(&(count, max)).unwrap()
    }

    pub fn partitions(count: u32, max: u32) -> Vec<Vec<u32>> {
        let mut len = 0;
        let mut output = Vec::new();
        while len < count {
            if len == 0 {
                for i in 0..=max {
                    output.push(vec![i])
                }
            } else if len == count-1 {
                let mut new_output = Vec::new();
                for mut part in output {
                    let sum = part.iter().sum::<u32>();
                    part.push(max-sum);
                    new_output.push(part);
                }
                output = new_output;
            } else {
                let mut new_output = Vec::new();

                for part in output {
                    let sum = part.iter().sum::<u32>();
                    for i in 0..=max-sum {
                        let mut new = part.clone();
                        new.push(i);
                        new_output.push(new);
                    }
                }
                output = new_output;
            }
            len+=1;
        }
        output
    }

    pub fn combine_cluemasks(cluemasks: &Vec<ClueMask>) -> ClueMask {
        let mut output = Vec::new();
        for i in 0..cluemasks.get(0).unwrap().mask.len() {
            let i_mask: Mask = cluemasks
                .iter()
                .map(|e| e.mask.get(i).unwrap())
                .fold(None, |acc, x| {
                    if acc.is_none() || acc.as_ref().unwrap() == x {
                        Some(*x)
                    } else {
                        Some(Mask::Unknown)
                    }
                })
                .unwrap();
            output.push(i_mask)
        }
        ClueMask::new(output)
    }
}

pub struct SolverCache {
    spacing_cache: util::SpacingCache,
    clue_mask_cache: HashMap<u32, Vec<ClueMask>>,
}

impl SolverCache {
    fn new() -> SolverCache {
        SolverCache {
            spacing_cache: HashMap::new(),
            clue_mask_cache: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct Puzzle {
    input: Input,
}

#[derive(Debug)]
pub struct Output {
    output_data: Vec<Mask>,
    width: usize,
    height: usize,
}

#[derive(Debug)]
pub struct Clue {
    sets: Vec<u32>,
}

#[derive(Debug)]
pub struct Input {
    columns: Vec<Clue>,
    rows: Vec<Clue>,
}

#[derive(Debug)]
pub struct Solver<'a> {
    input: &'a Input,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mask {
    Unknown,
    Set,
    Unset,
}

impl From<Mask> for String {
    fn from(mask: Mask) -> String {
        match mask {
            Mask::Set => String::from("#"),
            Mask::Unset => String::from(" "),
            Mask::Unknown => String::from("?"),
        }
    }
}

#[derive(Debug)]
pub struct ClueMask {
    mask: Vec<Mask>,
}

impl From<Vec<Mask>> for ClueMask {
    fn from(mask: Vec<Mask>) -> Self {
        ClueMask::new(mask)
    }
}

pub struct MaskGrid {
    width: usize,
    height: usize,
    grid: Vec<Mask>,
}

impl MaskGrid {
    fn new(width: usize, height: usize) -> MaskGrid {
        let mut grid = Vec::with_capacity(width * height);

        for _ in 0..width * height {
            grid.push(Mask::Unknown);
        }

        MaskGrid {
            width,
            height,
            grid,
        }
    }

    fn count_unsolved(&self) -> u32 {
        self.grid
            .iter()
            .fold(0, |acc, i| if *i == Mask::Unknown { acc + 1 } else { acc })
    }

    fn set_column(&mut self, index: usize, column: Vec<Mask>) {
        for (row, value) in column.iter().enumerate() {
            self.set_cell(row, index, *value);
        }
    }

    fn get_column(&self, index: usize) -> Vec<Mask> {
        let mut output = Vec::with_capacity(self.height);
        for i in 0..self.height {
            output.push(self.get_cell(i, index));
        }
        output
    }

    fn set_row(&mut self, index: usize, row: Vec<Mask>) {
        for (column, value) in row.iter().enumerate() {
            self.set_cell(index, column, *value);
        }
    }

    fn get_row(&self, index: usize) -> Vec<Mask> {
        let mut output = Vec::with_capacity(self.width);
        for i in 0..self.width {
            output.push(self.get_cell(index, i));
        }
        output
    }

    fn set_cell(&mut self, row: usize, col: usize, value: Mask) {
        let index = self.cell_idx(row, col);
        self.grid[index as usize] = value;
    }

    fn get_cell(&self, row: usize, col: usize) -> Mask {
        let index = self.cell_idx(row, col);
        self.grid[index]
    }

    fn cell_idx(&self, row: usize, col: usize) -> usize {
        row * self.width + col
    }
}

impl ClueMask {
    fn new(mask: Vec<Mask>) -> ClueMask {
        ClueMask { mask }
    }

    fn as_vec(&self) -> Vec<Mask> {
        self.mask.iter().map(|e| *e).collect()
    }

    fn iter(&self) -> std::slice::Iter<Mask> {
        self.mask.iter()
    }

    fn filter_match(&self, filter: &ClueMask) -> bool {
        for (lhs, rhs) in self.iter().zip(filter.iter()) {
            if *lhs == Mask::Set && *rhs == Mask::Unset || *lhs == Mask::Unset && *rhs == Mask::Set
            {
                return false;
            }
        }
        true
    }
}

impl Puzzle {
    pub fn new(input: String) -> Result<Puzzle> {
        let (cols, rows) = parse_input::parse_input_string(input)?;
        let input = Input::new(cols, rows);
        Ok(Puzzle { input })
    }

    pub fn solve(&self) -> Option<Output> {
        let solver = Solver::new(&self.input);
        solver.solve()
    }
}

impl Clue {
    pub fn new(sets: Vec<u32>) -> Clue {
        Clue { sets }
    }

    fn min_len(&self) -> usize {
        let mut min_len = self.sets.iter().sum::<u32>() as usize;
        min_len += self.sets.len() - 1;
        min_len
    }

    fn spaces_count(&self) -> u32 {
        self.sets.len() as u32 + 1
    }

    fn get_total_spaces(&self, len: usize) -> u32 {
        let set_count: u32 = self.sets.iter().sum();
        len as u32 - set_count
    }

    fn get_mask(
        &self,
        len: usize,
        filter: &ClueMask,
        mask_id: u32,
        cache: &mut SolverCache,
    ) -> Option<ClueMask> {
        let mut clue_mask_new = Vec::new();
        let clue_mask_set = {
            if cache.clue_mask_cache.contains_key(&mask_id) {
                cache.clue_mask_cache.get_mut(&mask_id).unwrap()
            } else {
                let spacings =
                    util::spacings(self.spaces_count(), self.get_total_spaces(len), cache);
                for spacing in spacings {
                    let mut clue_mask_option = Vec::new();
                    for (i, space) in spacing.iter().enumerate() {
                        if *space > 0 {
                            clue_mask_option.append(&mut vec![Mask::Unset; *space as usize]);
                        }
                        if i < spacing.len() - 1 {
                            clue_mask_option
                                .append(&mut vec![Mask::Set; *self.sets.get(i).unwrap() as usize]);
                        }
                    }
                    let clue_mask = ClueMask::from(clue_mask_option);
                    clue_mask_new.push(clue_mask);
                }
                cache.clue_mask_cache.insert(mask_id, clue_mask_new);
                cache.clue_mask_cache.get_mut(&mask_id).unwrap()
            }
        };

        clue_mask_set.retain(|e| e.filter_match(filter));
        Some(util::combine_cluemasks(clue_mask_set))
    }
}

impl Input {
    pub fn new(columns: Vec<Clue>, rows: Vec<Clue>) -> Input {
        Input { columns, rows }
    }
}

enum SolveIter {
    Colums,
    Rows,
}

impl<'a> Solver<'a> {
    pub fn new(input: &Input) -> Solver {
        Solver { input }
    }

    pub fn solve(&self) -> Option<Output> {
        let mut solver_cache = SolverCache::new();

        let width = self.input.columns.len();
        let height = self.input.rows.len();

        let mut mask_grid = MaskGrid::new(width, height);
        let mut unsolved = mask_grid.count_unsolved();

        for j in vec![SolveIter::Colums, SolveIter::Rows].iter().cycle() {
            match j {
                SolveIter::Colums => {
                    for (i, column) in self.input.columns.iter().enumerate() {
                        let col_mask = column.get_mask(
                            height,
                            &ClueMask::from(mask_grid.get_column(i)),
                            i as u32,
                            &mut solver_cache,
                        );
                        mask_grid.set_column(i, col_mask.unwrap().as_vec());
                    }
                }
                SolveIter::Rows => {
                    for (i, row) in self.input.rows.iter().enumerate() {
                        let row_mask = row.get_mask(
                            width,
                            &ClueMask::from(mask_grid.get_row(i)),
                            (width + i) as u32,
                            &mut solver_cache,
                        );
                        mask_grid.set_row(i, row_mask.unwrap().as_vec());
                    }
                }
            }
            let new_unsolved = mask_grid.count_unsolved();
            if new_unsolved == unsolved || new_unsolved == 0 {
                break;
            }
            unsolved = new_unsolved;
        }
        Some(Output::from(mask_grid))
    }
}

impl Output {
    fn new(width: usize, height: usize) -> Output {
        Output {
            output_data: Vec::with_capacity(width * height),
            width,
            height,
        }
    }

    pub fn get_size(&self) -> (usize, usize) {
        return (self.width, self.height);
    }

    pub fn to_rgba(&self, res: u32) -> Box<[u8]> {
        let mut rgba = Vec::new();
        for h in 0..self.height {
            for _ in 0..res {
                for w in 0..self.width {
                    let cell = self.width * h + w;
                    let color: Vec<u8> = match self.output_data[cell] {
                        Mask::Set => vec![0, 0, 0, 255],
                        Mask::Unset => vec![255, 255, 255, 255],
                        Mask::Unknown => vec![127, 127, 127, 255],
                    };
                    for _ in 0..res {
                        rgba.append(&mut color.clone());
                    }
                }
            }
        }
        rgba.into_boxed_slice()
    }
}

impl From<MaskGrid> for Output {
    fn from(maskgrid: MaskGrid) -> Output {
        let mut output = Output::new(maskgrid.width, maskgrid.height);
        output.output_data = maskgrid.grid;

        output
    }
}
