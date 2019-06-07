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
            PuzzleError
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
        let input: Vec<String> = input.split_whitespace().map(String::from).collect();
        let n_cols = input
            .get(0)
            .ok_or::<ErrorKind>(ErrorKind::InvalidInput)?
            .parse::<usize>()?;
        let n_rows = input
            .get(1)
            .ok_or::<ErrorKind>(ErrorKind::InvalidInput)?
            .parse::<usize>()?;
        let cols = input
            .get(2)
            .ok_or::<ErrorKind>(ErrorKind::InvalidInput)?;
        let rows = input
            .get(3)
            .ok_or::<ErrorKind>(ErrorKind::InvalidInput)?;

        let cols = parse_clues_string(cols)?;
        let rows = parse_clues_string(rows)?;

        if n_cols != cols.iter().len() || n_rows != rows.iter().len() {
            return Err(ErrorKind::InvalidInput.into());
        }

        Ok((cols, rows))
    }

    pub fn parse_clues_string(clues: &str) -> Result<Vec<Clue>> {
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
            } else if in_clue {
                    clue_string.push(c);
                } else {
                    return Err(ErrorKind::InvalidInput.into());
                }
        }
        if in_clue {
            return Err(ErrorKind::InvalidInput.into());
        }
        Ok(clue_list)
    }

    pub fn parse_clue_string(clue: &str) -> Result<Clue> {
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
    use super::HashMap;
    use super::{ClueMask, Mask, SolverCache};
    use super::error::*;

    pub type SpacingCache = HashMap<(u32, u32), Vec<Vec<u32>>>;

    pub fn spacings(count: u32, max: u32, cache: &mut SolverCache) -> Result<&Vec<Vec<u32>>> {
        if !cache.spacing_cache.contains_key(&(count, max)) {
            let parts = partitions(count, max)?;
            cache.spacing_cache.insert((count, max), parts);
        }
        Ok(cache.spacing_cache.get(&(count, max)).unwrap())
    }

    struct Partitions {
        count: u32,
        max: u32
    }

    struct PartitionIter {
        count: u32,
        max: u32,
        len: u32,
        subLen: u32,
        outIter: Option<Box<Iterator<Item=Vec<u32>>>>,
        index: usize
    }

    impl IntoIterator for Partitions {
        type Item = Vec<u32>;
        type IntoIter = PartitionIter;

        fn into_iter(self) -> Self::IntoIter {
            PartitionIter { count: self.count, max: self.max, len: 0, subLen: 0, outIter: None, index: 0 }
        }
    }

    impl Iterator for PartitionIter {
        type Item = Vec<u32>;
        fn next(&mut self) -> Option<Self::Item> {
            if self.len == 0 {
                self.outIter = Some(Box::new(
                    PartitionFirstIter {
                        count: self.count,
                        max: self.max,
                        index: 0
                    }
                ))
                self.len += 1;
            }
            if self.len < self.count {
                } else if len == count - 1 {
                    None
                } else {
                    let next = self.outIter.unwrap().next();
                    if next.is_none() {
                        self.outIter = Some(PartitionMidIter {
                            count: self.count,
                            max: self.max,
                            len: self.len,
                            outIter: None,
                            index: 0
                        })
                        next = self.outIter.unwrap().next();
                    } else {

                    }
                }
            }
            next
        }
    }

    struct PartitionFirstIter {
        count: u32,
        max: u32,
        index: usize
    }

    impl Iterator for PartitionFirstIter {
        type Item = Vec<u32>;
        fn next(&mut self) -> Option<Self::Item> {
            if self.index <= self.max as usize {
                let mut new = Vec::with_capacity(2);
                new.push(self.index as u32);
                self.index += 1;
                Some(new)
            } else {
                None
            }
        }
    }

    struct PartitionMidIter {
        count: u32,
        max: u32,
        len: u32,
        outIter: Option<Box<Iterator<Item=Vec<u32>>>>,
        index: usize
    }

    impl Iterator for PartitionMidIter {
        type Item = Vec<u32>;
        fn next(&mut self) -> Option<Self::Item> {

            None
        }
    }

    pub fn partitions(count: u32, max: u32) -> Result<Vec<Vec<u32>>> {
        let mut len = 0;
        let mut output = Vec::with_capacity(max as usize);
        while len < count {
            if len == 0 {
                for i in 0..=max {
                    let mut new = Vec::with_capacity(2);
                    new.push(i);
                    output.push(new);
                }
            } else if len == count - 1 {
                let mut new_output = Vec::with_capacity(output.len());
                for mut part in output {
                    let sum = part.iter().sum::<u32>();
                    part.push(max - sum);
                    new_output.push(part);
                }
                output = new_output;
            } else {
                let mut new_output = Vec::with_capacity(output.len() * (max as usize) / 2);

                for part in output {
                    let sum = part.iter().sum::<u32>();
                    for i in 1..=max - sum {
                        let mut new = Vec::with_capacity(part.len()+2);
                        new.extend_from_slice(&part);
                        new.push(i);
                        new_output.push(new);
                    }
                }
                output = new_output;
            }
            len += 1;
        }
        Ok(output)
    }

    pub fn combine_cluemasks(cluemasks: &[ClueMask]) -> Result<ClueMask> {
        if cluemasks.is_empty() {
            return Err(Error::from(ErrorKind::PuzzleError));
        }
        let len = cluemasks.first().unwrap().mask.len();
        let mut output = Vec::with_capacity(len);
        for i in 0..len {
            let mut mask_part = Mask::Unknown;
            for j in cluemasks.iter() {
                if mask_part == Mask::Unknown {
                    mask_part = j.mask[i];
                } else if j.mask[i] != mask_part {
                    mask_part = Mask::Unknown;
                    break;
                }
            }
            output.push(mask_part);
        }
        Ok(ClueMask::new(output))
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
        self.mask.to_vec()
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

    pub fn solve(&self) -> Result<Output> {
        let solver = Solver::new(&self.input);
        solver.solve()
    }
}

impl Clue {
    pub fn new(sets: Vec<u32>) -> Clue {
        Clue { sets }
    }

    fn spaces_count(&self) -> u32 {
        self.sets.len() as u32 + 1
    }

    fn get_total_spaces(&self, len: usize) -> u32 {
        let set_count: u32 = self.sets.iter().sum();
        len as u32 - set_count
    }

    fn calc_mask_options(&self, spacing: &[u32], len: usize) -> Vec<Mask> {
        let mut clue_mask_option = Vec::with_capacity(len);
        for (i, space) in spacing.iter().enumerate() {
            if *space > 0 {
                for _ in 0..*space {
                    clue_mask_option.push(Mask::Unset);
                }
            }
            if i < spacing.len() - 1 {
                for _ in 0..self.sets[i] {
                    clue_mask_option.push(Mask::Set);
                }
            }
        }
        clue_mask_option
    }

    fn get_mask(
        &self,
        len: usize,
        filter: &ClueMask,
        mask_id: u32,
        cache: &mut SolverCache,
    ) -> Result<ClueMask> {
        let mut clue_mask_set = cache.clue_mask_cache.get_mut(&mask_id);
        let in_cache = clue_mask_set.is_some();
        if !in_cache {
            let spacings = util::spacings(self.spaces_count(), self.get_total_spaces(len), cache)?;
            let mut clue_mask_new = Vec::with_capacity(spacings.len());
            for spacing in spacings {
                let clue_mask_option = self.calc_mask_options(spacing, len);
                let clue_mask = ClueMask::from(clue_mask_option);
                if clue_mask.filter_match(filter) {
                    clue_mask_new.push(clue_mask);
                }
            }
            cache.clue_mask_cache.insert(mask_id, clue_mask_new);
            clue_mask_set = cache.clue_mask_cache.get_mut(&mask_id)
        }
        let clue_mask_set = clue_mask_set.unwrap();
        if in_cache && clue_mask_set.len() > 1 {
            clue_mask_set.retain(|e| e.filter_match(filter));
        }
        util::combine_cluemasks(clue_mask_set)
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

    pub fn solve(&self) -> Result<Output> {
        let mut solver_cache = SolverCache::new();

        let width = self.input.columns.len();
        let height = self.input.rows.len();

        let mut mask_grid = MaskGrid::new(width, height);
        let mut unsolved = mask_grid.count_unsolved();

        for (_n, j) in vec![SolveIter::Colums, SolveIter::Rows]
            .iter()
            .cycle()
            .enumerate()
        {
            match j {
                SolveIter::Colums => {
                    for (i, column) in self.input.columns.iter().enumerate() {
                        let col_mask = column.get_mask(
                            height,
                            &ClueMask::from(mask_grid.get_column(i)),
                            i as u32,
                            &mut solver_cache,
                        )?;
                        mask_grid.set_column(i, col_mask.as_vec());
                    }
                }
                SolveIter::Rows => {
                    for (i, row) in self.input.rows.iter().enumerate() {
                        let row_mask = row.get_mask(
                            width,
                            &ClueMask::from(mask_grid.get_row(i)),
                            (width + i) as u32,
                            &mut solver_cache,
                        )?;
                        mask_grid.set_row(i, row_mask.as_vec());
                    }
                }
            }
            let new_unsolved = mask_grid.count_unsolved();
            if new_unsolved == 0 {
                break;
            } else if new_unsolved == unsolved {
                break;
            }
            unsolved = new_unsolved;
        }
        Ok(Output::from(mask_grid))
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
        (self.width, self.height)
    }

    pub fn to_rgba(&self, res: u32) -> Box<[u8]> {
        let black = [0, 0, 0, 255];
        let white = [255, 255, 255, 255];
        let grey  = [127, 127, 127, 255];
        let border  = [200, 200, 200, 255];
        
        let mut rgba = Vec::new();
        for h in 0..self.height {
            for i in 0..res {
                for w in 0..self.width {
                    let cell = self.width * h + w;
                    let mut color = match self.output_data[cell] {
                        Mask::Set => &black,
                        Mask::Unset => &white,
                        Mask::Unknown => &grey,
                    };
                    if i == 0 || i == res-1 {
                        color = &border;
                    }
                    for j in 0..res {
                        if j == 0 || j == res-1 {
                            rgba.extend_from_slice(&border);
                        } else {
                            rgba.extend_from_slice(color);
                        }

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
