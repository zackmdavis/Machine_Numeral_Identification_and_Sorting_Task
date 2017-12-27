extern crate mnist;
extern crate rusty_machine;

use std::env;
use std::path::Path;
use std::error::Error;

use mnist::{Mnist, MnistBuilder};
use rusty_machine::linalg::Matrix;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Set {
    Training,
    Validation,
    Test
}

#[derive(Debug, Clone)]
struct Digit {
    image: Matrix<u8>,
    label: u8
}

struct MnistData(Mnist);

impl MnistData {
    fn digit(&self, set: Set, index: usize) -> Digit {
        let raw_range = 784*index..784*(index+1); // 784 = 28²
        let raw_image = match set {
            Set::Training => &self.0.trn_img[raw_range],
            Set::Validation => &self.0.val_img[raw_range],
            Set::Test => &self.0.tst_img[raw_range]
        };
        let image = Matrix::new(28, 28, raw_image);
        let label = match set {
            Set::Training => self.0.trn_lbl[index],
            Set::Validation => self.0.val_lbl[index],
            Set::Test => self.0.tst_lbl[index]
        };
        Digit { image, label }
    }
}

fn get_data() -> Result<MnistData, Box<Error>> {
    // XXX TODO FIXME: this assumes that we have a copy of `mnist` checked out
    // in the same directory we're checked out in, which is kind of a stupid
    // assumption
    let pwd = env::var("PWD")?;
    let working_directory = Path::new(&pwd);
    let parent_directory = working_directory.parent()
        .expect("should compute parent directory");
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(1000)
        .validation_set_length(1000)
        .test_set_length(1000)
        .base_path(&format!("{}/mnist/data", parent_directory.display()))
        .finalize();
    Ok(MnistData(mnist))
}


fn main() {
    let data = get_data().expect("should get data");
    println!("{}", data.digit(Set::Training, 0).image);
}
