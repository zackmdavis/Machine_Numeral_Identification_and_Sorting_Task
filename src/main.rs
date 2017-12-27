extern crate mnist;
extern crate rusty_machine;

use std::env;
use std::error::Error;
use std::path::Path;
use std::ops::Range;

use mnist::{Mnist, MnistBuilder};
use rusty_machine::learning::svm::SVM;
use rusty_machine::learning::toolkit::kernel::HyperTan;
use rusty_machine::prelude::*;


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

    fn image_matrix(&self, set: Set, range: Range<usize>) -> Matrix<u8> {
        let raw_range = 784*range.start..784*range.end; // 784 = 28²
        let raw_images = match set {
            Set::Training => &self.0.trn_img[raw_range],
            Set::Validation => &self.0.val_img[raw_range],
            Set::Test => &self.0.tst_img[raw_range]
        };
        Matrix::new(range.len(), 784, raw_images)
    }

    fn labels(&self, set: Set, range: Range<usize>) -> Vector<u8> {
        match set {
            Set::Training => &self.0.trn_lbl[range],
            Set::Validation => &self.0.val_lbl[range],
            Set::Test => &self.0.tst_lbl[range]
        }.into()
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
        .training_set_length(10000)
        .validation_set_length(10000)
        .test_set_length(1000)
        .base_path(&format!("{}/mnist/data", parent_directory.display()))
        .finalize();
    Ok(MnistData(mnist))
}


fn main() {
    let data = get_data().expect("should get data");

    let training_data_u8 = data.image_matrix(Set::Training, 0..10000);
    let training_targets_u8 = data.labels(Set::Training, 0..10000);

    let validation_data_u8 = data.image_matrix(Set::Validation, 0..1000);
    let validation_targets_u8 = data.labels(Set::Validation, 0..1000);

    let training_data = Matrix::new(training_data_u8.rows(), training_data_u8.cols(), training_data_u8.into_vec().iter().map(|e| *e as f64).collect::<Vec<_>>());
    let training_targets = training_targets_u8.into_vec().iter().map(|e| *e as f64).collect::<Vec<_>>();

    let validation_data = Matrix::new(validation_data_u8.rows(), validation_data_u8.cols(), validation_data_u8.into_vec().iter().map(|e| *e as f64).collect::<Vec<_>>());
    let validation_targets = validation_targets_u8.into_vec().iter().map(|e| *e as f64).collect::<Vec<_>>();

    let mut model = SVM::new(HyperTan::new(100., 0.), 0.3);
    model.train(&training_data, &training_targets.into())
        .expect("couldn't train?!");

    println!("Trained up!");

    let predictions = model.predict(&validation_data)
        .expect("couldn't validate?!");

    let mut hits = 0;
    for (digit, predicted) in validation_targets.iter().zip(predictions.iter()) {
        if digit == predicted {
            hits += 1;
        }
    }
    println!("Accuracy: {}/{} = {:.1}%", hits, 1000., f64::from(hits)/1000. * 100.);

}
