extern crate mnist;
extern crate rusty_machine;

use std::env;
use std::error::Error;
use std::path::Path;
use std::ops::Range;

use mnist::{Mnist, MnistBuilder};
use rusty_machine::learning::naive_bayes::{self, NaiveBayes};
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
    #[cfg(XXX_TODO_FIXME_label_format)]
    fn digit(&self, set: Set, index: usize) -> Digit {
        let raw_range = 784*index..784*(index+1); // 784 = 28²
        let raw_image = match set {
            Set::Training => &self.0.trn_img[raw_range],
            Set::Validation => &self.0.val_img[raw_range],
            Set::Test => &self.0.tst_img[raw_range]
        };
        let image = Matrix::new(28, 28, raw_image);
        // XXX TODO FIXME: this assumes "digit" label format, but it looks like
        // we actually want one-hot vectors to feed into rusty-machine; convert
        // here?
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

    fn label_matrix(&self, set: Set, range: Range<usize>) -> Matrix<u8> {
        let raw_range = 10*range.start..10*range.end; // 784 = 28²
        let raw_labels = match set {
            Set::Training => &self.0.trn_lbl[raw_range],
            Set::Validation => &self.0.val_lbl[raw_range],
            Set::Test => &self.0.tst_lbl[raw_range]
        };
        Matrix::new(range.len(), 10, raw_labels)
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
        .label_format_one_hot()
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
    let training_targets_u8 = data.label_matrix(Set::Training, 0..10000);

    let validation_data_u8 = data.image_matrix(Set::Validation, 0..10000);
    let validation_targets_u8 = data.label_matrix(Set::Validation, 0..10000);

    let training_data = Matrix::new(training_data_u8.rows(), training_data_u8.cols(), training_data_u8.into_vec().iter().map(|e| *e as f64).collect::<Vec<_>>());
    let training_targets = Matrix::new(training_targets_u8.rows(), training_targets_u8.cols(), training_targets_u8.into_vec().iter().map(|e| *e as f64).collect::<Vec<_>>());

    let validation_data = Matrix::new(validation_data_u8.rows(), validation_data_u8.cols(), validation_data_u8.into_vec().iter().map(|e| *e as f64).collect::<Vec<_>>());
    let validation_targets = Matrix::new(validation_targets_u8.rows(), validation_targets_u8.cols(), validation_targets_u8.into_vec().iter().map(|e| *e as f64).collect::<Vec<_>>());

    let mut model = NaiveBayes::<naive_bayes::Gaussian>::new();
    model.train(&training_data, &training_targets)
        .expect("couldn't train?!");

    println!("Trained up!");

    let predictions = model.predict(&validation_data)
        .expect("couldn't validate?!");

    let mut hits = 0;
    for (digit, predicted) in validation_targets.iter_rows().zip(predictions.iter_rows()) {
        if digit == predicted {
            hits += 1;
        }
    }
    println!("Accuracy: {}/{} = {:.1}%", hits, 10000., f64::from(hits)/10000. * 100.);

}
