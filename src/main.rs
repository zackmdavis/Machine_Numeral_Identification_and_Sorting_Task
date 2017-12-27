extern crate mnist;
extern crate rusty_machine;

use std::env;
use std::path::Path;
use std::error::Error;

use mnist::{Mnist, MnistBuilder};

fn data() -> Result<Mnist, Box<Error>> {
    // XXX TODO FIXME: this assumes that we have a copy of `mnist` checked out
    // in the same directory we're checked out in, which is kind of a stupid
    // assumption
    let pwd = env::var("PWD")?;
    let working_directory = Path::new(&pwd);
    let parent_directory = working_directory.parent()
        .expect("should compute parent directory");
    Ok(MnistBuilder::new()
       .label_format_digit()
       // XXX: use non-puny set sizes
       .training_set_length(1000)
       .validation_set_length(1000)
       .test_set_length(1000)
       .base_path(&format!("{}/mnist/data", parent_directory.display()))
       .finalize())
}


fn main() {
    println!("{:?}", data().expect("should get data"));
}
