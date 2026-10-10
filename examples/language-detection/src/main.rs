mod dataset;
mod model;

use anyhow::{Context, Result};
use std::io::{self, Write};

use crate::dataset::load_dataset;
use crate::model::LanguageModel;

fn main() -> Result<()> {
    let dataset_path = "data/languages.json";

    println!("==================================");
    println!("     Rust Language Detection");
    println!("==================================");
    println!();

    // Load the training dataset.
    let samples = load_dataset(dataset_path)
        .with_context(|| format!("Could not load dataset from {dataset_path}"))?;

    println!("Loaded {} training samples.", samples.len());

    // Train the machine-learning model.
    println!("Training logistic regression...");

    let model = LanguageModel::train(&samples)?;

    println!("Training completed.");
    println!();

    println!("Number of languages: {}", model.language_count());

    println!("Vocabulary size: {}", model.vocabulary_size());

    println!();
    println!("Enter text to detect its language.");
    println!("Type 'exit' or 'quit' to stop.");
    println!();

    let stdin = io::stdin();

    loop {
        print!("text> ");
        io::stdout().flush()?;

        let mut input = String::new();

        if stdin.read_line(&mut input)? == 0 {
            break;
        }

        let input = input.trim();

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            println!("Goodbye!");
            break;
        }

        if input.is_empty() {
            continue;
        }

        match model.predict(input) {
            Some((language, score)) => {
                println!("Predicted language: {language}");

                println!("Model score: {:.2}%", score * 100.0);
            }

            None => {
                println!(
                    "Unable to predict the language. \
                     The text may be too short or contain \
                     no known character features."
                );
            }
        }

        println!();
    }

    Ok(())
}
