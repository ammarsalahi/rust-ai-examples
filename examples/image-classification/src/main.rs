mod downloader;
mod model;
mod preprocessing;

use anyhow::{Context, Result};
use std::path::Path;

fn main() -> Result<()> {
    println!("=================================");
    println!(" Rust Image Classification");
    println!(" Candle + ResNet-18");
    println!("=================================\n");

    let image_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/image.jpg".to_string());

    let image_path = Path::new(&image_path);

    if !image_path.exists() {
        anyhow::bail!(
            "Image not found: {}\n\
             Usage: cargo run --release -- path/to/image.jpg",
            image_path.display()
        );
    }

    println!("Checking model files...");
    downloader::ensure_model_files()?;

    let model_path = Path::new("models/model.safetensors");
    let labels_path = Path::new("models/imagenet-labels.json");

    let labels: Vec<String> = serde_json::from_slice(
        &std::fs::read(labels_path).context("Could not read ImageNet labels")?,
    )
    .context("Invalid ImageNet labels JSON")?;

    println!("\nLoading model...");
    let classifier = model::ImageClassifier::new(model_path)?;

    println!("Preparing image...");
    let input = preprocessing::load_image(image_path, classifier.device())?;

    println!("Running inference...\n");
    let predictions = classifier.predict(&input, 5)?;

    println!("Image: {}", image_path.display());
    println!("Top predictions:");
    println!("---------------------------------");

    for (rank, (class_id, probability)) in predictions.iter().enumerate() {
        let label = labels
            .get(*class_id)
            .map(String::as_str)
            .unwrap_or("Unknown class");

        println!(
            "{}. {} (class {}, {:.2}%)",
            rank + 1,
            label,
            class_id,
            probability * 100.0
        );
    }

    println!("---------------------------------");

    Ok(())
}

