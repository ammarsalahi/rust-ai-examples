use anyhow::{Context, Result};
use std::{
    fs::{self, File},
    io::{self, BufReader, Read, Write},
    path::Path,
};

const MODEL_DIR: &str = "models/recognizer";

const MODEL_FILES: &[&str] = &["config.json", "model.safetensors", "tokenizer.json"];

const MODEL_BASE_URL: &str = "https://huggingface.co/microsoft/trocr-base-printed/resolve/main/";

fn ensure_model_files() -> Result<()> {
    fs::create_dir_all(MODEL_DIR).context("Failed to create model directory")?;

    for filename in MODEL_FILES {
        let destination = Path::new(MODEL_DIR).join(filename);

        if destination.is_file() {
            let metadata = fs::metadata(&destination)?;

            if metadata.len() > 0 {
                println!("Model file exists: {}", destination.display());
                continue;
            }
        }

        download_model_file(filename, &destination)?;
    }

    Ok(())
}

fn download_model_file(filename: &str, destination: &Path) -> Result<()> {
    let url = format!("{MODEL_BASE_URL}{filename}");

    println!("Downloading {filename}...");
    println!("URL: {url}");

    let response = ureq::get(&url).call().with_context(|| {
        format!(
            "Failed to download {filename}. \
                 Check your internet connection and the Hugging Face URL."
        )
    })?;

    let temporary_path = destination.with_extension("download");

    let result = (|| -> Result<()> {
        let mut output =
            File::create(&temporary_path).context("Failed to create temporary model file")?;

        let mut reader = response.into_reader();

        io::copy(&mut reader, &mut output).context("Failed to write downloaded model file")?;

        output.flush()?;

        let size = output.metadata()?.len();

        if size == 0 {
            anyhow::bail!("Downloaded file is empty: {filename}");
        }

        drop(output);

        fs::rename(&temporary_path, destination)
            .context("Failed to move downloaded file into place")?;

        println!(
            "Downloaded {} ({:.2} MB)",
            destination.display(),
            size as f64 / (1024.0 * 1024.0)
        );

        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }

    result
}
