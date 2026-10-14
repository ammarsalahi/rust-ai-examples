use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

const MODEL_URL: &str =
    "https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/model.safetensors";

const LABELS_URL: &str = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json";

pub fn download_file(url: &str, destination: &Path) -> Result<()> {
    if destination.exists() {
        println!("Already exists: {}", destination.display());
        return Ok(());
    }

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }

    println!("Downloading: {url}");

    let response = reqwest::blocking::Client::new()
        .get(url)
        .send()
        .with_context(|| format!("Failed to request {url}"))?
        .error_for_status()
        .with_context(|| format!("Download failed: {url}"))?;

    let bytes = response.bytes()?;

    if bytes.is_empty() {
        anyhow::bail!("Downloaded file is empty: {url}");
    }

    let temporary_path = destination.with_extension("download");

    fs::write(&temporary_path, &bytes)
        .with_context(|| format!("Failed to write {}", temporary_path.display()))?;

    fs::rename(&temporary_path, destination)
        .with_context(|| format!("Failed to save {}", destination.display()))?;

    println!(
        "Downloaded {} ({:.2} MB)",
        destination.display(),
        bytes.len() as f64 / 1_000_000.0
    );

    Ok(())
}

pub fn ensure_model_files() -> Result<()> {
    let model_path = Path::new("models/model.safetensors");
    let labels_path = Path::new("models/imagenet-labels.json");

    download_file(MODEL_URL, model_path)?;
    download_file(LABELS_URL, labels_path)?;

    Ok(())
}
