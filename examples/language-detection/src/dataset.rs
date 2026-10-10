use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::Value;
use std::fs;

#[derive(Debug, Clone, Deserialize)]
pub struct LanguageSample {
    #[serde(alias = "content", alias = "sentence")]
    pub text: String,

    #[serde(alias = "label", alias = "lang", alias = "language_code")]
    pub language: String,
}

pub fn load_dataset(path: &str) -> Result<Vec<LanguageSample>> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("Failed to read dataset: {path}"))?;

    let json: Value =
        serde_json::from_str(&contents).with_context(|| format!("Invalid JSON in {path}"))?;

    let records = if json.is_array() {
        json
    } else if let Some(data) = json.get("data") {
        data.clone()
    } else if let Some(samples) = json.get("samples") {
        samples.clone()
    } else {
        return Err(anyhow!(
            "Expected a JSON array or an object containing 'data' or 'samples'"
        ));
    };

    let samples: Vec<LanguageSample> = serde_json::from_value(records)
        .context("Dataset records must contain text and language fields")?;

    let samples: Vec<LanguageSample> = samples
        .into_iter()
        .filter(|sample| !sample.text.trim().is_empty() && !sample.language.trim().is_empty())
        .map(|mut sample| {
            sample.text = sample.text.trim().to_string();
            sample.language = sample.language.trim().to_string();
            sample
        })
        .collect();

    if samples.is_empty() {
        return Err(anyhow!("No valid training samples were found"));
    }

    Ok(samples)
}
