use anyhow::{Result, anyhow};
use linfa::prelude::*;
use linfa_logistic::{MultiFittedLogisticRegression, MultiLogisticRegression};
use ndarray::{Array1, Array2};
use std::collections::{BTreeSet, HashMap};

use crate::dataset::LanguageSample;

const MIN_NGRAM: usize = 2;
const MAX_NGRAM: usize = 4;
const MAX_FEATURES: usize = 4_000;

type TrainedClassifier = MultiFittedLogisticRegression<f64, usize>;

pub struct LanguageModel {
    vocabulary: HashMap<String, usize>,
    idf: Vec<f64>,
    languages: Vec<String>,
    classifier: TrainedClassifier,
}

impl LanguageModel {
    pub fn train(samples: &[LanguageSample]) -> Result<Self> {
        // Remove invalid samples.
        let valid_samples: Vec<&LanguageSample> = samples
            .iter()
            .filter(|sample| !sample.language.trim().is_empty() && !sample.text.trim().is_empty())
            .collect();

        if valid_samples.len() < 2 {
            return Err(anyhow!("At least two valid training samples are required"));
        }

        // Build a deterministic list of language labels.
        let mut language_set = BTreeSet::new();

        for sample in &valid_samples {
            language_set.insert(sample.language.trim().to_string());
        }

        let languages: Vec<String> = language_set.into_iter().collect();

        if languages.len() < 2 {
            return Err(anyhow!("Training requires at least two distinct languages"));
        }

        // Encode language labels as integers.
        let language_ids: HashMap<String, usize> = languages
            .iter()
            .enumerate()
            .map(|(id, language)| (language.clone(), id))
            .collect();

        // Extract character n-grams from each document.
        let documents: Vec<Vec<String>> = valid_samples
            .iter()
            .map(|sample| extract_ngrams(&sample.text))
            .collect();

        // Calculate document frequency for every feature.
        let mut document_frequency: HashMap<String, usize> = HashMap::new();

        for document in &documents {
            let unique_features: BTreeSet<&str> = document.iter().map(String::as_str).collect();

            for feature in unique_features {
                *document_frequency.entry(feature.to_string()).or_insert(0) += 1;
            }
        }

        // Rank features by document frequency.
        let mut ranked_features: Vec<(String, usize)> = document_frequency
            .iter()
            .map(|(feature, frequency)| (feature.clone(), *frequency))
            .collect();

        ranked_features.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        ranked_features.truncate(MAX_FEATURES);

        // Create a stable vocabulary.
        let mut selected_features: Vec<String> = ranked_features
            .into_iter()
            .map(|(feature, _)| feature)
            .collect();

        selected_features.sort();

        if selected_features.is_empty() {
            return Err(anyhow!(
                "No character n-grams found. Use longer training texts."
            ));
        }

        let vocabulary: HashMap<String, usize> = selected_features
            .iter()
            .enumerate()
            .map(|(index, feature)| (feature.clone(), index))
            .collect();

        // Calculate smoothed IDF.
        //
        // IDF(t) = ln((N + 1) / (DF(t) + 1)) + 1
        let document_count = documents.len() as f64;

        let idf: Vec<f64> = selected_features
            .iter()
            .map(|feature| {
                let df = *document_frequency.get(feature).unwrap_or(&0) as f64;

                ((document_count + 1.0) / (df + 1.0)).ln() + 1.0
            })
            .collect();

        // Build the dense TF-IDF feature matrix.
        let mut feature_values = Vec::with_capacity(documents.len() * vocabulary.len());

        for document in &documents {
            let vector = vectorize(document, &vocabulary, &idf);

            feature_values.extend(vector);
        }

        let records = Array2::from_shape_vec((documents.len(), vocabulary.len()), feature_values)
            .map_err(|error| anyhow!("Failed to create feature matrix: {error}"))?;

        // Convert labels to class IDs.
        let targets = Array1::from(
            valid_samples
                .iter()
                .map(|sample| language_ids[sample.language.trim()])
                .collect::<Vec<usize>>(),
        );

        let dataset = Dataset::new(records, targets);

        // Train multinomial logistic regression.
        let classifier = MultiLogisticRegression::default()
            .max_iterations(200)
            .fit(&dataset)
            .map_err(|error| anyhow!("Logistic regression training failed: {error}"))?;

        Ok(Self {
            vocabulary,
            idf,
            languages,
            classifier,
        })
    }

    pub fn language_count(&self) -> usize {
        self.languages.len()
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn predict(&self, text: &str) -> Option<(String, f64)> {
        if text.trim().is_empty() {
            return None;
        }

        let features = extract_ngrams(text);

        if features.is_empty() {
            return None;
        }

        // Use the same vocabulary and IDF as training.
        let vector = vectorize(&features, &self.vocabulary, &self.idf);

        // Reject inputs that have no known features.
        if vector.iter().all(|value| *value == 0.0) {
            return None;
        }

        let records = Array2::from_shape_vec((1, self.vocabulary.len()), vector).ok()?;

        let probabilities = self.classifier.predict_probabilities(&records);

        // Select the class with the highest predicted probability.
        let mut best: Option<(usize, f64)> = None;

        for (column, &class_id) in self.classifier.classes().iter().enumerate() {
            if column >= probabilities.ncols() || class_id >= self.languages.len() {
                continue;
            }

            let probability = probabilities[[0, column]];

            if best.map_or(true, |(_, best_probability)| probability > best_probability) {
                best = Some((class_id, probability));
            }
        }

        let (class_id, probability) = best?;

        Some((
            self.languages[class_id].clone(),
            probability.clamp(0.0, 1.0),
        ))
    }
}

// Convert character n-grams into a normalized TF-IDF vector.
fn vectorize(features: &[String], vocabulary: &HashMap<String, usize>, idf: &[f64]) -> Vec<f64> {
    let mut counts: HashMap<usize, usize> = HashMap::new();

    let mut known_feature_count = 0usize;

    for feature in features {
        if let Some(&index) = vocabulary.get(feature) {
            *counts.entry(index).or_insert(0) += 1;
            known_feature_count += 1;
        }
    }

    let mut vector = vec![0.0; vocabulary.len()];

    if known_feature_count == 0 {
        return vector;
    }

    // TF-IDF = term frequency * inverse document frequency.
    for (index, count) in counts {
        let tf = count as f64 / known_feature_count as f64;

        vector[index] = tf * idf[index];
    }

    // L2 normalization.
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();

    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }

    vector
}

// Extract character n-grams of lengths 2, 3, and 4.
fn extract_ngrams(text: &str) -> Vec<String> {
    let normalized: Vec<char> = text
        .to_lowercase()
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();

    let mut features = Vec::new();

    for n in MIN_NGRAM..=MAX_NGRAM {
        if normalized.len() < n {
            continue;
        }

        for window in normalized.windows(n) {
            features.push(window.iter().collect::<String>());
        }
    }

    features
}
