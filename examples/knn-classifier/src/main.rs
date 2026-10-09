use anyhow::{Context, Result};
use linfa::DatasetBase;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::{BTreeSet, HashMap};

const DATASET_PATH: &str = "data/iris.csv";
const FEATURE_COUNT: usize = 4;
const K: usize = 5;
const TEST_RATIO: f64 = 0.2;

struct IrisDataset {
    features: Array2<f64>,
    labels: Vec<String>,
}

struct StandardScaler {
    means: [f64; FEATURE_COUNT],
    stds: [f64; FEATURE_COUNT],
}

impl StandardScaler {
    fn fit(features: &Array2<f64>, indices: &[usize]) -> Result<Self> {
        anyhow::ensure!(!indices.is_empty(), "Training dataset is empty");

        let mut means = [0.0; FEATURE_COUNT];
        let mut stds = [0.0; FEATURE_COUNT];

        for &index in indices {
            for column in 0..FEATURE_COUNT {
                means[column] += features[[index, column]];
            }
        }

        for mean in &mut means {
            *mean /= indices.len() as f64;
        }

        for &index in indices {
            for column in 0..FEATURE_COUNT {
                let difference = features[[index, column]] - means[column];
                stds[column] += difference * difference;
            }
        }

        for std in &mut stds {
            *std = (*std / indices.len() as f64).sqrt();

            if *std == 0.0 {
                *std = 1.0;
            }
        }

        Ok(Self { means, stds })
    }

    fn transform_one(&self, input: &[f64]) -> [f64; FEATURE_COUNT] {
        let mut output = [0.0; FEATURE_COUNT];

        for column in 0..FEATURE_COUNT {
            output[column] = (input[column] - self.means[column]) / self.stds[column];
        }

        output
    }

    fn transform_matrix(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let mut values = Vec::with_capacity(features.nrows() * FEATURE_COUNT);

        for row in 0..features.nrows() {
            let input = features.row(row).to_vec();
            values.extend_from_slice(&self.transform_one(&input));
        }

        Ok(Array2::from_shape_vec(
            (features.nrows(), FEATURE_COUNT),
            values,
        )?)
    }
}

struct KnnClassifier {
    features: Array2<f64>,
    labels: Vec<String>,
    k: usize,
}

impl KnnClassifier {
    fn new(features: Array2<f64>, labels: Vec<String>, k: usize) -> Result<Self> {
        anyhow::ensure!(k > 0, "k must be greater than zero");
        anyhow::ensure!(
            k <= features.nrows(),
            "k cannot exceed the number of training samples"
        );
        anyhow::ensure!(
            features.nrows() == labels.len(),
            "Feature and label counts do not match"
        );

        Ok(Self {
            features,
            labels,
            k,
        })
    }

    fn distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn predict_one(&self, input: &[f64]) -> Result<String> {
        let mut distances: Vec<(f64, usize)> = (0..self.features.nrows())
            .map(|index| {
                let row = self.features.row(index).to_vec();
                (Self::distance(input, &row), index)
            })
            .collect();

        distances.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut votes: HashMap<String, (usize, f64)> = HashMap::new();

        for (distance, index) in distances.iter().take(self.k) {
            let entry = votes.entry(self.labels[*index].clone()).or_insert((0, 0.0));

            entry.0 += 1;
            entry.1 += distance;
        }

        votes
            .into_iter()
            .max_by(|a, b| {
                a.1.0
                    .cmp(&b.1.0)
                    .then_with(|| b.1.1.total_cmp(&a.1.1))
                    .then_with(|| b.0.cmp(&a.0))
            })
            .map(|(label, _)| label)
            .context("Could not determine a prediction")
    }
}

fn load_dataset(path: &str) -> Result<IrisDataset> {
    let mut reader =
        csv::Reader::from_path(path).with_context(|| format!("Could not open dataset: {path}"))?;

    let headers = reader.headers()?.clone();

    let feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"];

    let mut feature_indices = [0; FEATURE_COUNT];

    for (i, name) in feature_names.iter().enumerate() {
        feature_indices[i] = headers
            .iter()
            .position(|header| header == *name)
            .with_context(|| format!("Missing CSV column: {name}"))?;
    }

    let label_index = headers
        .iter()
        .position(|header| header == "species")
        .context("Missing CSV column: species")?;

    let mut values = Vec::new();
    let mut labels = Vec::new();

    for (row_index, record) in reader.records().enumerate() {
        let record =
            record.with_context(|| format!("Invalid CSV record at line {}", row_index + 2))?;

        for &column in &feature_indices {
            let value: f64 = record
                .get(column)
                .context("Missing feature value")?
                .parse()
                .with_context(|| format!("Invalid number at CSV line {}", row_index + 2))?;

            anyhow::ensure!(
                value.is_finite(),
                "Non-finite value at CSV line {}",
                row_index + 2
            );

            values.push(value);
        }

        let label = record
            .get(label_index)
            .context("Missing species label")?
            .trim()
            .to_string();

        anyhow::ensure!(!label.is_empty(), "Empty species label");

        labels.push(label);
    }

    anyhow::ensure!(!labels.is_empty(), "Dataset contains no samples");

    let features = Array2::from_shape_vec((labels.len(), FEATURE_COUNT), values)?;

    Ok(IrisDataset { features, labels })
}

fn evaluate() -> Result<()> {
    println!("========================================");
    println!("       Rust KNN Classification");
    println!("========================================");

    println!("\nLoading dataset...");

    let dataset = load_dataset(DATASET_PATH)?;

    println!("Dataset: {}", DATASET_PATH);
    println!("Samples: {}", dataset.features.nrows());
    println!("Features: {}", dataset.features.ncols());

    let classes: BTreeSet<&str> = dataset.labels.iter().map(String::as_str).collect();

    println!("Classes: {}", classes.len());

    let mut indices: Vec<usize> = (0..dataset.features.nrows()).collect();

    let mut rng = StdRng::seed_from_u64(42);
    indices.shuffle(&mut rng);

    let test_count = ((indices.len() as f64) * TEST_RATIO).round().max(1.0) as usize;

    anyhow::ensure!(
        test_count < indices.len(),
        "Dataset is too small for the selected test ratio"
    );

    let test_indices = &indices[..test_count];
    let train_indices = &indices[test_count..];

    anyhow::ensure!(
        K <= train_indices.len(),
        "K exceeds the number of training samples"
    );

    println!("\nPreparing data...");
    println!("Training samples: {}", train_indices.len());
    println!("Testing samples: {}", test_indices.len());
    println!("K: {}", K);

    let scaler = StandardScaler::fit(&dataset.features, train_indices)?;
    let scaled_features = scaler.transform_matrix(&dataset.features)?;

    let mut train_values = Vec::with_capacity(train_indices.len() * FEATURE_COUNT);
    let mut train_labels = Vec::with_capacity(train_indices.len());

    for &index in train_indices {
        train_values.extend_from_slice(&scaled_features.row(index).to_vec());
        train_labels.push(dataset.labels[index].clone());
    }

    let train_features =
        Array2::from_shape_vec((train_indices.len(), FEATURE_COUNT), train_values)?;

    // Construct a Linfa DatasetBase for compatibility with the Linfa data model.
    let _linfa_dataset =
        DatasetBase::new(train_features.clone(), Array1::from(train_labels.clone()));

    let model = KnnClassifier::new(train_features, train_labels, K)?;

    println!("\nEvaluating model...");

    let mut actual = Vec::with_capacity(test_indices.len());
    let mut predicted = Vec::with_capacity(test_indices.len());

    for &index in test_indices {
        let input = scaled_features.row(index).to_vec();

        predicted.push(model.predict_one(&input)?);
        actual.push(dataset.labels[index].clone());
    }

    let correct = actual
        .iter()
        .zip(predicted.iter())
        .filter(|(a, p)| a == p)
        .count();

    let accuracy = correct as f64 / actual.len() as f64;

    println!("\n========================================");
    println!("           Evaluation Results");
    println!("========================================");

    println!("Correct predictions: {correct}/{}", actual.len());
    println!("Incorrect predictions: {}", actual.len() - correct);
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    println!("\nPer-class metrics");
    println!(
        "{:<20} {:>10} {:>10} {:>10}",
        "Class", "Precision", "Recall", "F1"
    );

    let all_classes: BTreeSet<String> = actual.iter().chain(predicted.iter()).cloned().collect();

    for class in &all_classes {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_count = 0usize;

        for (a, p) in actual.iter().zip(predicted.iter()) {
            if a == class && p == class {
                tp += 1;
            } else if a != class && p == class {
                fp += 1;
            } else if a == class && p != class {
                fn_count += 1;
            }
        }

        let precision = if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        };

        let recall = if tp + fn_count == 0 {
            0.0
        } else {
            tp as f64 / (tp + fn_count) as f64
        };

        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };

        println!(
            "{:<20} {:>9.2}% {:>9.2}% {:>9.2}%",
            class,
            precision * 100.0,
            recall * 100.0,
            f1 * 100.0
        );
    }

    println!("\nConfusion Matrix");
    println!("Rows = actual; columns = predicted\n");

    print!("{:<20}", "Actual / Predicted");

    for class in &all_classes {
        print!("{:>18}", class);
    }

    println!();

    for actual_class in &all_classes {
        print!("{:<20}", actual_class);

        for predicted_class in &all_classes {
            let count = actual
                .iter()
                .zip(predicted.iter())
                .filter(|(a, p)| *a == actual_class && *p == predicted_class)
                .count();

            print!("{:>18}", count);
        }

        println!();
    }

    println!("\nDone!");

    Ok(())
}

fn main() -> Result<()> {
    evaluate()
}
