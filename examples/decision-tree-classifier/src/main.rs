use anyhow::{bail, Context, Result};
use linfa::prelude::{Fit, Predict};
use linfa::Dataset;
use linfa_trees::DecisionTree;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Deserialize;
use std::fs::File;
use std::path::Path;

const DATASET_PATH: &str = "data/loans.csv";
const FEATURE_COUNT: usize = 4;
const TRAIN_RATIO: f64 = 0.8;
const MAX_TREE_DEPTH: usize = 5;

#[derive(Debug, Deserialize, Clone)]
struct LoanRecord {
    credit_score: f64,
    income_k: f64,
    debt_ratio: f64,
    employment_years: f64,
    loan_approved: usize,
}

impl LoanRecord {
    fn features(&self) -> [f64; FEATURE_COUNT] {
        [
            self.credit_score,
            self.income_k,
            self.debt_ratio,
            self.employment_years,
        ]
    }
}

fn load_dataset(path: &str) -> Result<Vec<LoanRecord>> {
    let file = File::open(path).with_context(|| format!("Could not open dataset: {path}"))?;

    let mut reader = csv::Reader::from_reader(file);
    let mut records = Vec::new();

    for result in reader.deserialize::<LoanRecord>() {
        let record = result.context("Could not parse CSV record")?;

        let features = record.features();

        if features.iter().any(|value| !value.is_finite()) {
            bail!("Dataset contains a non-finite feature value");
        }

        if record.credit_score < 0.0
            || record.income_k < 0.0
            || !(0.0..=1.0).contains(&record.debt_ratio)
            || record.employment_years < 0.0
        {
            bail!("Dataset contains an invalid feature value");
        }

        if record.loan_approved > 1 {
            bail!("loan_approved must be 0 or 1");
        }

        records.push(record);
    }

    if records.len() < 10 {
        bail!("At least 10 records are required; found {}", records.len());
    }

    if !records.iter().any(|record| record.loan_approved == 0)
        || !records.iter().any(|record| record.loan_approved == 1)
    {
        bail!("Dataset must contain both classes: 0 and 1");
    }

    Ok(records)
}

fn build_features(records: &[LoanRecord]) -> Result<Array2<f64>> {
    let values: Vec<f64> = records
        .iter()
        .flat_map(|record| record.features())
        .collect();

    Array2::from_shape_vec((records.len(), FEATURE_COUNT), values)
        .context("Could not construct feature matrix")
}

fn build_targets(records: &[LoanRecord]) -> Array1<usize> {
    Array1::from_iter(records.iter().map(|record| record.loan_approved))
}

fn accuracy(actual: &Array1<usize>, predicted: &Array1<usize>) -> f64 {
    if actual.is_empty() || actual.len() != predicted.len() {
        return 0.0;
    }

    let correct = actual
        .iter()
        .zip(predicted.iter())
        .filter(|(actual, predicted)| actual == predicted)
        .count();

    correct as f64 / actual.len() as f64
}

fn print_confusion_matrix(actual: &Array1<usize>, predicted: &Array1<usize>) {
    let mut true_negative = 0;
    let mut false_positive = 0;
    let mut false_negative = 0;
    let mut true_positive = 0;

    for (&actual, &predicted) in actual.iter().zip(predicted.iter()) {
        match (actual, predicted) {
            (0, 0) => true_negative += 1,
            (0, 1) => false_positive += 1,
            (1, 0) => false_negative += 1,
            (1, 1) => true_positive += 1,
            _ => {}
        }
    }

    println!("\nConfusion Matrix");
    println!("Rows = actual class, columns = predicted class");
    println!();
    println!("                 Predicted 0    Predicted 1");
    println!("Actual 0         {:<14} {}", true_negative, false_positive);
    println!("Actual 1         {:<14} {}", false_negative, true_positive);
}

fn class_name(class: usize) -> &'static str {
    match class {
        0 => "Rejected",
        1 => "Approved",
        _ => "Unknown",
    }
}

fn predict_new_loan(
    model: &linfa_trees::DecisionTree<f64, usize>,
    features: [f64; FEATURE_COUNT],
) -> Result<usize> {
    let input = Array2::from_shape_vec((1, FEATURE_COUNT), features.to_vec())?;

    let predictions = model.predict(&input);

    predictions
        .get(0)
        .copied()
        .context("Model did not return a prediction")
}

fn main() -> Result<()> {
    println!("=======================================");
    println!("      Rust Decision Tree Classifier");
    println!("=======================================\n");

    if !Path::new(DATASET_PATH).exists() {
        bail!(
            "Dataset not found: {DATASET_PATH}\n\
             Create data/loans.csv using the supplied dataset."
        );
    }

    // 1. Load dataset.
    let mut records = load_dataset(DATASET_PATH)?;

    println!("Dataset: {DATASET_PATH}");
    println!("Total records: {}", records.len());

    // 2. Shuffle the data reproducibly.
    let mut rng = StdRng::seed_from_u64(42);
    records.shuffle(&mut rng);

    // 3. Split into training and test sets.
    let train_size = ((records.len() as f64) * TRAIN_RATIO) as usize;

    if train_size == 0 || train_size >= records.len() {
        bail!("Could not split dataset into training and test sets");
    }

    let train_records = &records[..train_size];
    let test_records = &records[train_size..];

    println!("Training records: {}", train_records.len());
    println!("Test records:     {}\n", test_records.len());

    // 4. Build matrices.
    let x_train = build_features(train_records)?;
    let y_train = build_targets(train_records);

    let x_test = build_features(test_records)?;
    let y_test = build_targets(test_records);

    let train_dataset = Dataset::new(x_train, y_train);

    // 5. Train the decision tree.
    println!("Training decision tree...");

    let model = DecisionTree::params()
        .max_depth(Some(MAX_TREE_DEPTH))
        .fit(&train_dataset)
        .context("Failed to train decision tree")?;

    println!("Training completed.");
    println!("Maximum tree depth: {}\n", MAX_TREE_DEPTH);

    // 6. Evaluate on unseen test records.
    let predictions = model.predict(&x_test);

    let score = accuracy(&y_test, &predictions);

    println!("============== Evaluation ==============");
    println!("Accuracy: {:.2}%", score * 100.0);
    println!("========================================");

    print_confusion_matrix(&y_test, &predictions);

    // 7. Print test predictions.
    println!("\nTest-set predictions:");
    println!("{:<12} {:<14} {:<14}", "Actual", "Predicted", "Result");

    for (&actual, &predicted) in y_test.iter().zip(predictions.iter()) {
        let result = if actual == predicted {
            "Correct"
        } else {
            "Incorrect"
        };

        println!(
            "{:<12} {:<14} {}",
            class_name(actual),
            class_name(predicted),
            result
        );
    }

    // 8. Classify new examples.
    let examples = [
        ("High credit score", [760.0, 110.0, 0.18, 9.0]),
        ("Low credit score", [540.0, 30.0, 0.75, 1.0]),
    ];

    println!("\n============== New Predictions ==============");

    for (description, features) in examples {
        let prediction = predict_new_loan(&model, features)?;

        println!("\nExample: {description}");
        println!("Credit score:       {:.0}", features[0]);
        println!("Annual income:      ${:.0}K", features[1]);
        println!("Debt-to-income:     {:.0}%", features[2] * 100.0);
        println!("Employment:         {:.0} years", features[3]);
        println!("Predicted class:    {}", class_name(prediction));
    }

    println!("\n=============================================");
    println!("Finished.");

    Ok(())
}
