use anyhow::{Context, Result, bail};
use linfa::Dataset;
use linfa::traits::{Fit, Predict};
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use serde::Deserialize;
use std::fs::File;
use std::path::Path;

const DATASET_PATH: &str = "data/houses.csv";
const FEATURE_COUNT: usize = 5;
const TRAIN_RATIO: f64 = 0.8;

#[derive(Debug, Deserialize, Clone)]
struct House {
    area_sqft: f64,
    bedrooms: f64,
    bathrooms: f64,
    age_years: f64,
    distance_km: f64,
    price_usd: f64,
}

impl House {
    fn features(&self) -> [f64; FEATURE_COUNT] {
        [
            self.area_sqft,
            self.bedrooms,
            self.bathrooms,
            self.age_years,
            self.distance_km,
        ]
    }
}

#[derive(Debug)]
struct StandardScaler {
    means: [f64; FEATURE_COUNT],
    stds: [f64; FEATURE_COUNT],
}

impl StandardScaler {
    fn fit(houses: &[House]) -> Result<Self> {
        if houses.is_empty() {
            bail!("Cannot fit a scaler on an empty dataset");
        }

        let mut means = [0.0; FEATURE_COUNT];
        let mut stds = [0.0; FEATURE_COUNT];

        for house in houses {
            for (index, value) in house.features().iter().enumerate() {
                means[index] += value;
            }
        }

        for mean in &mut means {
            *mean /= houses.len() as f64;
        }

        for house in houses {
            for (index, value) in house.features().iter().enumerate() {
                let difference = value - means[index];
                stds[index] += difference * difference;
            }
        }

        for std in &mut stds {
            *std = (*std / houses.len() as f64).sqrt();

            // Avoid division by zero for constant features.
            if *std == 0.0 {
                *std = 1.0;
            }
        }

        Ok(Self { means, stds })
    }

    fn transform(&self, features: &[f64; FEATURE_COUNT]) -> [f64; FEATURE_COUNT] {
        let mut output = [0.0; FEATURE_COUNT];

        for index in 0..FEATURE_COUNT {
            output[index] =
                (features[index] - self.means[index]) / self.stds[index];
        }

        output
    }
}

fn load_dataset(path: &str) -> Result<Vec<House>> {
    let file = File::open(path)
        .with_context(|| format!("Could not open dataset: {path}"))?;

    let mut reader = csv::Reader::from_reader(file);

    let mut houses = Vec::new();

    for record in reader.deserialize::<House>() {
        let house = record.context("Could not parse a CSV row")?;

        let values = [
            house.area_sqft,
            house.bedrooms,
            house.bathrooms,
            house.age_years,
            house.distance_km,
            house.price_usd,
        ];

        if values.iter().any(|value| !value.is_finite()) {
            bail!("Dataset contains a non-finite numeric value");
        }

        if house.area_sqft <= 0.0 || house.price_usd <= 0.0 {
            bail!("Area and price must be greater than zero");
        }

        houses.push(house);
    }

    if houses.len() < 10 {
        bail!(
            "Dataset must contain at least 10 valid records; found {}",
            houses.len()
        );
    }

    Ok(houses)
}

fn make_feature_matrix(
    houses: &[House],
    scaler: &StandardScaler,
) -> Result<Array2<f64>> {
    let mut values = Vec::with_capacity(houses.len() * FEATURE_COUNT);

    for house in houses {
        let normalized = scaler.transform(&house.features());
        values.extend_from_slice(&normalized);
    }

    Array2::from_shape_vec((houses.len(), FEATURE_COUNT), values)
        .context("Could not construct feature matrix")
}

fn make_target_vector(houses: &[House]) -> Array1<f64> {
    Array1::from_iter(houses.iter().map(|house| house.price_usd))
}

fn calculate_mae(actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    actual
        .iter()
        .zip(predicted.iter())
        .map(|(actual, predicted)| (actual - predicted).abs())
        .sum::<f64>()
        / actual.len() as f64
}

fn calculate_rmse(actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    let mean_squared_error = actual
        .iter()
        .zip(predicted.iter())
        .map(|(actual, predicted)| (actual - predicted).powi(2))
        .sum::<f64>()
        / actual.len() as f64;

    mean_squared_error.sqrt()
}

fn calculate_r2(actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    let mean = actual.sum() / actual.len() as f64;

    let total_sum_squares = actual
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();

    let residual_sum_squares = actual
        .iter()
        .zip(predicted.iter())
        .map(|(actual, predicted)| (actual - predicted).powi(2))
        .sum::<f64>();

    if total_sum_squares == 0.0 {
        return 0.0;
    }

    1.0 - residual_sum_squares / total_sum_squares
}

fn predict_house_price(
    model: &linfa_linear::FittedLinearRegression<f64>,
    scaler: &StandardScaler,
    features: [f64; FEATURE_COUNT],
) -> Result<f64> {
    let normalized = scaler.transform(&features);

    let input = Array2::from_shape_vec(
        (1, FEATURE_COUNT),
        normalized.to_vec(),
    )?;

    let prediction = model.predict(&input);

    prediction
        .get(0)
        .copied()
        .context("Model did not return a prediction")
}

fn main() -> Result<()> {
    println!("======================================");
    println!("      Rust House Price Prediction");
    println!("      Linear Regression + Linfa");
    println!("======================================\n");

    if !Path::new(DATASET_PATH).exists() {
        bail!(
            "Dataset not found: {DATASET_PATH}\n\
             Create data/houses.csv using the supplied example dataset."
        );
    }

    // 1. Load data.
    let mut houses = load_dataset(DATASET_PATH)?;

    println!("Dataset: {DATASET_PATH}");
    println!("Total houses: {}", houses.len());

    // 2. Shuffle deterministically for reproducible results.
    let mut rng = StdRng::seed_from_u64(42);
    houses.shuffle(&mut rng);

    // 3. Split into training and test datasets.
    let train_size = ((houses.len() as f64) * TRAIN_RATIO) as usize;

    if train_size == 0 || train_size >= houses.len() {
        bail!("Unable to split the dataset into training and test sets");
    }

    let train_houses = &houses[..train_size];
    let test_houses = &houses[train_size..];

    println!("Training records: {}", train_houses.len());
    println!("Test records: {}\n", test_houses.len());

    // 4. Fit normalization using training records only.
    let scaler = StandardScaler::fit(train_houses)?;

    // 5. Convert data into matrices and target vectors.
    let x_train = make_feature_matrix(train_houses, &scaler)?;
    let y_train = make_target_vector(train_houses);

    let x_test = make_feature_matrix(test_houses, &scaler)?;
    let y_test = make_target_vector(test_houses);

    let train_dataset = Dataset::new(x_train, y_train);

    // 6. Train linear regression.
    println!("Training Linear Regression...");

    let model = LinearRegression::default()
        .fit(&train_dataset)
        .context("Could not train the regression model")?;

    println!("Training complete.\n");

    // 7. Evaluate on unseen test records.
    let predictions = model.predict(&x_test);

    let mae = calculate_mae(&y_test, &predictions);
    let rmse = calculate_rmse(&y_test, &predictions);
    let r2 = calculate_r2(&y_test, &predictions);

    println!("============== Evaluation ==============");
    println!("MAE:  ${mae:,.2}");
    println!("RMSE: ${rmse:,.2}");
    println!("R²:   {r2:.4}");
    println!("========================================\n");

    println!("Test-set predictions:");
    println!(
        "{:<12} {:<15} {:<15}",
        "Actual", "Predicted", "Absolute Error"
    );

    for (actual, predicted) in y_test.iter().zip(predictions.iter()) {
        println!(
            "${:<11.2} ${:<14.2} ${:<14.2}",
            actual,
            predicted,
            (actual - predicted).abs()
        );
    }

    // 8. Predict the price of a new example house.
    let example = [
        1800.0, // Area in square feet.
        3.0,    // Bedrooms.
        2.0,    // Bathrooms.
        5.0,    // Age in years.
        4.0,    // Distance from city center in kilometers.
    ];

    let estimated_price = predict_house_price(&model, &scaler, example)?;

    println!("\n============== New Prediction ==============");
    println!("Area:          {:.0} sq ft", example[0]);
    println!("Bedrooms:      {:.0}", example[1]);
    println!("Bathrooms:     {:.0}", example[2]);
    println!("Age:           {:.0} years", example[3]);
    println!("Distance:      {:.0} km", example[4]);
    println!("--------------------------------------------");
    println!("Estimated price: ${estimated_price:,.2}");
    println!("============================================");

    Ok(())
}
