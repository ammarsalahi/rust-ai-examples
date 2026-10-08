use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, linear, loss::cross_entropy,
};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use rand::seq::SliceRandom;
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};
use tokenizers::{
    PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, models::wordpiece::WordPiece,
    normalizers::BertNormalizer, pre_tokenizers::bert::BertPreTokenizer,
    processors::template::TemplateProcessing,
};

// ============================================================
// Fixed paths
// ============================================================

const DATASET_PATH: &str = "data/dataset.csv";

const CONFIG_PATH: &str = "models/config.json";
const WEIGHTS_PATH: &str = "models/model.safetensors";
const VOCAB_PATH: &str = "models/vocab.txt";

const MODELS_DIR: &str = "models";

// ============================================================
// Training configuration
// ============================================================

const MAX_LENGTH: usize = 64;

const EPOCHS: usize = 10;

const LEARNING_RATE: f64 = 0.001;

const CONFIDENCE_THRESHOLD: f32 = 0.60;

// ============================================================
// Dataset
// ============================================================

#[derive(Debug, Deserialize, Clone)]
struct Sample {
    text: String,
    label: String,
}

#[derive(Debug, Clone)]
struct LabeledSample {
    text: String,
    label: usize,
}

struct Dataset {
    samples: Vec<LabeledSample>,
    labels: Vec<String>,
}

struct DatasetSplit {
    train: Vec<LabeledSample>,
    validation: Vec<LabeledSample>,
}

// ============================================================
// Local model files
// ============================================================

#[derive(Debug, Clone)]
struct ModelFiles {
    config: PathBuf,
    vocab: PathBuf,
    weights: PathBuf,
}

// ============================================================
// BERT encoder
// ============================================================

struct Encoder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    hidden_size: usize,
}

// ============================================================
// Classification head
// ============================================================

struct Classifier {
    linear: Linear,
    varmap: VarMap,
    hidden_size: usize,
    num_classes: usize,
}

// ============================================================
// Main
// ============================================================

fn main() -> Result<()> {
    fs::create_dir_all("data").context("failed to create data directory")?;

    fs::create_dir_all(MODELS_DIR).context("failed to create models directory")?;

    println!("============================================================");
    println!("              Rust Text Classification with Bert");
    println!("============================================================");

    println!();
    println!("Fixed configuration:");
    println!("  Dataset : {}", DATASET_PATH);
    println!("  Config  : {}", CONFIG_PATH);
    println!("  Weights : {}", WEIGHTS_PATH);
    println!("  Vocab   : {}", VOCAB_PATH);

    // --------------------------------------------------------
    // Check required files
    // --------------------------------------------------------

    check_required_file(DATASET_PATH)?;
    check_required_file(CONFIG_PATH)?;
    check_required_file(WEIGHTS_PATH)?;
    check_required_file(VOCAB_PATH)?;

    let dataset_path = Path::new(DATASET_PATH);

    let dataset = load_dataset(dataset_path)?;

    if dataset.samples.is_empty() {
        return Err(anyhow!("dataset is empty"));
    }

    print_dataset_stats(&dataset);

    // --------------------------------------------------------
    // Device
    // --------------------------------------------------------

    let device = Device::Cpu;

    println!();
    println!("Device      : CPU");

    // --------------------------------------------------------
    // Local model
    // --------------------------------------------------------

    let model_files = ModelFiles {
        config: PathBuf::from(CONFIG_PATH),
        vocab: PathBuf::from(VOCAB_PATH),
        weights: PathBuf::from(WEIGHTS_PATH),
    };

    // --------------------------------------------------------
    // Load BERT encoder
    // --------------------------------------------------------

    let encoder = load_encoder(&model_files, device.clone())?;

    println!();
    println!("Model       : local BERT");
    println!("Hidden size : {}", encoder.hidden_size);
    println!("Max length  : {}", MAX_LENGTH);
    println!("Classes     : {}", dataset.labels.len());

    // --------------------------------------------------------
    // Train / validation split
    // --------------------------------------------------------

    let split = stratified_split(&dataset.samples, 0.8);

    println!();
    println!("Dataset split:");
    println!("  Training   : {}", split.train.len());
    println!("  Validation : {}", split.validation.len());

    // --------------------------------------------------------
    // Classifier path
    // --------------------------------------------------------

    let classifier_path = classifier_path(dataset.labels.len(), encoder.hidden_size);

    println!();
    println!("Classifier:");
    println!("  {}", classifier_path.display());

    // --------------------------------------------------------
    // Create classifier
    // --------------------------------------------------------

    let mut classifier = create_classifier(&device, encoder.hidden_size, dataset.labels.len())?;

    // --------------------------------------------------------
    // Load existing classifier or train new one
    // --------------------------------------------------------

    if classifier_path.exists() {
        println!();
        println!("Existing classifier found.");
        println!("Loading classifier...");

        classifier
            .varmap
            .load(&classifier_path)
            .context("failed to load classifier")?;

        println!("Classifier loaded successfully.");
    } else {
        println!();
        println!("No trained classifier found.");
        println!("Starting classifier training...");

        // ----------------------------------------------------
        // Encode training data
        // ----------------------------------------------------

        println!();
        println!("Encoding training samples...");

        let train_embeddings = encode_samples(&encoder, &split.train)?;

        // ----------------------------------------------------
        // Encode validation data
        // ----------------------------------------------------

        println!();
        println!("Encoding validation samples...");

        let validation_embeddings = encode_samples(&encoder, &split.validation)?;

        // ----------------------------------------------------
        // Train classifier
        // ----------------------------------------------------

        println!();
        println!("Training classifier...");

        train_classifier(&mut classifier, &train_embeddings, &validation_embeddings)?;

        // ----------------------------------------------------
        // Save classifier
        // ----------------------------------------------------

        classifier
            .varmap
            .save(&classifier_path)
            .context("failed to save classifier")?;

        println!();
        println!("============================================================");
        println!("Classifier saved:");
        println!("  {}", classifier_path.display());
        println!("============================================================");
    }

    // --------------------------------------------------------
    // Interactive prediction
    // --------------------------------------------------------

    interactive_loop(&encoder, &classifier, &dataset.labels)?;

    Ok(())
}

// ============================================================
// Check required file
// ============================================================

fn check_required_file(path: &str) -> Result<()> {
    let path = Path::new(path);

    if !path.exists() {
        return Err(anyhow!(
            "required file not found: {}\n\
             Please place the file at this path.",
            path.display()
        ));
    }

    if !path.is_file() {
        return Err(anyhow!("path exists but is not a file: {}", path.display()));
    }

    Ok(())
}

// ============================================================
// Load dataset
// ============================================================

fn load_dataset(path: &Path) -> Result<Dataset> {
    println!();
    println!("Loading dataset...");
    println!("  {}", path.display());

    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open dataset: {}", path.display()))?;

    let mut raw_samples = Vec::<Sample>::new();

    for result in reader.deserialize::<Sample>() {
        let sample = result.context("failed to parse CSV row")?;

        if sample.text.trim().is_empty() {
            continue;
        }

        if sample.label.trim().is_empty() {
            continue;
        }

        raw_samples.push(sample);
    }

    if raw_samples.is_empty() {
        return Err(anyhow!("dataset contains no valid samples"));
    }

    // --------------------------------------------------------
    // Find unique labels
    // --------------------------------------------------------

    let unique_labels: BTreeSet<String> = raw_samples
        .iter()
        .map(|sample| sample.label.trim().to_string())
        .collect();

    let labels: Vec<String> = unique_labels.into_iter().collect();

    if labels.len() < 2 {
        return Err(anyhow!("dataset must contain at least 2 different labels"));
    }

    // --------------------------------------------------------
    // Convert labels to numeric IDs
    // --------------------------------------------------------

    let mut samples = Vec::with_capacity(raw_samples.len());

    for sample in raw_samples {
        let label_name = sample.label.trim();

        let label_id = labels
            .iter()
            .position(|label| label == label_name)
            .ok_or_else(|| anyhow!("failed to map label `{}`", label_name))?;

        samples.push(LabeledSample {
            text: sample.text,
            label: label_id,
        });
    }

    println!();
    println!("Loaded {} samples.", samples.len());

    println!("Found {} classes.", labels.len());

    Ok(Dataset { samples, labels })
}

// ============================================================
// Dataset statistics
// ============================================================

fn print_dataset_stats(dataset: &Dataset) {
    let mut counts = vec![0usize; dataset.labels.len()];

    for sample in &dataset.samples {
        if sample.label < counts.len() {
            counts[sample.label] += 1;
        }
    }

    println!();
    println!("Dataset statistics:");
    println!("  Total samples: {}", dataset.samples.len());

    println!();

    for (index, label) in dataset.labels.iter().enumerate() {
        println!("  {:>3}. {:<20} {}", index, label, counts[index]);
    }
}

// ============================================================
// Load local BERT
// ============================================================

fn load_encoder(files: &ModelFiles, device: Device) -> Result<Encoder> {
    // --------------------------------------------------------
    // config.json
    // --------------------------------------------------------

    let config_text = fs::read_to_string(&files.config).context("failed to read config.json")?;

    let config: BertConfig =
        serde_json::from_str(&config_text).context("failed to parse BERT config.json")?;

    println!();
    println!("Model configuration:");
    println!("  hidden size     : {}", config.hidden_size);
    println!("  layers          : {}", config.num_hidden_layers);
    println!("  attention heads : {}", config.num_attention_heads);
    println!("  vocabulary      : {}", config.vocab_size);

    // --------------------------------------------------------
    // vocab.txt
    // --------------------------------------------------------

    let tokenizer = build_tokenizer(&files.vocab)?;

    println!("Tokenizer ready.");

    // --------------------------------------------------------
    // model.safetensors
    // --------------------------------------------------------

    println!();
    println!("Loading BERT weights...");
    println!("  {}", files.weights.display());

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&files.weights], DType::F32, &device) }
        .context("failed to memory-map model.safetensors")?;

    let model = BertModel::load(vb, &config).context("failed to load BERT model")?;

    println!("BERT model loaded.");

    Ok(Encoder {
        model,
        tokenizer,
        device,
        hidden_size: config.hidden_size,
    })
}

// ============================================================
// Build tokenizer from vocab.txt
// ============================================================

fn build_tokenizer(vocab_path: &Path) -> Result<Tokenizer> {
    let vocab = vocab_path.to_string_lossy().to_string();

    let wordpiece = WordPiece::from_file(&vocab)
        .unk_token("[UNK]".to_string())
        .continuing_subword_prefix("##".to_string())
        .build()
        .map_err(|e| anyhow!("failed to build WordPiece tokenizer: {e}"))?;

    let mut tokenizer = Tokenizer::new(wordpiece);

    // --------------------------------------------------------
    // Normalizer
    // --------------------------------------------------------

    let normalizer = BertNormalizer::new(true, true, Some(true), true);

    tokenizer.with_normalizer(Some(normalizer));

    // --------------------------------------------------------
    // Pre-tokenizer
    // --------------------------------------------------------

    tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));

    // --------------------------------------------------------
    // Post-processor
    // --------------------------------------------------------

    let processor = TemplateProcessing::builder()
        .try_single("[CLS] $A [SEP]")
        .map_err(|e| anyhow!("failed to configure tokenizer template: {e}"))?
        .special_tokens(vec![("[CLS]".to_string(), 101), ("[SEP]".to_string(), 102)])
        .build()
        .map_err(|e| anyhow!("failed to build tokenizer processor: {e}"))?;

    tokenizer.with_post_processor(Some(processor));

    // --------------------------------------------------------
    // Truncation
    // --------------------------------------------------------

    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: MAX_LENGTH,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("failed to configure truncation: {e}"))?;

    // --------------------------------------------------------
    // Padding
    // --------------------------------------------------------

    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(MAX_LENGTH),
        ..Default::default()
    }));

    Ok(tokenizer)
}

// ============================================================
// Stratified train / validation split
// ============================================================

fn stratified_split(samples: &[LabeledSample], train_ratio: f32) -> DatasetSplit {
    let max_label = samples.iter().map(|sample| sample.label).max().unwrap_or(0);

    let mut groups: Vec<Vec<LabeledSample>> = (0..=max_label).map(|_| Vec::new()).collect();

    for sample in samples {
        groups[sample.label].push(sample.clone());
    }

    let mut rng = rand::rng();

    let mut train = Vec::new();
    let mut validation = Vec::new();

    for group in &mut groups {
        group.shuffle(&mut rng);

        let mut train_size = ((group.len() as f32) * train_ratio).round() as usize;

        if group.len() > 1 {
            train_size = train_size.max(1).min(group.len() - 1);
        } else {
            train_size = group.len();
        }

        train.extend_from_slice(&group[..train_size]);

        validation.extend_from_slice(&group[train_size..]);
    }

    train.shuffle(&mut rng);
    validation.shuffle(&mut rng);

    DatasetSplit { train, validation }
}

// ============================================================
// Encode dataset samples
// ============================================================

fn encode_samples(encoder: &Encoder, samples: &[LabeledSample]) -> Result<Vec<(Vec<f32>, usize)>> {
    let mut result = Vec::with_capacity(samples.len());

    for (index, sample) in samples.iter().enumerate() {
        let embedding = encode_text(encoder, &sample.text)?;

        result.push((embedding, sample.label));

        if (index + 1) % 10 == 0 || index + 1 == samples.len() {
            println!("  encoded {}/{}", index + 1, samples.len());
        }
    }

    Ok(result)
}

// ============================================================
// Encode one text
// ============================================================

fn encode_text(encoder: &Encoder, text: &str) -> Result<Vec<f32>> {
    // --------------------------------------------------------
    // Tokenize
    // --------------------------------------------------------

    let encoding = encoder
        .tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("tokenization failed: {e}"))?;

    // --------------------------------------------------------
    // Input IDs
    // --------------------------------------------------------

    let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

    // --------------------------------------------------------
    // Token type IDs
    // --------------------------------------------------------

    let type_ids: Vec<i64> = encoding
        .get_type_ids()
        .iter()
        .map(|&id| id as i64)
        .collect();

    // --------------------------------------------------------
    // Attention mask
    // --------------------------------------------------------

    let attention_mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&id| id as i64)
        .collect();

    // --------------------------------------------------------
    // Validate length
    // --------------------------------------------------------

    if ids.len() != MAX_LENGTH {
        return Err(anyhow!(
            "unexpected tokenized length: {}, expected {}",
            ids.len(),
            MAX_LENGTH
        ));
    }

    // --------------------------------------------------------
    // Candle tensors
    // --------------------------------------------------------

    let input_ids = Tensor::new(ids.as_slice(), &encoder.device)?.unsqueeze(0)?;

    let token_type_ids = Tensor::new(type_ids.as_slice(), &encoder.device)?.unsqueeze(0)?;

    let attention_mask = Tensor::new(attention_mask.as_slice(), &encoder.device)?.unsqueeze(0)?;

    // --------------------------------------------------------
    // BERT forward pass
    // --------------------------------------------------------

    let hidden = encoder
        .model
        .forward(&input_ids, &token_type_ids, Some(&attention_mask))
        .context("BERT forward pass failed")?;

    // --------------------------------------------------------
    // CLS token
    // --------------------------------------------------------

    let sequence = hidden.get(0)?;

    let cls = sequence.get(0)?.contiguous()?;

    let embedding = cls.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    // --------------------------------------------------------
    // Validate embedding size
    // --------------------------------------------------------

    if embedding.len() != encoder.hidden_size {
        return Err(anyhow!(
            "unexpected embedding size: {}, expected {}",
            embedding.len(),
            encoder.hidden_size
        ));
    }

    Ok(embedding)
}

// ============================================================
// Classifier path
// ============================================================

fn classifier_path(num_classes: usize, hidden_size: usize) -> PathBuf {
    PathBuf::from(format!(
        "{}/classifier_h{}_c{}.safetensors",
        MODELS_DIR, hidden_size, num_classes
    ))
}

// ============================================================
// Create classifier
// ============================================================

fn create_classifier(
    device: &Device,
    hidden_size: usize,
    num_classes: usize,
) -> Result<Classifier> {
    if num_classes < 2 {
        return Err(anyhow!("classifier requires at least 2 classes"));
    }

    let varmap = VarMap::new();

    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

    let linear_layer = linear(hidden_size, num_classes, vb.pp("classifier"))?;

    Ok(Classifier {
        linear: linear_layer,
        varmap,
        hidden_size,
        num_classes,
    })
}

// ============================================================
// Train classifier
// ============================================================

fn train_classifier(
    classifier: &mut Classifier,
    train: &[(Vec<f32>, usize)],
    validation: &[(Vec<f32>, usize)],
) -> Result<()> {
    let mut optimizer = AdamW::new(
        classifier.varmap.all_vars(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        },
    )?;

    let device = classifier
        .varmap
        .all_vars()
        .first()
        .ok_or_else(|| anyhow!("classifier has no variables"))?
        .device()
        .clone();

    println!();
    println!("Classifier architecture:");
    println!("  input  : {}", classifier.hidden_size);
    println!("  output : {}", classifier.num_classes);

    println!();

    for epoch in 1..=EPOCHS {
        let mut total_loss = 0.0f64;

        let mut correct = 0usize;

        // ----------------------------------------------------
        // Shuffle training data
        // ----------------------------------------------------

        let mut indices: Vec<usize> = (0..train.len()).collect();

        indices.shuffle(&mut rand::rng());

        // ----------------------------------------------------
        // Training
        // ----------------------------------------------------

        for &index in &indices {
            let (embedding, label) = &train[index];

            let input =
                Tensor::new(embedding.as_slice(), &device)?.reshape((1, classifier.hidden_size))?;

            let target = Tensor::new(&[*label as u32], &device)?;

            let logits = classifier.linear.forward(&input)?;

            let loss = cross_entropy(&logits, &target)?;

            let loss_value = loss.to_scalar::<f32>()?;

            total_loss += loss_value as f64;

            let prediction = argmax_row(&logits)?;

            if prediction == *label {
                correct += 1;
            }

            optimizer.backward_step(&loss)?;
        }

        // ----------------------------------------------------
        // Accuracy
        // ----------------------------------------------------

        let train_accuracy = if train.is_empty() {
            0.0
        } else {
            correct as f64 / train.len() as f64
        };

        let validation_accuracy = evaluate_classifier(classifier, validation)?;

        let average_loss = if train.is_empty() {
            0.0
        } else {
            total_loss / train.len() as f64
        };

        println!(
            "epoch {:>2}/{:<2} | loss {:.4} | train {:.2}% | validation {:.2}%",
            epoch,
            EPOCHS,
            average_loss,
            train_accuracy * 100.0,
            validation_accuracy * 100.0,
        );
    }

    Ok(())
}

// ============================================================
// Evaluate classifier
// ============================================================

fn evaluate_classifier(classifier: &Classifier, dataset: &[(Vec<f32>, usize)]) -> Result<f64> {
    if dataset.is_empty() {
        return Ok(0.0);
    }

    let device = classifier
        .varmap
        .all_vars()
        .first()
        .ok_or_else(|| anyhow!("classifier has no variables"))?
        .device()
        .clone();

    let mut correct = 0usize;

    for (embedding, label) in dataset {
        let input =
            Tensor::new(embedding.as_slice(), &device)?.reshape((1, classifier.hidden_size))?;

        let logits = classifier.linear.forward(&input)?;

        let prediction = argmax_row(&logits)?;

        if prediction == *label {
            correct += 1;
        }
    }

    Ok(correct as f64 / dataset.len() as f64)
}

// ============================================================
// Argmax
// ============================================================

fn argmax_row(logits: &Tensor) -> Result<usize> {
    let values = logits.flatten_all()?.to_vec1::<f32>()?;

    let mut best_index = 0usize;

    let mut best_value = f32::NEG_INFINITY;

    for (index, &value) in values.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }

    Ok(best_index)
}

// ============================================================
// Prediction
// ============================================================

fn predict(encoder: &Encoder, classifier: &Classifier, text: &str) -> Result<(usize, Vec<f32>)> {
    // --------------------------------------------------------
    // Encode input
    // --------------------------------------------------------

    let embedding = encode_text(encoder, text)?;

    let device = classifier
        .varmap
        .all_vars()
        .first()
        .ok_or_else(|| anyhow!("classifier has no variables"))?
        .device()
        .clone();

    // --------------------------------------------------------
    // Classifier input
    // --------------------------------------------------------

    let input = Tensor::new(embedding.as_slice(), &device)?.reshape((1, classifier.hidden_size))?;

    // --------------------------------------------------------
    // Logits
    // --------------------------------------------------------

    let logits = classifier.linear.forward(&input)?;

    // --------------------------------------------------------
    // Softmax
    // --------------------------------------------------------

    let probabilities = candle_nn::ops::softmax(&logits, 1)?;

    let probabilities = probabilities.flatten_all()?.to_vec1::<f32>()?;

    // --------------------------------------------------------
    // Find best class
    // --------------------------------------------------------

    let mut best_index = 0usize;

    let mut best_probability = f32::NEG_INFINITY;

    for (index, &probability) in probabilities.iter().enumerate() {
        if probability > best_probability {
            best_probability = probability;

            best_index = index;
        }
    }

    Ok((best_index, probabilities))
}

// ============================================================
// Interactive prediction loop
// ============================================================

fn interactive_loop(encoder: &Encoder, classifier: &Classifier, labels: &[String]) -> Result<()> {
    println!();
    println!("============================================================");
    println!("                  Interactive Prediction");
    println!("============================================================");

    println!();
    println!("Classes:");

    for (index, label) in labels.iter().enumerate() {
        println!("  {} -> {}", index, label);
    }

    println!();
    println!("Type text and press Enter.");
    println!("Type 'exit' or 'quit' to stop.");

    println!();

    let stdin = io::stdin();

    loop {
        print!("text> ");

        io::stdout().flush()?;

        let mut input = String::new();

        stdin.read_line(&mut input)?;

        let input = input.trim();

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            println!("Goodbye.");
            break;
        }

        if input.is_empty() {
            continue;
        }

        match predict(encoder, classifier, input) {
            Ok((class_id, probabilities)) => {
                println!();

                println!("Prediction: {}", labels[class_id]);

                println!();

                println!("Probabilities:");

                for (index, probability) in probabilities.iter().enumerate() {
                    println!("  {:<20} {:.2}%", labels[index], probability * 100.0);
                }

                let confidence = probabilities[class_id];

                println!();

                println!("Confidence: {:.2}%", confidence * 100.0);

                if confidence < CONFIDENCE_THRESHOLD {
                    println!();
                    println!("Warning: low confidence.");
                    println!("The text may be outside the training domains.");
                }

                println!();
            }

            Err(error) => {
                eprintln!("Prediction error: {error:#}");
            }
        }
    }

    Ok(())
}
