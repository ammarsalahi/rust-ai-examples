# rust-ai-examples

This is a project designed to assist those who wish to work with the Rust language in the field of artificial intelligence and its sub-disciplines.

## Examples

| Example | Description | Directory |
|---------|-------------|-----------|
| **Text Classification** | BERT-based text classification with custom dataset training and interactive prediction using Candle and tokenizers | [examples/text-classification](./examples/text-classification) |
| **Email Spam Classification** | Spam detection using transformer models (work in progress) | [examples/email-spam-classification](./examples/email-spam-classification) |
| **Language Detection** | Logistic regression-based language identification with character n-gram features using Linfa | [examples/language-detection](./examples/language-detection) |
| **KNN Classifier** | K-nearest neighbors classification on the Iris dataset with evaluation metrics using Linfa | [examples/knn-classifier](./examples/knn-classifier) |
| **Sentiment Analysis** | DistilBERT-based sentiment analysis (positive/negative) with dataset evaluation and interactive CLI using Candle and Hugging Face Hub | [examples/sentiment-analysis](./examples/sentiment-analysis) |

## Running Examples

Each example is a standalone Cargo project. Navigate to the example directory and run:

```bash
cd examples/<example-name>
cargo run
```

For the sentiment analysis example, you can also use CLI commands:

```bash
cargo run -- predict "I love this!"
cargo run -- evaluate
cargo run -- evaluate-file path/to/dataset.csv
```