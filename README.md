# rust-ai-examples🦀🤖

This is a project designed to assist those who wish to work with the Rust language in the field of artificial intelligence and its sub-disciplines.

## Examples

| Example | Description | Directory |
|---------|-------------|-----------|
| **Text Classification** | BERT-based text classification with custom dataset training and interactive prediction using Candle and tokenizers | [examples/text-classification](./examples/text-classification) |
| **Email Spam Classification** | Spam detection using transformer models (work in progress) | [examples/email-spam-classification](./examples/email-spam-classification) |
| **Language Detection** | Logistic regression-based language identification with character n-gram features using Linfa | [examples/language-detection](./examples/language-detection) |
| **KNN Classifier** | K-nearest neighbors classification on the Iris dataset with evaluation metrics using Linfa | [examples/knn-classifier](./examples/knn-classifier) |
| **Sentiment Analysis** | DistilBERT-based sentiment analysis (positive/negative) with dataset evaluation and interactive CLI using Candle and Hugging Face Hub | [examples/sentiment-analysis](./examples/sentiment-analysis) |
| **OCR** | TrOCR-based optical character recognition using Candle and Hugging Face transformers | [examples/OCR](./examples/OCR) |
| **Decision Tree Classifier** | Loan approval prediction using decision trees with evaluation metrics using Linfa | [examples/decision-tree-classifier](./examples/decision-tree-classifier) |
| **House Price Prediction** | Linear regression for house price prediction with MAE, RMSE, and R² metrics using Linfa | [examples/house-price-prediction](./examples/house-price-prediction) |
| **Image Classification** | ResNet-18 based image classification on ImageNet using Candle | [examples/image-classification](./examples/image-classification) |
| **Keyword Extraction** | TF-IDF based keyword and keyphrase extraction from text documents using regex and stemming | [examples/keyword-extraction](./examples/keyword-extraction) |

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
