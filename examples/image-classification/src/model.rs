use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::resnet;
use std::path::Path;

pub struct ImageClassifier {
    model: resnet::ResNet,
    device: Device,
}

impl ImageClassifier {
    pub fn new(model_path: &Path) -> Result<Self> {
        let device = Device::Cpu;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device) }
            .context("Could not load the ResNet safetensors checkpoint")?;

        let model = resnet::resnet18(vb, 1000)
            .context("Could not initialize ResNet-18; check checkpoint compatibility")?;

        Ok(Self { model, device })
    }

    pub fn predict(&self, image: &Tensor, top_k: usize) -> Result<Vec<(usize, f32)>> {
        use candle_nn::Module;

        let logits = self
            .model
            .forward(image)
            .context("Model inference failed")?;

        let probabilities = candle_nn::ops::softmax(&logits, 1)?
            .squeeze(0)?
            .to_vec1::<f32>()?;

        let mut predictions: Vec<(usize, f32)> = probabilities.into_iter().enumerate().collect();

        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        predictions.truncate(top_k);

        Ok(predictions)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
