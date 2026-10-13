use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, ImageReader};

pub struct ImageProcessor {
    width: u32,
    height: u32,
}

impl ImageProcessor {
    pub fn new() -> Self {
        Self {
            width: 384,
            height: 384,
        }
    }

    pub fn load(&self, path: &str) -> Result<DynamicImage> {
        let image = ImageReader::open(path)
            .with_context(|| format!("Cannot open image: {path}"))?
            .decode()
            .context("Cannot decode image")?;

        Ok(image)
    }

    pub fn preprocess(&self, image: &DynamicImage, device: &Device) -> Result<Tensor> {
        let resized = image.resize_exact(
            self.width,
            self.height,
            image::imageops::FilterType::Triangle,
        );

        let rgb = resized.to_rgb8();
        let raw = rgb.into_raw();

        let tensor = Tensor::from_vec(raw, (self.height as usize, self.width as usize, 3), device)?
            .permute((2, 0, 1))?
            .to_dtype(DType::F32)?;

        // Convert pixels from [0, 255] to [0, 1].
        let tensor = (tensor / 255.0)?;

        // Normalize using mean=0.5 and std=0.5 per RGB channel.
        let mean = Tensor::from_vec(vec![0.5_f32, 0.5, 0.5], (3, 1, 1), device)?;

        let std = Tensor::from_vec(vec![0.5_f32, 0.5, 0.5], (3, 1, 1), device)?;

        let normalized = tensor.broadcast_sub(&mean)?.broadcast_div(&std)?;

        // TrOCR expects a batch dimension: [1, 3, 384, 384].
        Ok(normalized.unsqueeze(0)?)
    }
}
