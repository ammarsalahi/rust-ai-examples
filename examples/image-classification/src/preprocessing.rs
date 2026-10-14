use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use image::imageops::FilterType;
use std::path::Path;

const IMAGE_SIZE: u32 = 224;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub fn load_image(path: &Path, device: &Device) -> Result<Tensor> {
    let image = image::open(path)
        .with_context(|| format!("Could not open image: {}", path.display()))?
        .to_rgb8();

    let resized = image::imageops::resize(&image, IMAGE_SIZE, IMAGE_SIZE, FilterType::Triangle);

    let mut channels = vec![0.0_f32; 3 * IMAGE_SIZE as usize * IMAGE_SIZE as usize];

    let plane_size = (IMAGE_SIZE * IMAGE_SIZE) as usize;

    for (x, y, pixel) in resized.enumerate_pixels() {
        let index = (y * IMAGE_SIZE + x) as usize;

        for channel in 0..3 {
            let value = pixel[channel] as f32 / 255.0;

            channels[channel * plane_size + index] = (value - MEAN[channel]) / STD[channel];
        }
    }

    let tensor = Tensor::from_vec(
        channels,
        (1, 3, IMAGE_SIZE as usize, IMAGE_SIZE as usize),
        device,
    )?;

    Ok(tensor)
}
