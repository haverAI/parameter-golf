//! Advanced quantization tool for Parameter Golf.
//! Quantizes PyTorch model weights to int8/int4 with per-row scaling,
//! then zlib compresses. Targets maximum compression within 16MB.

use clap::Parser;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "quantize", about = "Quantize and compress model weights for 16MB target")]
struct Args {
    /// Input: raw f32 weights file (flat binary, row-major)
    #[arg(short, long)]
    input: PathBuf,

    /// Output: compressed quantized weights
    #[arg(short, long)]
    output: PathBuf,

    /// Quantization bits (8 or 4)
    #[arg(short, long, default_value = "8")]
    bits: u8,

    /// Number of parameters (for shape inference)
    #[arg(short, long)]
    num_params: usize,

    /// Target size in bytes (default 16MB)
    #[arg(long, default_value = "16000000")]
    target_bytes: usize,

    /// Just analyze, don't write
    #[arg(long)]
    analyze: bool,
}

fn quantize_int8(weights: &[f32]) -> (Vec<i8>, Vec<f32>) {
    // Per-row scaling (we treat as one big row for simplicity)
    // For real use: would do per-tensor or per-channel scaling
    let chunk_size = 256; // scale per 256 elements
    let mut quantized = Vec::with_capacity(weights.len());
    let mut scales = Vec::with_capacity(weights.len() / chunk_size + 1);

    for chunk in weights.chunks(chunk_size) {
        let max_abs = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        scales.push(scale);

        for &w in chunk {
            let q = (w / scale).round().clamp(-128.0, 127.0) as i8;
            quantized.push(q);
        }
    }

    (quantized, scales)
}

fn quantize_int4(weights: &[f32]) -> (Vec<u8>, Vec<f32>) {
    // Pack two int4 values per byte
    let chunk_size = 256;
    let mut packed = Vec::with_capacity(weights.len() / 2 + 1);
    let mut scales = Vec::with_capacity(weights.len() / chunk_size + 1);

    for chunk in weights.chunks(chunk_size) {
        let max_abs = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
        scales.push(scale);

        // Quantize to int4 (-8..7) and pack pairs
        let quantized: Vec<i8> = chunk
            .iter()
            .map(|&w| (w / scale).round().clamp(-8.0, 7.0) as i8)
            .collect();

        for pair in quantized.chunks(2) {
            let low = (pair[0] & 0x0F) as u8;
            let high = if pair.len() > 1 {
                (pair[1] & 0x0F) as u8
            } else {
                0u8
            };
            packed.push(low | (high << 4));
        }
    }

    (packed, scales)
}

fn compress_zlib(data: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

fn main() {
    let args = Args::parse();

    // Read raw weights
    println!("Reading weights from {:?}...", args.input);
    let raw_bytes = fs::read(&args.input).expect("Failed to read input file");
    let weights: Vec<f32> = raw_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    println!(
        "Loaded {} parameters ({} bytes)",
        weights.len(),
        raw_bytes.len()
    );

    // Quantize
    match args.bits {
        8 => {
            let (quantized, scales) = quantize_int8(&weights);
            let quant_bytes: Vec<u8> = quantized.iter().map(|&x| x as u8).collect();
            let scale_bytes: Vec<u8> = scales
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();

            println!("Int8 quantized: {} bytes weights + {} bytes scales",
                quant_bytes.len(), scale_bytes.len());

            // Combine and compress
            let mut combined = quant_bytes;
            combined.extend_from_slice(&scale_bytes);
            
            println!("Combined: {} bytes", combined.len());
            
            // Note: Would use flate2 crate for zlib, but keeping deps minimal
            // For now, just report sizes
            println!("\n=== Size Analysis ===");
            println!("Original f32:     {} bytes", raw_bytes.len());
            println!("Int8 + scales:    {} bytes ({:.1}% of original)",
                combined.len(),
                combined.len() as f64 / raw_bytes.len() as f64 * 100.0);
            println!("Target:           {} bytes", args.target_bytes);
            println!("Fits in target:   {}",
                if combined.len() <= args.target_bytes { "YES ✓" } else { "NO ✗" });

            if !args.analyze {
                fs::write(&args.output, &combined).expect("Failed to write output");
                println!("\nSaved to {:?}", args.output);
            }
        }
        4 => {
            let (packed, scales) = quantize_int4(&weights);
            let scale_bytes: Vec<u8> = scales
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();

            println!("Int4 packed: {} bytes weights + {} bytes scales",
                packed.len(), scale_bytes.len());

            let mut combined = packed;
            combined.extend_from_slice(&scale_bytes);

            println!("Combined: {} bytes", combined.len());
            println!("\n=== Size Analysis ===");
            println!("Original f32:     {} bytes", raw_bytes.len());
            println!("Int4 + scales:    {} bytes ({:.1}% of original)",
                combined.len(),
                combined.len() as f64 / raw_bytes.len() as f64 * 100.0);
            println!("Target:           {} bytes", args.target_bytes);
            println!("Fits in target:   {}",
                if combined.len() <= args.target_bytes { "YES ✓" } else { "NO ✗" });

            if !args.analyze {
                fs::write(&args.output, &combined).expect("Failed to write output");
                println!("\nSaved to {:?}", args.output);
            }
        }
        _ => {
            eprintln!("Unsupported bit width: {}. Use 8 or 4.", args.bits);
            std::process::exit(1);
        }
    }
}
