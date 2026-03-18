//! Fast BPE tokenizer trainer for Parameter Golf.
//! Trains a byte-level BPE tokenizer optimized for FineWeb distribution.
//! Outputs a sentencepiece-compatible .model file or a simple vocab+merges format.

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train-tokenizer", about = "Train BPE tokenizer for Parameter Golf")]
struct Args {
    /// Input text file(s) for training
    #[arg(short, long)]
    input: Vec<PathBuf>,

    /// Target vocabulary size
    #[arg(short, long, default_value = "1024")]
    vocab_size: usize,

    /// Output path for the trained tokenizer
    #[arg(short, long, default_value = "tokenizer.json")]
    output: PathBuf,

    /// Maximum lines to process (0 = all)
    #[arg(long, default_value = "0")]
    max_lines: usize,
}

/// A pair of token IDs that could be merged
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
struct Pair(u32, u32);

/// Core BPE trainer
struct BPETrainer {
    vocab_size: usize,
    vocab: Vec<Vec<u8>>,     // token_id -> bytes
    merges: Vec<(u32, u32)>, // ordered merge rules
}

impl BPETrainer {
    fn new(vocab_size: usize) -> Self {
        // Initialize with 256 byte-level tokens
        let mut vocab: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        // Ensure we have room for special tokens if needed
        assert!(vocab_size > 256, "vocab_size must be > 256");
        Self {
            vocab_size,
            vocab,
            merges: Vec::new(),
        }
    }

    /// Count all adjacent pairs in the corpus
    fn count_pairs(sequences: &[Vec<u32>]) -> HashMap<Pair, u64> {
        let chunk_size = (sequences.len() / rayon::current_num_threads().max(1)).max(1);
        
        sequences
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut counts: HashMap<Pair, u64> = HashMap::new();
                for seq in chunk {
                    for window in seq.windows(2) {
                        *counts.entry(Pair(window[0], window[1])).or_default() += 1;
                    }
                }
                counts
            })
            .reduce(HashMap::new, |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_default() += v;
                }
                a
            })
    }

    /// Apply a single merge to all sequences
    fn apply_merge(sequences: &mut [Vec<u32>], pair: Pair, new_id: u32) {
        sequences.par_iter_mut().for_each(|seq| {
            let mut i = 0;
            while i + 1 < seq.len() {
                if seq[i] == pair.0 && seq[i + 1] == pair.1 {
                    seq[i] = new_id;
                    seq.remove(i + 1);
                }
                i += 1;
            }
        });
    }

    /// Train the tokenizer on raw text
    fn train(&mut self, texts: &[String]) {
        let start = Instant::now();
        
        // Convert texts to byte-level token sequences
        println!("Converting {} texts to byte sequences...", texts.len());
        let mut sequences: Vec<Vec<u32>> = texts
            .par_iter()
            .map(|text| text.bytes().map(|b| b as u32).collect())
            .collect();

        let num_merges = self.vocab_size - 256;
        println!("Training {} merges...", num_merges);
        
        let pb = ProgressBar::new(num_merges as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:50} {pos}/{len} merges ({eta} remaining)")
                .unwrap(),
        );

        for merge_idx in 0..num_merges {
            // Count pairs
            let pair_counts = Self::count_pairs(&sequences);
            
            if pair_counts.is_empty() {
                println!("No more pairs to merge at step {}", merge_idx);
                break;
            }

            // Find most frequent pair
            let (best_pair, best_count) = pair_counts
                .iter()
                .max_by_key(|(_, count)| *count)
                .unwrap();

            let new_id = (256 + merge_idx) as u32;
            
            // Create new token by concatenating the pair's bytes
            let mut new_token = self.vocab[best_pair.0 as usize].clone();
            new_token.extend_from_slice(&self.vocab[best_pair.1 as usize]);
            self.vocab.push(new_token);
            self.merges.push((best_pair.0, best_pair.1));

            // Apply merge to all sequences
            Self::apply_merge(&mut sequences, *best_pair, new_id);

            if merge_idx % 50 == 0 || merge_idx == num_merges - 1 {
                let total_tokens: usize = sequences.iter().map(|s| s.len()).sum();
                pb.println(format!(
                    "  merge {}: ({}, {}) -> {} (count={}, corpus_tokens={})",
                    merge_idx, best_pair.0, best_pair.1, new_id, best_count, total_tokens
                ));
            }
            pb.inc(1);
        }
        pb.finish();

        let elapsed = start.elapsed();
        println!(
            "Training complete in {:.1}s. Vocab size: {}, Merges: {}",
            elapsed.as_secs_f64(),
            self.vocab.len(),
            self.merges.len()
        );

        // Print compression stats
        let original_bytes: usize = texts.iter().map(|t| t.len()).sum();
        let final_tokens: usize = sequences.iter().map(|s| s.len()).sum();
        println!(
            "Compression: {} bytes -> {} tokens ({:.2} bytes/token, {:.2}x compression)",
            original_bytes,
            final_tokens,
            original_bytes as f64 / final_tokens as f64,
            original_bytes as f64 / final_tokens as f64
        );
    }

    /// Save tokenizer as JSON (vocab + merges)
    fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        let mut file = fs::File::create(path)?;
        
        // Build JSON output
        let vocab_entries: Vec<serde_json::Value> = self.vocab.iter().enumerate().map(|(id, bytes)| {
            serde_json::json!({
                "id": id,
                "bytes": bytes,
                "text": String::from_utf8_lossy(bytes).to_string(),
            })
        }).collect();

        let merge_entries: Vec<serde_json::Value> = self.merges.iter().map(|(a, b)| {
            serde_json::json!([a, b])
        }).collect();

        let output = serde_json::json!({
            "vocab_size": self.vocab.len(),
            "vocab": vocab_entries,
            "merges": merge_entries,
        });

        writeln!(file, "{}", serde_json::to_string_pretty(&output)?)?;
        println!("Saved tokenizer to {:?}", path);
        Ok(())
    }
}

fn main() {
    let args = Args::parse();

    // Read input texts
    let mut texts = Vec::new();
    for input_path in &args.input {
        println!("Reading {:?}...", input_path);
        let file = fs::File::open(input_path).expect("Failed to open input file");
        let reader = BufReader::new(file);
        for (i, line) in reader.lines().enumerate() {
            if args.max_lines > 0 && i >= args.max_lines {
                break;
            }
            if let Ok(line) = line {
                if !line.is_empty() {
                    texts.push(line);
                }
            }
        }
    }
    println!("Loaded {} lines of text", texts.len());

    // Train
    let mut trainer = BPETrainer::new(args.vocab_size);
    trainer.train(&texts);

    // Save
    trainer.save(&args.output).expect("Failed to save tokenizer");
}
