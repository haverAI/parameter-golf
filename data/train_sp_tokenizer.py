"""Train a SentencePiece BPE tokenizer with custom vocab size on FineWeb data.

Usage:
    python data/train_sp_tokenizer.py --vocab_size 2048 --input_docs data/docs_selected.jsonl
    python data/train_sp_tokenizer.py --vocab_size 4096 --input_docs data/docs_selected.jsonl

This trains a SentencePiece model compatible with the parameter-golf eval pipeline.
Output: data/tokenizers/fineweb_{vocab_size}_bpe.model
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import sentencepiece as spm


def extract_text_from_jsonl(jsonl_path: str, max_docs: int = 500_000, max_chars: int = 100_000_000) -> str:
    """Extract text from docs_selected.jsonl for tokenizer training."""
    texts = []
    total_chars = 0
    
    print(f"Reading from {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs or total_chars >= max_chars:
                break
            doc = json.loads(line)
            text = doc.get("text", "")
            if text:
                texts.append(text)
                total_chars += len(text)
    
    print(f"Extracted {len(texts)} docs, {total_chars:,} chars")
    return texts


def extract_text_from_bin_shards(data_dir: str, tokenizer_path: str, max_tokens: int = 50_000_000) -> list[str]:
    """Decode text from existing .bin shards using the baseline tokenizer."""
    import numpy as np
    
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    shard_files = sorted(Path(data_dir).glob("fineweb_train_*.bin"))
    
    texts = []
    total_tokens = 0
    
    for shard in shard_files:
        print(f"Reading shard {shard.name}...")
        with open(shard, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=256)
            tokens = np.fromfile(f, dtype=np.uint16)
        
        # Decode in chunks
        chunk_size = 10000
        for start in range(0, len(tokens), chunk_size):
            chunk = tokens[start:start + chunk_size].tolist()
            text = sp.decode(chunk)
            if text.strip():
                texts.append(text)
                total_tokens += len(chunk)
            
            if total_tokens >= max_tokens:
                break
        
        if total_tokens >= max_tokens:
            break
    
    print(f"Decoded {len(texts)} text chunks from {total_tokens:,} tokens")
    return texts


def train_tokenizer(texts: list[str], vocab_size: int, output_prefix: str):
    """Train a SentencePiece BPE tokenizer."""
    # Write texts to a temp file for SentencePiece training
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        tmp_path = f.name
        for text in texts:
            # SentencePiece expects one sentence per line
            for line in text.split("\n"):
                line = line.strip()
                if line:
                    f.write(line + "\n")
    
    print(f"Training SentencePiece BPE with vocab_size={vocab_size}...")
    print(f"Temp training file: {tmp_path}")
    
    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        # Match the baseline tokenizer settings
        character_coverage=1.0,
        byte_fallback=True,
        # Important: match the special tokens
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        # Training params
        num_threads=os.cpu_count() or 8,
        train_extremely_large_corpus=True,
        max_sentence_length=16384,
        # Normalization - minimal to preserve web text structure
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        add_dummy_prefix=True,
    )
    
    # Clean up temp file
    os.unlink(tmp_path)
    
    # Verify
    sp = spm.SentencePieceProcessor(model_file=f"{output_prefix}.model")
    print(f"Trained tokenizer: vocab_size={sp.vocab_size()}")
    
    # Test compression ratio
    test_text = "The quick brown fox jumps over the lazy dog. Machine learning is a subset of artificial intelligence."
    tokens = sp.encode(test_text)
    print(f"Test: '{test_text[:60]}...'")
    print(f"  Tokens: {len(tokens)}, Bytes: {len(test_text.encode('utf-8'))}, Bytes/token: {len(test_text.encode('utf-8'))/len(tokens):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer for Parameter Golf")
    parser.add_argument("--vocab_size", type=int, required=True, help="Tokenizer vocab size")
    parser.add_argument("--input_docs", type=str, default=None, help="Path to docs_selected.jsonl")
    parser.add_argument("--input_shards", type=str, default=None, help="Path to tokenized shard directory")
    parser.add_argument("--baseline_tokenizer", type=str, default="./data/tokenizers/fineweb_1024_bpe.model",
                       help="Baseline tokenizer for decoding shards")
    parser.add_argument("--max_docs", type=int, default=500_000, help="Max docs to use for training")
    parser.add_argument("--max_chars", type=int, default=100_000_000, help="Max chars for training")
    parser.add_argument("--output_dir", type=str, default="./data/tokenizers", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / f"fineweb_{args.vocab_size}_bpe")
    
    if args.input_docs and Path(args.input_docs).exists():
        texts = extract_text_from_jsonl(args.input_docs, args.max_docs, args.max_chars)
    elif args.input_shards and Path(args.input_shards).exists():
        texts = extract_text_from_bin_shards(args.input_shards, args.baseline_tokenizer)
    else:
        # Try to download docs
        print("No input specified. Attempting to download docs_selected.jsonl from HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(
                repo_id="willdepueoai/parameter-golf",
                filename="datasets/docs_selected.jsonl",
                repo_type="dataset",
            )
            texts = extract_text_from_jsonl(local, args.max_docs, args.max_chars)
        except Exception as e:
            print(f"Failed to download: {e}")
            print("Please provide --input_docs or --input_shards")
            sys.exit(1)
    
    train_tokenizer(texts, args.vocab_size, output_prefix)
    print(f"\nOutput: {output_prefix}.model")
    print(f"Next step: retokenize data with this tokenizer:")
    print(f"  python data/download_hf_docs_and_tokenize.py --vocab_size {args.vocab_size}")


if __name__ == "__main__":
    main()
