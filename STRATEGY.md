# Parameter Golf — Haver AI Strategy Document
## Target: Beat 1.2244 BPB → Sub-1.20 BPB

### Challenge Summary
- **Metric:** bits per byte (BPB) on FineWeb validation set (tokenizer-agnostic)
- **Constraint:** 16MB artifact (compressed model + code), 10min on 8xH100s
- **Baseline:** 1.2244 BPB (9-layer, 512-dim, 1024 vocab, tied embeddings, 4 KV heads)
- **Non-record baseline:** 1.2074 BPB (same arch, 4hrs unlimited compute → 329K steps)
- **Deadline:** April 30, 2026
- **Current rank:** No submission yet

### Key Insight
The gap between 10min (1.2244) and 4hr (1.2074) is only 0.017 BPB. This means:
1. Architecture improvements could close that gap within 10min
2. The same architecture with better training recipe could close it
3. Combining both = leaderboard contender

### Competitive Landscape (PRs as of Mar 18)
| PR | Author | Approach | Status |
|----|--------|----------|--------|
| #21 | monroestephenson | Depth recurrence (8×2=16 layers) + SwiGLU + QAT noise + dim=640 | Pending compute |
| #11 | adityawrk | Looped transformer (3×5=15 layers) + LoRA adapters + U-Net skips | WIP |
| #5 | albertorkive | Sparse attention + recursive weight sharing | WIP |
| #15 | ArthurKaroyan | Recursive weight sharing | WIP |
| #8 | iranzithierry | Depth recurrence + SwiGLU (M3 8GB) | Submitted |

**Nobody has beaten the baseline yet.** Everyone is still waiting for compute.

### Our Strategy

#### Tier 1: Architecture (Highest Impact)
1. **Depth Recurrence** — 6 physical layers × 3 loops = 18 effective layers
   - Why: Nearly doubles effective depth at zero parameter cost
   - Risk: Diminishing returns past ~3 loops
   - Implementation: Loop `self.blocks` multiple times in forward()
   
2. **SwiGLU MLP** — Replace ReLU² with SwiGLU
   - Why: Better quality per parameter, proven in modern LLMs
   - Risk: Slightly more params per MLP block
   
3. **Wider Model** — dim=640-768 (vs 512)
   - Why: More capacity per layer
   - Risk: Must stay under 16MB after int8+zlib

#### Tier 2: Training Recipe
4. **Learning Rate Schedule** — Cosine with warmup + cooldown
   - Current: Linear warmup + linear decay
   - Better: Cosine annealing often improves final loss
   
5. **Batch Size Annealing** — Start small, increase during training
   - Why: Better gradient signal early, better throughput late
   
6. **Longer Warmup** — May help with deeper (recurrent) models

#### Tier 3: Compression
7. **Quantization-Aware Training (QAT)**
   - Inject int8 quantization noise during last 30% of training
   - Reduces post-quant BPB degradation
   - PR #21 claims 0.0076 BPB delta on random weights
   
8. **Int4 Quantization** — If we can train with QAT noise at int4
   - Would halve model size → could fit 2× more parameters
   - Risk: Quality degradation may exceed savings

#### Tier 4: Moonshot Ideas
9. **Custom Tokenizer** — Larger vocab (2048-4096) trained on FineWeb
   - Why: Better BPE merges = fewer tokens = less to predict
   - Risk: Must prove val_bpb is correctly calculated (scrutiny)
   
10. **Test-Time Compute** — Iterative refinement during eval
    - Explicitly allowed by rules!
    - Could use depth recurrence at eval time with more loops
    
11. **Multi-Token Prediction** — Auxiliary training objective
    - PR #11 uses this (training only, excluded from artifact)
    - May improve representation quality

### Execution Plan

**Phase 1: Local Prototyping (Now → RunPod credits arrive)**
- [x] Fork repo, clone locally
- [x] Apply for $500 RunPod dev grant
- [x] Submit participant form
- [ ] Implement depth recurrence locally (modify train_gpt.py)
- [ ] Implement SwiGLU MLP replacement
- [ ] Test int8+zlib compression to verify <16MB

**Phase 2: Single GPU Experiments (Credits arrive → +3 days)**
- [ ] Run baseline on 1xH100 to reproduce 1.2244
- [ ] Test depth recurrence: 6×2, 6×3, 8×2 configurations
- [ ] Test SwiGLU vs ReLU²
- [ ] Test dim=640 vs 512
- [ ] Hyperparameter sweep: LR, batch size, warmup steps

**Phase 3: 8xH100 Competition Runs (Phase 2 + 3 days)**
- [ ] Combine best architecture from Phase 2
- [ ] Full 8xH100 10-minute runs
- [ ] QAT noise injection experiments
- [ ] Submit PR when we beat baseline by >0.005 nats

### Budget
- $500 RunPod dev grant (applied, pending 1-2 days)
- Can request $1000 advanced grant once near top of leaderboard
- Local 2080 Ti for code prototyping (not for actual training)

### Critical Insights (Updated Mar 18 evening)

**Tokenizer constraint:** The eval script is HARDCODED to SentencePiece `.model` files.
Our Rust BPE tokenizer outputs JSON — not directly compatible. To use a custom tokenizer:
1. Must train a SentencePiece model (`sentencepiece.SentencePieceTrainer.train()`)
2. Must retokenize the entire dataset into new `.bin` shards
3. Must update `tokenizer_specs.json` and run `download_hf_docs_and_tokenize.py`

**Tokenizer math:** Our Rust tokenizer showed:
- 1024 vocab: 2.52 bytes/token
- 2048 vocab: 3.01 bytes/token (+19.4%)
- 4096 vocab: 3.54 bytes/token (+40.5%)

But bigger vocab = bigger embedding table = more params. Trade-off:
- 1024 vocab × 512 dim = 524K embedding params
- 2048 vocab × 512 dim = 1.05M embedding params (+524K)
- 4096 vocab × 512 dim = 2.10M embedding params (+1.57M)

With tied embeddings, the extra params go into better compression.
**4096 vocab might NOT help** because the extra embedding params eat budget that could go to deeper/wider layers.
**2048 vocab is the sweet spot** — modest param increase for 19% better compression.

**SentencePiece tokenizer training script created:** `data/train_sp_tokenizer.py`
Ready to train on H100s once we download the raw docs.

**Everyone is doing the same thing.** All 13 PRs converge on depth recurrence + SwiGLU + QAT.
Our edge must come from:
1. Better hyperparameter tuning (more systematic sweeps)
2. Custom tokenizer (unique approach)
3. Combination of multiple improvements that interact well

### Key Numbers to Track
- **Baseline BPB:** 1.2244
- **Target BPB:** <1.20 (competitive), <1.18 (exceptional)
- **Beat threshold:** Must improve by ≥0.005 nats for SOTA claim
- **Model size budget:** 16,000,000 bytes (code + compressed model)
- **Baseline model bytes:** 15,863,489 (99.1% of budget)
- **Baseline code bytes:** 47,642
- **Headroom:** 136,511 bytes (~133KB for more params)
