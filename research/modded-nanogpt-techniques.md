# Modded-NanoGPT Techniques Applicable to Parameter Golf

## Already in our baseline train_gpt.py
- [x] RoPE (Rotary Embeddings)
- [x] ReLU² activation
- [x] Muon optimizer
- [x] QK-Norm (via qk_gain_init)
- [x] U-Net skip connections (embedding to every block)
- [x] Tied embeddings
- [x] Logit softcap

## Techniques we've added
- [x] Depth recurrence (weight tying across loops)
- [x] SwiGLU MLP (alternative to ReLU²)
- [x] QAT noise injection

## NOT YET IMPLEMENTED — High Priority

### 1. Value Embeddings (Extra embeddings mixed into attention values)
- Zhou et al. 2024 inspired
- Add learnable value embeddings that get mixed into V in attention
- Low param cost, meaningful quality boost
- **Impact: Medium-High**

### 2. Batch Size Schedule
- Start with smaller batch size, increase during training
- Better gradient signal early, better throughput late
- Easy to implement: just change train_batch_tokens over time
- **Impact: Medium**

### 3. Multi-Token Prediction (Auxiliary Loss)
- Predict next 2-4 tokens simultaneously during training
- Auxiliary heads discarded at inference (don't count toward 16MB)
- Improves representation quality
- PR #11 in parameter-golf already uses this
- **Impact: Medium-High**

### 4. Sequence Length Schedule
- Start training with shorter sequences, increase over time
- Helps model learn local patterns first, then long-range
- Fits naturally with RoPE
- **Impact: Medium**

### 5. Untie Embed and LM Head at 2/3 of Training
- Train with tied embeddings for efficiency
- Untie them in the last 1/3 to specialize
- Extra params only at end, might not work with 16MB constraint
- **Impact: Low-Medium (param budget issue)**

### 6. Gradient Accumulation for Embed/Head
- Accumulate gradients for embedding + lm_head for 2 steps before updating
- Smooths noisy gradients on these large matrices
- Simple change
- **Impact: Low-Medium**

### 7. Bigram Hash Embedding  
- Learn bigram (pair) statistics as additional embedding
- Hash pairs to a small lookup table
- Very cheap, captures common word patterns
- **Impact: Low-Medium**

## Priority Implementation Order
1. Multi-token prediction (biggest quality win, no size cost)
2. Value embeddings (proven in modded-nanogpt)
3. Batch size schedule (easy, helps convergence)
4. Sequence length schedule (easy, helps convergence)
5. Gradient accumulation for embed (tiny code change)
