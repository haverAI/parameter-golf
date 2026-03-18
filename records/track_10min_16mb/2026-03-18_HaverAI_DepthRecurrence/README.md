# Depth Recurrence 7×2 — Haver AI

## Summary
- **7 physical transformer layers** looped **2 times** = **14 effective layers** (vs baseline's 9)
- Same architecture otherwise: dim=512, 8 heads, 4 KV heads, tied embeddings, ReLU² MLP
- U-Net skip connections applied within each recurrence pass
- Preserves baseline's Muon optimizer, learning rate schedule, and training recipe
- ~17M parameters, estimated ~99% of 16MB budget after int8+zlib

## Hypothesis
Depth recurrence is the highest-leverage single change: it increases effective depth at zero parameter cost. The baseline already uses a U-Net architecture with skip connections — looping through the same blocks twice effectively doubles the number of skip-connected processing steps while sharing all weights.

## Key Design Choices
1. **7 layers × 2 loops** over 6×3 or 4×4: Maximizes stored parameters (uses 99% of budget) while still getting 55% more effective depth than baseline.
2. **Same dim=512**: Wider dims (640) would exceed 16MB budget with enough physical layers.
3. **Skip connections per pass**: Each recurrence pass runs its own encoder/decoder skip pattern, providing gradient highways even in the second pass.

## Config
```
NUM_LAYERS=7
NUM_RECURRENCE=2  
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
VOCAB_SIZE=1024
TIE_EMBEDDINGS=1
```

## Status
- [x] Architecture implemented
- [x] Parameter count verified under 16MB
- [ ] Full 8xH100 training run (pending compute credits)
- [ ] BPB score validated

## Author
Will Haver — Haver AI Inc. (haverAI on GitHub)
