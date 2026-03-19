# Distillation Analysis for Parameter Golf

## The Question
Can we use knowledge distillation from a large teacher model to improve our 16MB student model's BPB on FineWeb?

## How It Would Work

### Approach 1: Offline Logit Distillation (Most Promising)
1. **Pre-compute step (offline, before the 10-min clock):**
   - Run a large teacher model (GPT-2 Large, Llama 3.2 1B, etc.) on FineWeb training data
   - Save the teacher's softmax logits (or top-K logits) for each token position
   - Store these alongside the training data shards

2. **Training step (within 10 min on 8xH100s):**
   - Load both ground-truth tokens AND teacher logits
   - Train with combined loss: `L = α * CE(student, ground_truth) + (1-α) * KL(student, teacher)`
   - The KL divergence term provides "soft labels" — teacher tells student not just WHAT the next token is, but the full probability distribution over all possible next tokens

### Approach 2: Data Curriculum (MiniPLM-style)
- Use teacher model to RANK training examples by difficulty
- Train student on examples where teacher-student gap is largest
- Essentially: smarter data selection, not logit matching

### Approach 3: Synthetic Data Augmentation
- Use teacher to generate additional training text
- Mix with original FineWeb data
- Student sees more diverse examples

## Key Rules Analysis

**Is distillation allowed?**
From the FAQ: "Tuning your Adam hyperparameters across a bunch of runs is fine, but if there's evidence that you're sneaking in additional compute unfairly, such as brute-forcing ridiculous seeds, we won't allow it."

Critical constraint: "No external downloads, training dataset access, or network calls are allowed during evaluation."

**But during TRAINING:** The rules say the 10-minute clock is for training. They explicitly allow:
- Custom tokenizers (with scrutiny)
- Test-time compute
- Any architecture

The question is: can the pre-computed teacher logits be part of the "training data" that's available during the run?

**My reading:** The training data shards are already downloaded from HuggingFace. If we pre-compute teacher logits as a PREPROCESSING step (like tokenization), and include them with the data shards, this is likely allowed. The key FAQ answer says they reserve the right to disqualify things "not in the spirit of the challenge." 

Distillation is a legitimate ML technique. The 16MB limit and 10-min limit still apply. We're not sneaking in compute — we're using a better training signal.

## Feasibility Assessment

### Storage Requirements
For logit distillation, we need to store teacher logits:
- FineWeb 10B tokens, vocab 1024 = 10B × 1024 × 2 bytes (float16) = 20 TB ❌ WAY TOO BIG
- Top-K logits (K=32): 10B × 32 × 4 bytes = 1.28 TB ❌ Still huge
- Top-K logits (K=8): 10B × 8 × 4 bytes = 320 GB — borderline
- Just teacher loss per token: 10B × 4 bytes = 40 GB — manageable but less useful

### Practical Approach: Selective Distillation
Instead of saving logits for ALL tokens, save for a SUBSET:
- Hardest tokens (where teacher loss is high = rare/interesting patterns)
- Or every Nth token
- Or just the top-K probabilities for each token

### Better Approach: Pre-train Teacher ONCE, Distill Into Student
1. Train a LARGER model (say 50M params) on FineWeb for hours (unlimited compute track)
2. Save that model's weights
3. During the 10-min run: load the 50M teacher, do online distillation into the 17M student
4. But wait — the teacher also has to fit in the environment!

### Best Approach: Self-Distillation / Progressive Distillation
1. Train baseline model for 4 hours (unlimited compute) → teacher
2. Use teacher logits as soft labels for 10-min student training
3. Student learns faster because the soft labels contain MORE information than hard one-hot labels

## Expected Improvement

From literature:
- **12% perplexity improvement** from logit distillation (Llama 3B → distilled 3B)
- **10-20% training speedup** in reaching same loss level (KDEP paper, CVPR 2022)
- **MiniPLM:** ~2-5% improvement on downstream tasks with data curriculum approach

For our case (17M param model):
- Baseline BPB: 1.2244
- If distillation gives 5% loss improvement: 1.2244 × 0.95 = 1.163 BPB 🎯
- If 2% improvement: 1.2244 × 0.98 = 1.200 BPB (still beats baseline!)
- Even 1%: 1.2244 × 0.99 = 1.212 BPB (beats baseline by 0.012)

## Implementation Plan

### Phase 1: Pre-compute Teacher Logits (Unlimited Compute)
```python
# Run on 8xH100 for several hours (doesn't count toward 10-min limit)
# Use the 4-hour baseline model (1.2074 BPB) as teacher
# OR use a public model like GPT-2 (124M params)

teacher = load_model("4hr_baseline.pt")  # or GPT-2
for shard in training_shards:
    tokens = load_shard(shard)
    with torch.no_grad():
        logits = teacher(tokens)
        # Save top-K logits per position
        topk_vals, topk_ids = logits.topk(K)
        save_teacher_data(shard_id, topk_ids, topk_vals)
```

### Phase 2: Distillation Training (10 min)
```python
# Modified training loop
for batch in data_loader:
    tokens = batch['tokens']
    teacher_topk = batch['teacher_logits']  # pre-computed
    
    student_logits = model(tokens)
    
    # Hard label loss (standard CE)
    ce_loss = F.cross_entropy(student_logits, tokens[1:])
    
    # Soft label loss (KL divergence with teacher)
    # Reconstruct sparse teacher distribution from top-K
    teacher_probs = sparse_softmax(teacher_topk)
    student_probs = F.softmax(student_logits / T, dim=-1)
    kl_loss = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')
    
    loss = alpha * ce_loss + (1 - alpha) * T^2 * kl_loss
```

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Disqualified for "unfair compute" | HIGH | Ask in Discord first; distillation is standard ML |
| Storage too large | MEDIUM | Use top-8 logits, ~320GB manageable |
| Marginal improvement | MEDIUM | Literature suggests 2-12% — even 2% wins |
| Teacher model mismatch | LOW | Use same tokenizer/vocab for teacher and student |
| Implementation complexity | LOW | Just add KL term to loss |

## Verdict

**Distillation is worth pursuing, but with caveats:**

1. **Best path:** Use the 4-hour baseline (1.2074 BPB) as teacher. It's already trained on FineWeb with the same tokenizer. Pre-compute its logits, then distill into our architecture.

2. **Ask in Discord first** whether pre-computed teacher logits are allowed as "preprocessing."

3. **If not allowed:** Data curriculum approach (rank examples by teacher-student gap) is almost certainly allowed, since it's just data ordering.

4. **Combine with architecture improvements:** Distillation + depth recurrence + custom tokenizer = potentially 0.05+ BPB improvement.

## Alternative: Auxiliary Model During Training
Instead of offline distillation, could train a SMALLER auxiliary model online:
- Train a tiny 1M-param model for the first 2 minutes
- Use it as teacher for remaining 8 minutes
- Self-distillation within the 10-min window
- Clearly within rules since everything happens during training

## Priority Ranking

1. **Architecture (depth recurrence + SwiGLU)** — highest confidence, most impact
2. **Offline logit distillation** — high potential but rule uncertainty
3. **Custom tokenizer** — moderate impact, rule scrutiny
4. **Data curriculum** — moderate impact, clearly allowed
5. **Online self-distillation** — lower impact but clearly within rules
6. **QAT** — incremental improvement at quantization time
