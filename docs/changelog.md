# SER Capstone — Model Improvement Changelog

## Starting Baseline (Capstone v1 / v2)
- Two previous attempts, both with bugs
- Actor leakage between train/val/test splits
- MFCC per-row normalization missing → 200× scale difference between rows
- Scalar features computed on padded audio → contaminated by silence padding
- Result: unreliable, buggy

---

## Session 1 — Full Rewrite (64.2% ensemble)

### Architecture decisions
| Model | Architecture | Accuracy |
|-------|-------------|----------|
| Model 1 | SVM + Random Forest on 260-dim scalar features | 54.3% |
| Model 4 | EfficientNet-B0, 3-channel (mel+chroma+spectral contrast), two-phase training | 58.1% |
| Model 5 | ResNet-18 dual-input (mel branch + raw waveform 1D-CNN branch) | 62.1% |
| Ensemble (3 models) | Weighted average, fixed config weights | **64.2%** |

### Key bug fixes from v1/v2
- **Speaker-independent splits**: actors shuffled per dataset then split 70/15/15. Post-split assertion crashes loudly if any actor appears in 2 splits.
- **MFCC per-row normalization**: each of 40 rows × 3 channels normalized to mean=0, std=1. Previously rows differed by 200×.
- **Scalar features on unpadded audio**: `y_orig` passed to `compute_scalar()` before `center_pad()`. Prevents silence padding inflating ZCR/RMS/pitch.
- **Center-padding**: symmetric padding to exactly 88,200 samples (4s @ 22,050 Hz).
- **num_workers=0**: enforced everywhere. Windows PyTorch multiprocessing requires this.
- **AMP training**: `autocast + GradScaler` for all PyTorch models.
- **LabelSmoothingCrossEntropy (smoothing=0.1)** + class weights to handle neutral class imbalance.

### Infrastructure built
- `config.py` — single source of truth for all parameters
- `data/extract_features.py` — 6 feature types extracted offline
- `data/build_manifest.py` — speaker-aware splits with leakage assertion
- `data/augment.py` — offline augmentation: Gaussian noise + pitch shift ±1 semitone → 3× train
- `data/dataset.py` — SERDataset with online SpecAugment (time + freq masking)
- `training/train_utils.py` — shared AMP loop, EarlyStopping, TrainingHistory
- `evaluation/evaluate.py` — confusion matrices, classification reports, training curves, comparison table
- `models/ensemble.py` — auto-detects available checkpoints, normalizes weights

---

## Session 2 — Architecture Fixes + New Models (68% ensemble)

### Problem diagnosis
- Model 5 (ResNet-18 dual-input) was **53%** — worse than before
  - Root cause: waveform 1D-CNN branch caused repeated gradient explosions (val_loss spiked to 2.50 at epoch 8). The waveform branch and mel branch share gradients, so waveform explosions corrupted mel learning.
  - Fix: **removed the waveform branch entirely**. The old 72% model was mel-only.
- Model 5 was only doing `AdaptiveAvgPool2d` — collapsing all temporal info into one vector
  - Fix: added **BiLSTM on ResNet layer4 output**. Mean-pool over freq axis → 7 time steps → BiLSTM(512→128bi) → 256-dim embedding.

### New models implemented
| Model | Architecture | Notes |
|-------|-------------|-------|
| Model 2 | CNN+BiLSTM+Attention on mel (128×173) | 4 CNN blocks (freq-only pool) → BiLSTM(256→128bi) → self-attention |
| Model 3 | CNN+BiLSTM+Attention on MFCC stack (40×173×3) | 3 CNN blocks → BiLSTM(128→64bi) → self-attention |
| Model 6 | Fusion: parallel MFCC branch + Chroma branch | MFCC→128-dim, Chroma→64-dim, concat→192→classifier |

### Regularization improvements
| Change | Files | Why |
|--------|-------|-----|
| `clip_grad_norm_(1.0)` | `train_utils.py` | Kills gradient explosions seen in ResNet-18 waveform branch |
| `weight_decay=1e-4` | All training scripts | L2 regularization reduces train/val gap |
| Double Dropout (before + after FC) | `resnet18_dual.py` | Prevents over-reliance on any single feature dimension |
| `AdamW` instead of `Adam` | `train_cnn_bilstm.py`, `train_fusion.py` | AdamW applies weight decay correctly (decoupled from gradient) |

### EfficientNet fixes
| Change | Value | Why |
|--------|-------|-----|
| `unfreeze_all()` instead of top-3 blocks | All 9 MBConv blocks | Mel spectrograms differ enough from ImageNet that deeper layers need to adapt |
| Phase 2 LR | 1e-4 → 5e-5 | Prevents destroying pretrained weights now that all layers are unfrozen |
| Phase 2 epochs | 40 → 60 | More time for full backbone to adapt |
| Optimizer | Adam → AdamW | Correct weight decay |
| ReduceLROnPlateau patience | 5 → 4 | Faster LR reduction |

### Augmentation expansion
| Added | Description | Train size |
|-------|-------------|-----------|
| Time stretch (rate 0.9 or 1.1) | Slows/speeds audio without changing pitch | +1 variant |
| Volume perturbation (gain 0.7 or 1.3) | Quiet/loud version | +1 variant |
| Total | Original + noise + pitch + stretch + volume | **5× train data** (was 3×) |

### Ensemble improvements
- **Val-optimized weights**: `scipy.optimize.minimize` (SLSQP) finds optimal weights on validation set before test evaluation. Replaces fixed config weights.
- **All 6 models** now auto-detected and included
- Fixed stale class name bugs (`CNNBiLSTMMel` → `CNNBiLSTMMelSER`, etc.)
- Fixed `torch.load(weights_only=True)` warning

### Bug fixes
- `torch.cuda.amp.GradScaler` → `torch.amp.GradScaler` (deprecated)
- `torch.cuda.amp.autocast` → `torch.amp.autocast(_DEVICE_TYPE)` (deprecated)
- `GradScaler` not imported in `train_utils.py` → added `from torch.amp import GradScaler`
- Stale comment in `resnet18_dual.py`: said `(B, 768)` but concat was 256+256=512

### Result
| Model | Accuracy |
|-------|---------|
| Model 1: Traditional ML | ~54% |
| Model 2: CNN+BiLSTM (mel) | ~65% |
| Model 3: CNN+BiLSTM (MFCC) | ~60% |
| Model 4: EfficientNet-B0 | ~58% |
| Model 5: ResNet-18+BiLSTM | ~62% |
| Model 6: Fusion | ~58% |
| **Ensemble (val-optimized weights + 6 models)** | **68%** |

---

## Session 3 — Model Refinement and Optimization (70.7% best single model)

### Completed Changes
- **Removed Model 3 (CNN+BiLSTM MFCC)**: Accuracy 58.97% < 60% threshold. Deleted `models/cnn_bilstm_mfcc.py`, `outputs/model3_cnn_bilstm_mfcc/`, updated `run_pipeline.py` to skip train3, modified `train_cnn_bilstm.py` to only handle Mel variant.
- **Model 8 (Multi-Feature CNN+BiLSTM) completed**: Combines all 267 features (mel 128 + MFCC 120 + chroma 12 + spectral 7). Achieved 70.73% accuracy, best single model so far.
- **Model 9 (Wav2Vec2 fine-tuning) in progress**: Using safetensors loading to bypass torch 2.6 requirement. Fine-tuning top 6 transformer layers on 16kHz resampled audio.
- **GPU utilization verified**: All models use CUDA with AMP, pin_memory, optimized DataLoader (num_workers=0 for Windows).
- **Speed optimizations**: No bottlenecks found; training efficient with batch_size=32, AMP, early stopping.

### Planned Improvements
- **Hyperparameter tuning with Optuna**: Integrate Bayesian optimization for LR, dropout, batch_size, etc. Expected +2-5% accuracy boost. Will modify training scripts to define search spaces and run 50-100 trials.
- **State-of-the-art practices audit**: Review for self-attention, focal loss, larger pre-trained models, UAR metrics. Update models accordingly for max accuracy.
- **Ensemble update**: Include Model 8 (and Model 9 when complete) in weighted ensemble for >70% overall.

### Hyperparameter Tuning with Optuna
To push accuracy beyond current levels, we'll integrate Optuna for automated hyperparameter optimization. Optuna uses Bayesian optimization to efficiently search hyperparameter spaces, pruning poor configurations early.

**How it works**:
- Define search space: e.g., learning_rate (1e-5 to 1e-2), dropout (0.1-0.5), batch_size (16-64), weight_decay (1e-5 to 1e-3).
- Objective function: Train model for fixed epochs, return validation accuracy.
- Run 50-100 trials; Optuna suggests next params based on past results.
- Best config applied to full training.

**Expected benefits**: +2-5% accuracy (e.g., 68% → 70-72%) by fine-tuning model-specific params like attention heads or LSTM units. Reduces manual tuning time from days to hours.

**Integration plan**: Modify `train_cnn_bilstm.py` etc. to wrap training in Optuna study. Run overnight on GPU.

### Result so far
| Model | Accuracy | Status |
|-------|----------|--------|
| Model 1: Traditional ML | 52.6% | Kept (per user request) |
| Model 2: CNN+BiLSTM (Mel) | 68.2% | Active |
| Model 4: EfficientNet-B0 | 68.3% | Active |
| Model 5: ResNet-18+BiLSTM | 61.2% | Active |
| Model 6: Fusion | 61.5% | Active |
| Model 7: Ensemble | 70.1% | Active (needs update for Model 8) |
| Model 8: Multi-Feature | 70.7% | Completed |
| Model 9: Wav2Vec2 | TBD | In progress |

---

### 1. SE (Squeeze-Excitation) blocks in all CNN models
**Files**: `models/cnn_bilstm_mel.py` (shared `_cnn_block` used by Models 2, 3, 6)

Each CNN block now applies channel attention after the conv:
```
Conv2d → BN → ReLU → SE(GlobalAvgPool → FC(C→C//16) → ReLU → FC→Sigmoid) → MaxPool
```
SE learns to upweight the frequency channels that carry the most emotion information and suppress uninformative ones. This adds ~1-2% accuracy with minimal parameter overhead (reduction ratio 16).

### 2. Mixup data augmentation (online, training loop only)
**Files**: `training/train_utils.py`

Applied in `train_one_epoch` via `use_mixup=True` (default):
```
mel_mix = λ·mel_a + (1-λ)·mel_b     λ ~ Beta(0.4, 0.4)
loss    = λ·loss(pred, label_a) + (1-λ)·loss(pred, label_b)
```
Forces smooth decision boundaries between emotion classes. Particularly effective for SER because emotion is a continuous spectrum — blending "angry" + "fearful" represents plausible real recordings. Expected gain: +2-4%.

### 3. Cosine Annealing LR schedule (replaces ReduceLROnPlateau)
**Files**: `train_cnn_bilstm.py`, `train_fusion.py`, `train_resnet18.py`

`CosineAnnealingLR(T_max=MAX_EPOCHS, eta_min=1e-6)` decays LR smoothly from 1e-3 to 1e-6 following a cosine curve. Called unconditionally every epoch (not waiting for val_loss plateau).

Why better than ReduceLROnPlateau:
- Reactive schedulers wait for failure before reducing LR — often too late
- Cosine annealing consistently drives models into sharper minima
- No patience hyperparameter to tune
- Early stopping patience raised to 15 to give cosine schedule room to recover from local plateaus

### 4. Test-Time Augmentation (TTA) in ensemble — no retraining needed
**Files**: `models/ensemble.py`

During ensemble inference, each test sample is run through each PyTorch model **3 times**:
- View 1: original mel
- View 2: center time-masked (15 frames zeroed)
- View 3: center freq-masked (10 bins zeroed)

Probabilities averaged across 3 views before weighting. This costs 3× inference time (still seconds, not hours) and typically adds +1-2% with zero retraining.

---

## Expected Results After Session 3 Retrain

| Component | Expected contribution |
|-----------|----------------------|
| SE blocks | +1-2% per model |
| Mixup | +2-4% on CNN+BiLSTM models |
| Cosine annealing | +2-3% (smoother convergence) |
| TTA | +1-2% (free, no retrain) |
| **Cumulative ensemble** | **Target: 73-78%** |

---

## Run Order (after Session 3 changes)
```bash
py run_pipeline.py --from train2
```

---

## Session 4 — Actual Results + Tier 1 Fixes

### Results from Session 3 retrain

| Model | Best Val Acc | Final Test Acc | Gap |
|-------|-------------|---------------|-----|
| Model 2: CNN+BiLSTM (Mel) | 70.36% (epoch 32) | 68.22% | -2.14% |
| Model 3: CNN+BiLSTM (MFCC) | 65.02% (epoch 27) | 58.97% | **-6.05%** |
| Model 4: EfficientNet-B0 | 70.06% (epoch 21) | 68.31% | -1.75% |
| Model 5: ResNet-18+BiLSTM | 63.81% (epoch 10) | 61.22% | -2.59% |
| Model 6: MFCC+Chroma Fusion | 65.22% (epoch 25) | 61.47% | -3.75% |
| **Model 7: Ensemble** | **71.27%** | **70.06%** | -1.21% |

### Problem diagnosis from results

**Model 3 (MFCC) — 32-point train/test gap (91% train, 59% test)**
Root cause: BiLSTM with hidden=64 has too much capacity for 40 MFCC coefficients on a
speaker-independent split. Standard `nn.Dropout` zeros random scalar values but preserves
temporal patterns — the model memorizes speaker-specific vocal characteristics that
happen to correlate with emotion in the training set but don't generalize.

**Model 5 (ResNet-18) — volatile, stopped at epoch 22**
Root cause: `CosineAnnealingLR(T_max=100)` with early stopping at epoch 22 means
the LR was still at ~5e-4 when training stopped — the model never reached the
low-LR fine-tuning phase that produces sharp minima.

### Changes made (Tier 1 fixes)

#### 1. Spatial Dropout on BiLSTM input — Models 2 and 3
**Files**: `models/cnn_bilstm_mfcc.py`, `models/cnn_bilstm_mel.py`

Added `nn.Dropout1d(p=0.2)` applied to the `(B, T, F)` tensor before it enters BiLSTM.

`Dropout1d` zeros entire time steps (all features at time t become 0) rather than
random individual values. This directly forces the model to not rely on any single
frame — eliminating temporal memorization of speaker-specific patterns.

Literature source: MS-SENet paper used spatial dropout and reported +3.72% WA across
6 corpora compared to standard dropout.

#### 2. Reduced BiLSTM hidden size in Model 3
**File**: `models/cnn_bilstm_mfcc.py`

Reduced hidden units: 64 → 32 per direction (128 → 64 total output dim).
Rationale: MFCC has only 40 frequency bins — a hidden size of 64 gives the model
more representational capacity than the input dimension warrants, enabling memorization.
32 units is still sufficient to capture temporal emotion dynamics in 40-coefficient space.

Also updated `SelfAttention(hidden_dim=128→64)` and `Linear(128→64)` in classifier
to match the new output dimension.

#### 3. CosineAnnealingWarmRestarts for ResNet-18
**File**: `training/train_resnet18.py`

Replaced `CosineAnnealingLR(T_max=100)` with:
```python
CosineAnnealingWarmRestarts(T_0=20, T_mult=2, eta_min=1e-6)
```
With warm restarts, each 20-epoch cycle completes a full cosine sweep from 1e-3 to
1e-6 regardless of when early stopping fires. The first restart happens at epoch 20,
the second at epoch 60. Early stopping at epoch 22 now captures a model that has
completed one full cosine cycle rather than being cut off mid-decay.

### Expected impact

| Fix | Target model | Expected gain |
|-----|-------------|--------------|
| Spatial Dropout (Dropout1d) | Model 3 | Close 6% gap → ~+4% test acc |
| BiLSTM hidden 64→32 | Model 3 | Reduce memorization capacity |
| Spatial Dropout | Model 2 | +1-2% generalization |
| CosineAnnealingWarmRestarts | Model 5 | +2-3% (full LR cycle before stopping) |
| **Ensemble** | All | **Target: 72-75%** |

### Run order
```bash
py run_pipeline.py --from train2
```

---

## Session 5 — Path A + Path B Implementation (target: 78-82%)

### New models added

#### Model 8: Multi-Feature CNN+BiLSTM (Path B)
**Files**: `models/multifeature_cnn_bilstm.py`, `training/train_multifeature.py`
**Output**: `outputs/model8_multifeature/`

Stacks ALL per-frame features (mel 128 + MFCC+Δ+ΔΔ flat 120 + chroma 12 + SC 7 = 267 dims)
along the frequency axis, then runs through 5 CNN blocks → BiLSTM(hidden=256 bi) → attention.

Why: Models 2–6 each see only one or two feature types. Cross-feature patterns
(high pitch AND fast MFCC delta AND dissonant chroma = angry) can only be learned
when all features enter the same model simultaneously. Based on DCRF-BiLSTM paper
which achieves 97.83% on RAVDESS using this strategy.

Architecture changes vs mel model:
- 5 CNN blocks instead of 4 (267 → 8 vs 128 → 8)
- BiLSTM hidden=256 per direction (vs 128) → 512-dim output
- Same SE blocks, Spatial Dropout1d(0.2), SelfAttention reused from cnn_bilstm_mel.py
- CosineAnnealingLR(T_max=80), patience=15

#### Model 9: Wav2Vec 2.0 Fine-tuned (Path A)
**Files**: `models/wav2vec2_ser.py`, `training/train_wav2vec2.py`
**Output**: `outputs/model9_wav2vec2/`
**New dependency**: `pip install transformers`

Uses `facebook/wav2vec2-base` pretrained on 960h LibriSpeech. Fine-tunes top 6 of
12 transformer layers + classifier head. Bottom 6 layers + CNN feature extractor
frozen. Raw waveform input, resampled from 22050→16000 Hz in forward().

Training differences:
- batch_size=8 (not 32) — 94M param backbone
- lr=1e-4 (not 1e-3) — fine-tuning, not training from scratch
- use_mixup=False — waveform input doesn't use mel mixup
- MAX_EPOCHS=40 — pretrained model converges faster
- CosineAnnealingWarmRestarts(T_0=20)

### Config + infrastructure changes
| File | Change |
|------|--------|
| `config.py` | Added `model8_multifeature` and `model9_wav2vec2` to `ENSEMBLE_WEIGHTS` and `_MODEL_OUTPUT_DIRS` |
| `config.py` | Rebalanced weights: wav2vec2=0.25 (highest), multifeature=0.15, others reduced slightly |
| `models/ensemble.py` | Added `_load_pytorch_model` branches for `multifeature` and `wav2vec2` |
| `run_pipeline.py` | Added `step_train8()`, `step_train9()`, added to STEPS and FULL_ORDER |
| `run_pipeline.py` | Added `transformers` version check in `step_env()` |
| `requirements.txt` | Added `transformers>=4.35.0` |

### Expected impact

| Model | Expected test acc | Notes |
|-------|------------------|-------|
| Model 8: Multi-Feature CNN+BiLSTM | 73–76% | All features jointly |
| Model 9: Wav2Vec2 | 78–85% | Pretrained on 960h audio |
| Ensemble (8+9 + existing 6 models) | **78–82%** | SLSQP-optimized weights |

### Run order (after this session)
```bash
pip install transformers
py run_pipeline.py --from train2
```

This retrains models 2–6 (with Tier 1 fixes from Session 4), trains the two new
models (8, 9), then runs the ensemble. Total estimated time: 6–10 hours overnight.
