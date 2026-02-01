# microWakeWord Trainer & Recorder - AI Agent Instructions

## Project Overview
Docker-based wake word training system combining a **FastAPI web recorder** with **TensorFlow/microWakeWord ML pipeline**. Trains quantized TFLite models for embedded wake word detection (ESPHome compatible).

## Architecture

### Two-Process System
1. **Recorder Server** (`recorder_server.py` + `run_recorder.sh`)
   - FastAPI app on port 2704 serving web UI for voice recording
   - Separate Python venv at `/data/.recorder-venv` (FastAPI, uvicorn)
   - Spawns training subprocess via `train_wake_word` CLI
   - Streams training logs to UI via `/api/training/log-tail`

2. **Training Pipeline** (`train_wake_word` + `cli/*` scripts)
   - Separate Python venv at `/data/.venv` (TensorFlow, librosa, scikit-learn)
   - Three-stage pipeline: **generate → augment → train**
   - All work happens in `/data/work/`, outputs to `/data/output/`

### Critical Data Flow
```
/data/personal_samples/*.wav  →  augmenter  →  /data/work/personal_augmented_features/
/data/work/wake_word_samples/ →  augmenter  →  /data/work/wake_word_samples_augmented/
                                                         ↓
                                              wake_word_sample_trainer
                                                         ↓
                                              /data/output/<timestamp>-<id>/*.tflite
```

## Key Conventions

### Language Detection & Transliteration
- **Auto-detect**: Cyrillic regex `[А-Яа-яЁё]` → `ru`, else `en`
- **Russian TTS**: Uses Piper voice `ru_RU-dmitri-medium.onnx`
- **Safe ID generation**: Cyrillic → ASCII via `_RU_TRANSLIT` dict (е.g., "привет дом" → "privet_dom")
- See `recorder_server.py:detect_language()` and `train_wake_word:130-165`

### Dual Venv Strategy
**Never mix dependencies!**
- Recorder venv: pinned FastAPI/uvicorn versions (see `run_recorder.sh:17-19`)
- Training venv: TensorFlow + ML stack (see `requirements.txt`)
- Activation: `source /data/.venv/bin/activate` for training, `/data/.recorder-venv/bin/activate` for recorder

### Personal Sample Weighting
- If `/data/personal_samples/*.wav` exists → augmenter creates `/data/work/personal_augmented_features/training/`
- Trainer detects this and injects YAML block with `sampling_weight: 3.0` (vs 2.0 for TTS)
- See `cli/wake_word_sample_trainer:95-105` and `cli/wake_word_sample_trainer:175-190`

### TensorFlow Environment
**Hard-coded in `train_wake_word:195-202`** to avoid XLA/CUDA crashes:
```bash
export TF_CPP_MIN_LOG_LEVEL=9
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"  # DO NOT append user flags
unset XLA_FLAGS
```

### CLI Argument Parsing
All `cli/*` scripts use `shell.functions` for unified arg parsing:
- `--key=value` → sets `${KEY}` variable (dashes → underscores, uppercase)
- `--no-key` → sets `${KEY}=false`
- Positional args → `${POSITIONAL_ARGS[@]}`
- Unknown args → `${UNKNOWN_ARGS[@]}` (triggers error if non-empty)

## Developer Workflows

### Testing Training Locally (without Docker)
```bash
# Setup venv once
cli/setup_python_venv --data-dir=/data

# Download datasets once (slow!)
cli/setup_training_datasets --data-dir=/data

# Train a wake word
train_wake_word --phrase "hey jarvis" --lang en --samples 50000 --training-steps 40000
```

### Adding New Language Support
1. Update `detect_language()` regex in `recorder_server.py`
2. Add transliteration dict to `_RU_TRANSLIT` (if non-Latin)
3. Update `cli/wake_word_sample_generator:85-95` with Piper model URL
4. Test with `train_wake_word --phrase "<phrase>" --lang <code>`

### Debugging Training Failures
- **Check log**: `/data/recorder_training.log` (recorder) or `/data/output/<timestamp>/logs/training.log` (CLI)
- **GPU OOM**: Trainer auto-retries with `CUDA_VISIBLE_DEVICES=""` (CPU fallback) on GPU errors
- **Missing datasets**: Run `cli/setup_training_datasets` manually
- **XLA errors**: Verify `TF_XLA_FLAGS` is NOT overridden by user environment

### Modifying Training Parameters
Edit YAML template in `cli/wake_word_sample_trainer:105-155`:
- `batch_size`: 16 (reduce if GPU OOM)
- `sampling_weight`: Controls dataset balance (wake:2.0, personal:3.0, negatives:5-12)
- `training_steps`: Default 40000 (more = better accuracy, longer training)

## Integration Points

### FastAPI Endpoints
- `POST /api/upload-sample` → Saves to `/data/personal_samples/speaker{N}_take{M}.wav`
- `POST /api/train` → Spawns `train_wake_word` subprocess, streams logs
- `GET /api/training/log-tail` → Returns last 400 lines of `recorder_training.log`

### Output Format (ESPHome)
Training produces:
- `<wake_word>.tflite` → Quantized streaming model
- `<wake_word>.json` → Metadata with `wake_word`, `authors`, `language`, `version`

### External Dependencies
- **Piper TTS models**: Auto-downloaded to `/data/tools/piper-sample-generator/models/`
- **Training datasets**: AudioSet, FMA, MIT RIRs (setup via `cli/setup_*` scripts)
- **microWakeWord**: Installed in training venv from `requirements.txt`

## Common Pitfalls
- **Don't delete `/data` between runs** → Re-downloads 10GB+ datasets
- **Don't modify `TF_XLA_FLAGS`** → Causes hard crashes on some CUDA versions
- **Don't use `python` directly** → Use venv-aware commands from `cli/` scripts
- **Don't assume English** → Always check `detect_language()` for Cyrillic input
