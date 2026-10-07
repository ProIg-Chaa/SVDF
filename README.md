# SVDF — Sparse Visual Difference Feedback

SVDF briefly adjusts multimodal reasoning using an image-conditioned direction, then returns to ordinary decoding. This repository exports the **fixed-window input-feedback version** currently used by the project.

## Method

1. Feed back one raw probability-weighted soft embedding at step 0. From step 1, the default input feedback is the selected discrete token embedding.
2. During steps 2–9 (zero-based), capture at the first step whose raw entropy exceeds both the model threshold and the step-1 reference entropy. Skip visual feedback if no step qualifies.
3. Reuse the original-image live path and replay the same question and generated prefix once with a mean-RGB weakened image, using an independent cache.
4. With original soft embedding `s_I`, weakened soft embedding `s_W`, and selected hard embedding `h`, use `r = (s_I - s_W) * ||s_I - h|| / ||s_I - s_W||`. A negligible difference produces a zero update.
5. Add this direction to the hard feedback at the capture step and the next three steps. Each feedback affects the next model forward. After that, continue ordinary decoding. Feedback after EOS is not consumed.

The path does not use refinement, RDC, continuous two-image decoding, or newline/transition-anchor mixing. Greedy candidate selection retains the validated BF16 temperature/filter/softmax/argmax order.

## Core files

- `core/lean_generation.py`: main decoding loop.
- `core/online_provider.py`: weak-image replay and direction construction.
- `core/frozen_engine.py`: preparation, replay, and state-isolation integration.
- `core/source/`: required model, prompt, compatibility replay, and cache utilities.
- `configs/models.example.json`: model contracts and entropy thresholds.
- `run_svdf.py`: SVDF, CoT, and one-soft single-image entry point.
- `SOURCE_PROVENANCE.json`: original source hashes and export provenance.

## Run

The reference environment uses NVIDIA PyTorch 24.01 (`torch 2.2.0a0+81ea7a4`), Transformers 4.49.0, Qwen VL Utils 0.0.14, NumPy 1.24.4, Pillow 10.2.0, and Accelerate 1.10.1. Install a CUDA-enabled PyTorch separately, then `pip install -r requirements.txt`. The implementation retains the tested Qwen2.5-VL cache and vision-SDPA interfaces.

Supported checkpoint families are R1-Onevision-7B-RL, OpenVLThinker-7B, and Qwen2.5-VL-3B-Instruct. The exported contracts use one CUDA device. Weights and datasets are not included or automatically downloaded.

```bash
CUDA_VISIBLE_DEVICES=0 python run_svdf.py --model r1 \
  --model-path /path/to/R1-Onevision-7B-RL \
  --image /path/to/image.jpg --question "What is the color of the lock?" \
  --run-dir runs/example
```

Use `--options` for multiline answer choices, and `--method cot` or `--method one-soft` for the corresponding baseline. Each run directory must be new. Preserve the original question/options for existing evaluation protocols; this CLI does not perform benchmark scoring.

Frozen entropy thresholds: R1 **1.25**, OpenVL **0.01**, Qwen2.5-VL-3B **0.66015625**. They are checkpoint-specific settings.

## Provenance

Sixteen dependency modules are exported byte-for-byte from the frozen fixed-input implementation. The `OnlineProvider` class is extracted with identical syntax structure; the old experiment driver is omitted. The portable CLI and configuration paths are new packaging. Syntax, imports, source hashes, and dependency closure are checked without allocating another GPU. This is not a new GPU equivalence experiment.

The original experiment tree and running jobs remain unchanged. Models, datasets, generated answers, credentials, and historical experiment outputs are not uploaded.

## Acknowledgments

Replay and prompt infrastructure derive from [LEAD](https://github.com/mlrm-LEAD/mlrm-LEAD). Its original MIT license and copyright are retained in `LICENSE`. Other historical controls in the vendored compatibility decoder are inactive on the SVDF path described above.
