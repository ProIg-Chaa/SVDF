"""Single-image entry point for the frozen fixed-window input SVDF method."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="SVDF: early capture and four visual input feedbacks")
    parser.add_argument("--model", choices=["r1", "openvl", "qwen3b", "rethinker"], default="r1")
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "models.example.json"),
                        help="Model contracts and initialization profile (JSON)")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", default="", help="Optional multiline A/B/C choices")
    parser.add_argument("--method", choices=["svdf", "cot", "one-soft"], default="svdf")
    parser.add_argument("--run-dir", default="runs/single", help="New directory; never overwritten")
    args = parser.parse_args()
    image = Path(args.image).resolve()
    model_path = Path(args.model_path).resolve()
    if not image.is_file() or not model_path.is_dir():
        parser.error("image must be a file and model-path must be an existing local checkpoint")
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / "core"))
    sys.path.insert(0, str(ROOT / "core" / "source"))
    import frozen_engine as engine
    from lean_generation import generate_lean_fixed
    from online_provider import OnlineProvider
    from common_decode import generate_baseline
    spec = json.loads(Path(args.model_config).read_text())[args.model]
    if args.method == "one-soft":
        spec["initialization"] = "onesoft"
    spec["path"] = str(model_path)
    sample = {"id": "single", "dataset": "user", "question": args.question,
              "options": args.options, "image": str(image), "pair_id": None,
              "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
    engine.init(spec, run, spec["threshold"])
    provider = None
    if args.method == "cot":
        inputs, _ = engine.base.prepare(engine.processor, sample, sample["image"], engine.model.device)
        result = generate_baseline(engine.model, engine.tokenizer, inputs,
            config={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                    "do_sample": False, "max_new_tokens": 1024}, seed=42)
        tokens, trace, meta = result["token_ids"], result["trace"], result["metadata"]
    else:
        if args.method == "svdf":
            provider = OnlineProvider(sample, run / "weak.png")
        tokens, trace, meta = engine.generate(sample, cap=0, lam=0.95,
            provider=provider, generator=generate_lean_fixed)
    text = engine.tokenizer.decode(tokens, skip_special_tokens=True)
    report = {"method": args.method, "model": args.model, "text": text,
              "generated_token_ids": tokens, "trace": trace, "metadata": meta,
              "capture_event": provider.event if provider else None,
              "capture_audit": provider.audit if provider else None}
    (run / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(text)

if __name__ == "__main__":
    main()
