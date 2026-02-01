import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple transformer inference helper.")
    parser.add_argument("--model_dir", default="outputs/transformer_deploy")
    parser.add_argument("--text", nargs="+", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_meta_max_len(model_dir: Path) -> Optional[int]:
    meta_path = model_dir / "training_meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("max_len")


def load_id2label(model) -> Dict[int, str]:
    id2label = getattr(model.config, "id2label", {}) or {}
    return {int(k): v for k, v in id2label.items()}


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def predict(
    texts: List[str],
    model,
    tokenizer,
    device: torch.device,
    max_len: Optional[int],
    topk: int,
    id2label: Dict[int, str],
) -> List[List[Dict[str, float]]]:
    if max_len is None:
        max_len = getattr(tokenizer, "model_max_length", None)
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
    topk = min(topk, probs.size(1))
    top_probs, top_ids = torch.topk(probs, topk, dim=-1)
    results: List[List[Dict[str, float]]] = []
    for row_probs, row_ids in zip(top_probs.cpu(), top_ids.cpu()):
        preds = []
        for score, idx in zip(row_probs.tolist(), row_ids.tolist()):
            label = id2label.get(int(idx), str(idx))
            preds.append({"label": label, "score": float(score)})
        results.append(preds)
    return results


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    device = resolve_device(args.device)
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    load_end = time.perf_counter()

    max_len = args.max_len
    if max_len is None:
        max_len = load_meta_max_len(model_dir)
    id2label = load_id2label(model)

    infer_start = time.perf_counter()
    results = predict(
        texts=args.text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_len=max_len,
        topk=args.topk,
        id2label=id2label,
    )
    infer_end = time.perf_counter()

    load_ms = (load_end - load_start) * 1000.0
    infer_ms = (infer_end - infer_start) * 1000.0
    print(f"Weight load time: {load_ms:.1f} ms")
    print(f"Inference time: {infer_ms:.1f} ms")

    for text, preds in zip(args.text, results):
        print(f"Text: {text}")
        for rank, pred in enumerate(preds, start=1):
            print(f"  {rank}. {pred['label']} ({pred['score']:.4f})")


if __name__ == "__main__":
    main()
