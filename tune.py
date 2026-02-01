import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from ml import (
    FastTextClassifier,
    RNNWithAttention,
    TextCNN,
    build_vocab,
    collate_fasttext,
    collate_text,
    set_seed,
    topk_metrics,
    train_transformer_with_val,
    train_with_val,
    FastTextDataset,
    TextDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune emoji prediction models.")
    parser.add_argument("--train_path", default="data/train_data.csv")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="List of models to tune or 'all'.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_vocab", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=2)
    return parser.parse_args()


SKLEARN_GRIDS = {
    "tfidf_svm": [
        {
            "word_ngram": (1, 2),
            "char_ngram": (3, 5),
            "min_df": 2,
            "max_features": 200000,
            "C": 1.0,
            "class_weight": "balanced",
        },
        {
            "word_ngram": (1, 3),
            "char_ngram": (3, 6),
            "min_df": 2,
            "max_features": 300000,
            "C": 2.0,
            "class_weight": "balanced",
        },
    ],
    "tfidf_logreg": [
        {
            "word_ngram": (1, 2),
            "char_ngram": (3, 5),
            "min_df": 2,
            "max_features": 200000,
            "C": 2.0,
        },
        {
            "word_ngram": (1, 3),
            "char_ngram": (3, 6),
            "min_df": 2,
            "max_features": 300000,
            "C": 4.0,
        },
    ],
    "tfidf_nb": [
        {
            "word_ngram": (1, 2),
            "char_ngram": (3, 5),
            "min_df": 2,
            "max_features": 200000,
            "alpha": 0.5,
        },
        {
            "word_ngram": (1, 3),
            "char_ngram": (3, 6),
            "min_df": 2,
            "max_features": 300000,
            "alpha": 1.0,
        },
    ],
}

FASTTEXT_GRID = [
    {
        "embed_dim": 100,
        "lr": 1e-3,
        "dropout": 0.2,
        "min_n": 3,
        "max_n": 6,
        "num_buckets": 200000,
        "max_words": 200,
        "max_ngrams": 2000,
        "batch_size": 128,
        "epochs": 5,
    },
    {
        "embed_dim": 200,
        "lr": 5e-4,
        "dropout": 0.3,
        "min_n": 3,
        "max_n": 6,
        "num_buckets": 300000,
        "max_words": 200,
        "max_ngrams": 3000,
        "batch_size": 128,
        "epochs": 6,
    },
]

CNN_GRID = [
    {
        "embed_dim": 100,
        "num_filters": 100,
        "filter_sizes": (3, 4, 5),
        "dropout": 0.5,
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 5,
        "max_len": 200,
    },
    {
        "embed_dim": 200,
        "num_filters": 128,
        "filter_sizes": (3, 4, 5),
        "dropout": 0.5,
        "lr": 5e-4,
        "batch_size": 128,
        "epochs": 6,
        "max_len": 200,
    },
]

RNN_GRID = [
    {
        "rnn_type": "lstm",
        "embed_dim": 100,
        "hidden_size": 128,
        "dropout": 0.4,
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 6,
        "max_len": 200,
    },
    {
        "rnn_type": "gru",
        "embed_dim": 100,
        "hidden_size": 128,
        "dropout": 0.4,
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 6,
        "max_len": 200,
    },
]

TRANSFORMER_GRID = [
    {
        "model_name": "distilbert-base-uncased",
        "lr": 2e-5,
        "batch_size": 32,
        "epochs": 3,
        "max_len": 128,
    }
]


ALL_MODELS = [
    "tfidf_svm",
    "tfidf_logreg",
    "tfidf_nb",
    "fasttext",
    "cnn",
    "bilstm",
    "bigru",
    "transformer",
]


def resolve_models(models: List[str]) -> List[str]:
    if not models or "all" in models:
        return ALL_MODELS
    return models


def apply_overrides(params: Dict, args: argparse.Namespace) -> Dict:
    updated = dict(params)
    if args.batch_size is not None:
        updated["batch_size"] = args.batch_size
    if args.epochs is not None:
        updated["epochs"] = args.epochs
    return updated


def read_data(path: str):
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    df = df.dropna(subset=["text", "emoji"])
    return df["text"].tolist(), df["emoji"].tolist()


def maybe_sample(texts, labels, max_samples: int, seed: int):
    if max_samples and max_samples < len(texts):
        texts, _, labels, _ = train_test_split(
            texts,
            labels,
            train_size=max_samples,
            stratify=labels,
            random_state=seed,
        )
    return texts, labels


def build_tfidf_union(params: Dict):
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=params["word_ngram"],
        min_df=params["min_df"],
        max_features=params["max_features"],
        lowercase=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=params["char_ngram"],
        min_df=params["min_df"],
        max_features=params["max_features"],
        lowercase=True,
    )
    return FeatureUnion([("word", word_vec), ("char", char_vec)])


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        return scores
    raise ValueError("Model does not support scoring.")


def ensure_dir(path: str) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def append_result(path: Path, row: Dict):
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def format_params(params: Dict) -> str:
    return json.dumps(params, sort_keys=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    texts, labels = read_data(args.train_path)
    texts, labels = maybe_sample(texts, labels, args.max_samples, args.seed)
    label_list = sorted(set(labels))
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}
    y = np.array([label2id[l] for l in labels])

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        y,
        test_size=args.val_size,
        stratify=y,
        random_state=args.seed,
    )

    out_dir = ensure_dir(args.output_dir)
    results_path = out_dir / "tuning_results.csv"
    best_params: Dict[str, Dict] = {}

    models = resolve_models(args.models)

    ks = (1, 3, 5)

    if any(m in models for m in ["tfidf_svm", "tfidf_logreg", "tfidf_nb"]):
        for model_name in ["tfidf_svm", "tfidf_logreg", "tfidf_nb"]:
            if model_name not in models:
                continue
            grid = SKLEARN_GRIDS[model_name]
            for idx, params in enumerate(grid, start=1):
                print(f"Tuning {model_name} [{idx}/{len(grid)}] params={format_params(params)}")
                vectorizer = build_tfidf_union(params)
                if model_name == "tfidf_svm":
                    clf = LinearSVC(C=params["C"], class_weight=params["class_weight"])
                elif model_name == "tfidf_logreg":
                    clf = LogisticRegression(
                        C=params["C"],
                        max_iter=2000,
                        solver="saga",
                        n_jobs=-1,
                    )
                else:
                    clf = MultinomialNB(alpha=params["alpha"])
                pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])
                pipeline.fit(X_train, y_train)
                scores = get_scores(pipeline, X_val)
                metrics = topk_metrics(y_val, scores, ks)
                print(
                    f"Eval {model_name} [{idx}/{len(grid)}] "
                    f"top1={metrics.get('top1_accuracy'):.4f} "
                    f"macro_f1@1={metrics.get('macro_f1@1'):.4f}"
                )
                row = {
                    "model": model_name,
                    "params": format_params(params),
                    **metrics,
                }
                append_result(results_path, row)
                score = metrics.get("macro_f1@1", 0.0)
                if model_name not in best_params or score > best_params[model_name]["score"]:
                    best_params[model_name] = {"params": params, "metrics": metrics, "score": score}

    torch_models = {"fasttext", "cnn", "bilstm", "bigru", "transformer"}
    if torch_models.intersection(models):
        vocab = build_vocab(X_train, max_size=args.max_vocab, min_freq=args.min_freq)

    if "fasttext" in models:
        for idx, params in enumerate(FASTTEXT_GRID, start=1):
            params = apply_overrides(params, args)
            print(f"Tuning fasttext [{idx}/{len(FASTTEXT_GRID)}] params={format_params(params)}")
            train_ds = FastTextDataset(
                X_train,
                y_train,
                vocab,
                max_words=params["max_words"],
                min_n=params["min_n"],
                max_n=params["max_n"],
                num_buckets=params["num_buckets"],
                max_ngrams=params["max_ngrams"],
            )
            val_ds = FastTextDataset(
                X_val,
                y_val,
                vocab,
                max_words=params["max_words"],
                min_n=params["min_n"],
                max_n=params["max_n"],
                num_buckets=params["num_buckets"],
                max_ngrams=params["max_ngrams"],
            )
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=params["batch_size"],
                shuffle=True,
                num_workers=0,
                collate_fn=lambda b: collate_fasttext(b, vocab.pad_id),
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=params["batch_size"],
                shuffle=False,
                num_workers=0,
                collate_fn=lambda b: collate_fasttext(b, vocab.pad_id),
            )
            model = FastTextClassifier(
                vocab_size=len(vocab),
                ngram_buckets=params["num_buckets"],
                embed_dim=params["embed_dim"],
                num_classes=len(label2id),
                pad_id=vocab.pad_id,
                dropout=params["dropout"],
            ).to(device)
            model, metrics = train_with_val(
                model,
                train_loader,
                val_loader,
                device,
                "fasttext",
                params["lr"],
                params["epochs"],
                args.patience,
                ks,
            )
            print(
                f"Eval fasttext [{idx}/{len(FASTTEXT_GRID)}] "
                f"top1={metrics.get('top1_accuracy'):.4f} "
                f"macro_f1@1={metrics.get('macro_f1@1'):.4f}"
            )
            row = {
                "model": "fasttext",
                "params": format_params(params),
                **metrics,
            }
            append_result(results_path, row)
            score = metrics.get("macro_f1@1", 0.0)
            if "fasttext" not in best_params or score > best_params["fasttext"]["score"]:
                best_params["fasttext"] = {"params": params, "metrics": metrics, "score": score}

    if "cnn" in models:
        for idx, params in enumerate(CNN_GRID, start=1):
            params = apply_overrides(params, args)
            print(f"Tuning cnn [{idx}/{len(CNN_GRID)}] params={format_params(params)}")
            train_ds = TextDataset(X_train, y_train, vocab, max_len=params["max_len"])
            val_ds = TextDataset(X_val, y_val, vocab, max_len=params["max_len"])
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=params["batch_size"],
                shuffle=True,
                num_workers=0,
                collate_fn=lambda b: collate_text(b, vocab.pad_id),
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=params["batch_size"],
                shuffle=False,
                num_workers=0,
                collate_fn=lambda b: collate_text(b, vocab.pad_id),
            )
            model = TextCNN(
                vocab_size=len(vocab),
                embed_dim=params["embed_dim"],
                num_classes=len(label2id),
                pad_id=vocab.pad_id,
                filter_sizes=params["filter_sizes"],
                num_filters=params["num_filters"],
                dropout=params["dropout"],
            ).to(device)
            model, metrics = train_with_val(
                model,
                train_loader,
                val_loader,
                device,
                "cnn",
                params["lr"],
                params["epochs"],
                args.patience,
                ks,
            )
            print(
                f"Eval cnn [{idx}/{len(CNN_GRID)}] "
                f"top1={metrics.get('top1_accuracy'):.4f} "
                f"macro_f1@1={metrics.get('macro_f1@1'):.4f}"
            )
            row = {
                "model": "cnn",
                "params": format_params(params),
                **metrics,
            }
            append_result(results_path, row)
            score = metrics.get("macro_f1@1", 0.0)
            if "cnn" not in best_params or score > best_params["cnn"]["score"]:
                best_params["cnn"] = {"params": params, "metrics": metrics, "score": score}

    if "bilstm" in models or "bigru" in models:
        for idx, params in enumerate(RNN_GRID, start=1):
            params = apply_overrides(params, args)
            model_name = "bilstm" if params["rnn_type"] == "lstm" else "bigru"
            if model_name not in models:
                continue
            print(f"Tuning {model_name} [{idx}/{len(RNN_GRID)}] params={format_params(params)}")
            train_ds = TextDataset(X_train, y_train, vocab, max_len=params["max_len"])
            val_ds = TextDataset(X_val, y_val, vocab, max_len=params["max_len"])
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=params["batch_size"],
                shuffle=True,
                num_workers=0,
                collate_fn=lambda b: collate_text(b, vocab.pad_id),
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=params["batch_size"],
                shuffle=False,
                num_workers=0,
                collate_fn=lambda b: collate_text(b, vocab.pad_id),
            )
            model = RNNWithAttention(
                vocab_size=len(vocab),
                embed_dim=params["embed_dim"],
                hidden_size=params["hidden_size"],
                num_classes=len(label2id),
                pad_id=vocab.pad_id,
                rnn_type=params["rnn_type"],
                dropout=params["dropout"],
            ).to(device)
            model, metrics = train_with_val(
                model,
                train_loader,
                val_loader,
                device,
                model_name,
                params["lr"],
                params["epochs"],
                args.patience,
                ks,
            )
            print(
                f"Eval {model_name} [{idx}/{len(RNN_GRID)}] "
                f"top1={metrics.get('top1_accuracy'):.4f} "
                f"macro_f1@1={metrics.get('macro_f1@1'):.4f}"
            )
            row = {
                "model": model_name,
                "params": format_params(params),
                **metrics,
            }
            append_result(results_path, row)
            score = metrics.get("macro_f1@1", 0.0)
            if model_name not in best_params or score > best_params[model_name]["score"]:
                best_params[model_name] = {"params": params, "metrics": metrics, "score": score}

    if "transformer" in models:
        for idx, params in enumerate(TRANSFORMER_GRID, start=1):
            params = apply_overrides(params, args)
            print(f"Tuning transformer [{idx}/{len(TRANSFORMER_GRID)}] params={format_params(params)}")
            _, _, metrics = train_transformer_with_val(
                model_name=params["model_name"],
                train_texts=X_train,
                train_labels=y_train,
                val_texts=X_val,
                val_labels=y_val,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
                lr=params["lr"],
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                max_len=params["max_len"],
                device=device,
                ks=ks,
            )
            print(
                f"Eval transformer [{idx}/{len(TRANSFORMER_GRID)}] "
                f"top1={metrics.get('top1_accuracy'):.4f} "
                f"macro_f1@1={metrics.get('macro_f1@1'):.4f}"
            )
            row = {
                "model": "transformer",
                "params": format_params(params),
                **metrics,
            }
            append_result(results_path, row)
            score = metrics.get("macro_f1@1", 0.0)
            if "transformer" not in best_params or score > best_params["transformer"]["score"]:
                best_params["transformer"] = {"params": params, "metrics": metrics, "score": score}

    best_path = out_dir / "best_params.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, sort_keys=True)

    print(f"Saved tuning results to {results_path}")
    print(f"Saved best params to {best_path}")


if __name__ == "__main__":
    main()
