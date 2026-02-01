import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
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
    make_transformer_loader,
    predict_scores,
    predict_transformer_scores,
    set_seed,
    topk_metrics,
    train_full,
    train_transformer_full,
    FastTextDataset,
    TextDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tuned models and evaluate on test data.")
    parser.add_argument("--train_path", default="data/train_data.csv")
    parser.add_argument("--test_path", default="data/test_data.csv")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--best_params", default="outputs/best_params.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="List of models to train or 'all'.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_vocab", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=2)
    return parser.parse_args()


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


def read_data(path: str):
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    df = df.dropna(subset=["text", "emoji"])
    return df["text"].tolist(), df["emoji"].tolist()


def build_tfidf_union(params: Dict):
    def _as_ngram_range(value):
        if isinstance(value, list):
            return tuple(value)
        return value

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=_as_ngram_range(params["word_ngram"]),
        min_df=params["min_df"],
        max_features=params["max_features"],
        lowercase=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=_as_ngram_range(params["char_ngram"]),
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

    out_dir = ensure_dir(args.output_dir)
    results_path = out_dir / "test_results.csv"

    texts_train, labels_train = read_data(args.train_path)
    texts_test, labels_test = read_data(args.test_path)

    label_list = sorted(set(labels_train))
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}

    y_train = np.array([label2id[l] for l in labels_train])
    filtered_texts_test = []
    filtered_y_test = []
    for text, lab in zip(texts_test, labels_test):
        if lab in label2id:
            filtered_texts_test.append(text)
            filtered_y_test.append(label2id[lab])
    texts_test = filtered_texts_test
    y_test = np.array(filtered_y_test)
    dropped = len(labels_test) - len(texts_test)
    if dropped:
        print(f"Skipped {dropped} test samples with unseen labels.")

    best_params: Dict[str, Dict] = {}
    best_path = Path(args.best_params)
    if best_path.exists():
        with best_path.open("r", encoding="utf-8") as f:
            best_params = json.load(f)

    models = resolve_models(args.models)
    ks = (1, 3, 5)

    if any(m in models for m in ["tfidf_svm", "tfidf_logreg", "tfidf_nb"]):
        for model_name in ["tfidf_svm", "tfidf_logreg", "tfidf_nb"]:
            if model_name not in models:
                continue
            params = best_params.get(model_name, {}).get("params")
            if not params:
                raise ValueError(f"Missing tuned params for {model_name}.")
            print(f"Training {model_name} params={format_params(params)}")
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
            pipeline.fit(texts_train, y_train)
            print(f"Evaluating {model_name}")
            scores = get_scores(pipeline, texts_test)
            metrics = topk_metrics(y_test, scores, ks)
            print(
                f"Eval {model_name} top1={metrics.get('top1_accuracy'):.4f} "
                f"macro_f1@1={metrics.get('macro_f1@1'):.4f}"
            )
            row = {
                "model": model_name,
                "params": format_params(params),
                **metrics,
            }
            append_result(results_path, row)

    torch_models = {"fasttext", "cnn", "bilstm", "bigru", "transformer"}
    if torch_models.intersection(models):
        vocab = build_vocab(texts_train, max_size=args.max_vocab, min_freq=args.min_freq)

    if "fasttext" in models:
        params = best_params.get("fasttext", {}).get("params")
        if not params:
            raise ValueError("Missing tuned params for fasttext.")
        print(f"Training fasttext params={format_params(params)}")
        train_ds = FastTextDataset(
            texts_train,
            y_train,
            vocab,
            max_words=params["max_words"],
            min_n=params["min_n"],
            max_n=params["max_n"],
            num_buckets=params["num_buckets"],
            max_ngrams=params["max_ngrams"],
        )
        test_ds = FastTextDataset(
            texts_test,
            y_test,
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
        test_loader = torch.utils.data.DataLoader(
            test_ds,
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
        train_full(model, train_loader, device, "fasttext", params["lr"], params["epochs"])
        print("Evaluating fasttext")
        scores, y_true = predict_scores(model, test_loader, device, "fasttext")
        metrics = topk_metrics(y_true, scores, ks)
        print(
            f"Eval fasttext top1={metrics.get('top1_accuracy'):.4f} "
            f"macro_f1@1={metrics.get('macro_f1@1'):.4f}"
        )
        row = {
            "model": "fasttext",
            "params": format_params(params),
            **metrics,
        }
        append_result(results_path, row)

    if "cnn" in models:
        params = best_params.get("cnn", {}).get("params")
        if not params:
            raise ValueError("Missing tuned params for cnn.")
        print(f"Training cnn params={format_params(params)}")
        train_ds = TextDataset(texts_train, y_train, vocab, max_len=params["max_len"])
        test_ds = TextDataset(texts_test, y_test, vocab, max_len=params["max_len"])
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=0,
            collate_fn=lambda b: collate_text(b, vocab.pad_id),
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
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
        train_full(model, train_loader, device, "cnn", params["lr"], params["epochs"])
        print("Evaluating cnn")
        scores, y_true = predict_scores(model, test_loader, device, "cnn")
        metrics = topk_metrics(y_true, scores, ks)
        print(
            f"Eval cnn top1={metrics.get('top1_accuracy'):.4f} "
            f"macro_f1@1={metrics.get('macro_f1@1'):.4f}"
        )
        row = {
            "model": "cnn",
            "params": format_params(params),
            **metrics,
        }
        append_result(results_path, row)

    if "bilstm" in models or "bigru" in models:
        for model_name in ["bilstm", "bigru"]:
            if model_name not in models:
                continue
            params = best_params.get(model_name, {}).get("params")
            if not params:
                raise ValueError(f"Missing tuned params for {model_name}.")
            print(f"Training {model_name} params={format_params(params)}")
            train_ds = TextDataset(texts_train, y_train, vocab, max_len=params["max_len"])
            test_ds = TextDataset(texts_test, y_test, vocab, max_len=params["max_len"])
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=params["batch_size"],
                shuffle=True,
                num_workers=0,
                collate_fn=lambda b: collate_text(b, vocab.pad_id),
            )
            test_loader = torch.utils.data.DataLoader(
                test_ds,
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
            train_full(model, train_loader, device, model_name, params["lr"], params["epochs"])
            print(f"Evaluating {model_name}")
            scores, y_true = predict_scores(model, test_loader, device, model_name)
            metrics = topk_metrics(y_true, scores, ks)
            print(
                f"Eval {model_name} top1={metrics.get('top1_accuracy'):.4f} "
                f"macro_f1@1={metrics.get('macro_f1@1'):.4f}"
            )
            row = {
                "model": model_name,
                "params": format_params(params),
                **metrics,
            }
            append_result(results_path, row)

    if "transformer" in models:
        params = best_params.get("transformer", {}).get("params")
        if not params:
            raise ValueError("Missing tuned params for transformer.")
        print(f"Training transformer params={format_params(params)}")
        tokenizer, model = train_transformer_full(
            model_name=params["model_name"],
            train_texts=texts_train,
            train_labels=y_train,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            lr=params["lr"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            max_len=params["max_len"],
            device=device,
        )
        print("Evaluating transformer")
        test_loader = make_transformer_loader(
            tokenizer, texts_test, y_test, params["max_len"], params["batch_size"], False
        )
        scores, y_true = predict_transformer_scores(model, test_loader, device)
        metrics = topk_metrics(y_true, scores, ks)
        print(
            f"Eval transformer top1={metrics.get('top1_accuracy'):.4f} "
            f"macro_f1@1={metrics.get('macro_f1@1'):.4f}"
        )
        row = {
            "model": "transformer",
            "params": format_params(params),
            **metrics,
        }
        append_result(results_path, row)

    print(f"Saved test results to {results_path}")


if __name__ == "__main__":
    main()
