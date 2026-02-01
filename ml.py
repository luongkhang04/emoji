from __future__ import annotations

import math
import random
import re
import zlib
from collections import Counter
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def basic_tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, counter: Counter, max_size: int, min_freq: int) -> None:
        self.itos: List[str] = ["<pad>", "<unk>"]
        for token, freq in counter.most_common():
            if freq < min_freq:
                continue
            if token in self.itos:
                continue
            if len(self.itos) >= max_size:
                break
            self.itos.append(token)
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(tok, self.unk_id) for tok in tokens]

    def __len__(self) -> int:
        return len(self.itos)



def build_vocab(
    texts: Sequence[str],
    tokenizer=basic_tokenize,
    max_size: int = 50000,
    min_freq: int = 2,
) -> Vocab:
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenizer(text))
    return Vocab(counter, max_size=max_size, min_freq=min_freq)


@lru_cache(maxsize=500000)
def subword_ngrams(word: str, min_n: int, max_n: int) -> Tuple[str, ...]:
    if not word:
        return tuple()
    if len(word) < min_n:
        return (word,)
    wrapped = f"<{word}>"
    ngrams: List[str] = []
    for n in range(min_n, max_n + 1):
        for i in range(len(wrapped) - n + 1):
            ngrams.append(wrapped[i : i + n])
    return tuple(ngrams)


def hash_ngrams(ngrams: Iterable[str], num_buckets: int) -> List[int]:
    ids: List[int] = []
    for ng in ngrams:
        h = zlib.crc32(ng.encode("utf-8")) & 0xFFFFFFFF
        ids.append(int(h % num_buckets))
    return ids


class TextDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        vocab: Vocab,
        max_len: int,
        tokenizer=basic_tokenize,
    ) -> None:
        self.labels = list(labels)
        self.seqs: List[List[int]] = []
        for text in texts:
            tokens = tokenizer(text)
            ids = vocab.encode(tokens)[:max_len]
            if not ids:
                ids = [vocab.unk_id]
            self.seqs.append(ids)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.seqs[idx], self.labels[idx]


class FastTextDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        vocab: Vocab,
        max_words: int,
        min_n: int,
        max_n: int,
        num_buckets: int,
        max_ngrams: int,
        tokenizer=basic_tokenize,
    ) -> None:
        self.labels = list(labels)
        self.word_ids: List[List[int]] = []
        self.ngram_ids: List[List[int]] = []
        for text in texts:
            tokens = tokenizer(text)[:max_words]
            if not tokens:
                tokens = ["<unk>"]
            wids = vocab.encode(tokens)
            ngrams: List[int] = []
            for tok in tokens:
                ngs = subword_ngrams(tok, min_n, max_n)
                ngrams.extend(hash_ngrams(ngs, num_buckets))
            if max_ngrams and len(ngrams) > max_ngrams:
                ngrams = ngrams[:max_ngrams]
            if not ngrams:
                ngrams = [0]
            self.word_ids.append(wids)
            self.ngram_ids.append(ngrams)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.word_ids[idx], self.ngram_ids[idx], self.labels[idx]



def _pad_sequences(seqs: Sequence[Sequence[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item()) if lengths.numel() else 1
    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        if not seq:
            continue
        end = len(seq)
        padded[i, :end] = torch.tensor(seq, dtype=torch.long)
    return padded, lengths



def collate_text(batch, pad_id: int) -> Dict[str, torch.Tensor]:
    seqs, labels = zip(*batch)
    input_ids, lengths = _pad_sequences(seqs, pad_id)
    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "labels": torch.tensor(labels, dtype=torch.long),
    }



def collate_fasttext(batch, pad_id: int) -> Dict[str, torch.Tensor]:
    word_ids, ngram_ids, labels = zip(*batch)
    word_pad, _ = _pad_sequences(word_ids, pad_id)
    ngram_pad, _ = _pad_sequences(ngram_ids, 0)
    word_mask = (word_pad != pad_id).float()
    ngram_mask = (ngram_pad != 0).float()
    return {
        "word_ids": word_pad,
        "word_mask": word_mask,
        "ngram_ids": ngram_pad,
        "ngram_mask": ngram_mask,
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class FastTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ngram_buckets: int,
        embed_dim: int,
        num_classes: int,
        pad_id: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.ngram_emb = nn.Embedding(ngram_buckets, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        word_ids: torch.Tensor,
        word_mask: torch.Tensor,
        ngram_ids: torch.Tensor,
        ngram_mask: torch.Tensor,
    ) -> torch.Tensor:
        word_vecs = self.word_emb(word_ids) * word_mask.unsqueeze(-1)
        word_sum = word_vecs.sum(dim=1)
        word_count = word_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        ngram_vecs = self.ngram_emb(ngram_ids) * ngram_mask.unsqueeze(-1)
        ngram_sum = ngram_vecs.sum(dim=1)
        ngram_count = ngram_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        sent_vec = (word_sum + ngram_sum) / (word_count + ngram_count)
        sent_vec = self.dropout(sent_vec)
        return self.fc(sent_vec)


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        pad_id: int,
        filter_sizes: Sequence[int],
        num_filters: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)  # (batch, seq, embed)
        x = x.transpose(1, 2)  # (batch, embed, seq)
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            h = torch.max(h, dim=2).values
            feats.append(h)
        out = torch.cat(feats, dim=1)
        out = self.dropout(out)
        return self.fc(out)


class RNNWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_classes: int,
        pad_id: int,
        rnn_type: str,
        dropout: float,
        num_layers: int = 1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.directions = 2 if bidirectional else 1
        self.attn = nn.Linear(hidden_size * self.directions, hidden_size * self.directions)
        self.context = nn.Parameter(torch.randn(hidden_size * self.directions))
        self.fc = nn.Linear(hidden_size * self.directions, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        attn_scores = torch.tanh(self.attn(out))
        attn_scores = torch.matmul(attn_scores, self.context)
        max_len = out.size(1)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_vec = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)
        attn_vec = self.dropout(attn_vec)
        return self.fc(attn_vec)



def topk_metrics(y_true: Sequence[int], scores: np.ndarray, ks: Sequence[int]) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T
    num_classes = scores.shape[1]
    max_k = max(ks)
    order = np.argsort(scores, axis=1)[:, ::-1]
    results: Dict[str, float] = {}
    for k in ks:
        topk = order[:, :k]
        correct = (topk == y_true[:, None]).any(axis=1)
        acc = float(np.mean(correct))
        recalls: List[float] = []
        f1s: List[float] = []
        for c in range(num_classes):
            true_mask = y_true == c
            if true_mask.sum() == 0:
                continue
            pred_mask = (topk == c).any(axis=1)
            tp = int(np.logical_and(true_mask, pred_mask).sum())
            rec = tp / int(true_mask.sum()) if true_mask.sum() else 0.0
            prec = tp / int(pred_mask.sum()) if pred_mask.sum() else 0.0
            f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
            recalls.append(rec)
            f1s.append(f1)
        results[f"top{k}_accuracy"] = acc
        results[f"macro_recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        results[f"macro_f1@{k}"] = float(np.mean(f1s)) if f1s else 0.0
    return results



def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}



def _forward_from_batch(model: nn.Module, batch: Dict[str, torch.Tensor], model_type: str) -> torch.Tensor:
    if model_type == "fasttext":
        return model(batch["word_ids"], batch["word_mask"], batch["ngram_ids"], batch["ngram_mask"])
    if model_type == "cnn":
        return model(batch["input_ids"])
    if model_type in {"bilstm", "bigru"}:
        return model(batch["input_ids"], batch["lengths"])
    raise ValueError(f"Unknown model_type: {model_type}")



def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_type: str,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad()
        logits = _forward_from_batch(model, batch, model_type)
        loss = loss_fn(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch["labels"].size(0)
        total += batch["labels"].size(0)
    return total_loss / max(total, 1)


def predict_scores(
    model: nn.Module,
    loader,
    device: torch.device,
    model_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            logits = _forward_from_batch(model, batch, model_type)
            scores.append(logits.detach().cpu().numpy())
            labels.append(batch["labels"].detach().cpu().numpy())
    return np.vstack(scores), np.concatenate(labels)



def train_with_val(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    model_type: str,
    lr: float,
    epochs: int,
    patience: int,
    ks: Sequence[int],
) -> Tuple[nn.Module, Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_score = -math.inf
    best_metrics: Dict[str, float] = {}
    bad_epochs = 0
    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer, device, model_type)
        scores, y_true = predict_scores(model, val_loader, device, model_type)
        metrics = topk_metrics(y_true, scores, ks)
        score = metrics.get("macro_f1@1", 0.0)
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics



def train_full(
    model: nn.Module,
    train_loader,
    device: torch.device,
    model_type: str,
    lr: float,
    epochs: int,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer, device, model_type)


class TransformerDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[int]], labels: Sequence[int]):
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item



def build_transformer(model_name: str, num_labels: int, id2label: Dict[int, str], label2id: Dict[str, int]):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id,
    )
    return tokenizer, model



def make_transformer_loader(
    tokenizer,
    texts: Sequence[str],
    labels: Sequence[int],
    max_len: int,
    batch_size: int,
    shuffle: bool,
):
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_len,
    )
    dataset = TransformerDataset(encodings, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def train_transformer_with_val(
    model_name: str,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    lr: float,
    epochs: int,
    batch_size: int,
    max_len: int,
    device: torch.device,
    ks: Sequence[int],
) -> Tuple[object, object, Dict[str, float]]:
    from transformers import get_linear_schedule_with_warmup

    tokenizer, model = build_transformer(model_name, num_labels, id2label, label2id)
    model.to(device)
    train_loader = make_transformer_loader(tokenizer, train_texts, train_labels, max_len, batch_size, True)
    val_loader = make_transformer_loader(tokenizer, val_texts, val_labels, max_len, batch_size, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    best_state = None
    best_metrics: Dict[str, float] = {}
    best_score = -math.inf
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        scores, y_true = predict_transformer_scores(model, val_loader, device)
        metrics = topk_metrics(y_true, scores, ks)
        score = metrics.get("macro_f1@1", 0.0)
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return tokenizer, model, best_metrics



def train_transformer_full(
    model_name: str,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    lr: float,
    epochs: int,
    batch_size: int,
    max_len: int,
    device: torch.device,
) -> Tuple[object, object]:
    from transformers import get_linear_schedule_with_warmup

    tokenizer, model = build_transformer(model_name, num_labels, id2label, label2id)
    model.to(device)
    train_loader = make_transformer_loader(tokenizer, train_texts, train_labels, max_len, batch_size, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
    return tokenizer, model



def predict_transformer_scores(model, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            labels.append(batch["labels"].detach().cpu().numpy())
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            scores.append(outputs.logits.detach().cpu().numpy())
    return np.vstack(scores), np.concatenate(labels)
