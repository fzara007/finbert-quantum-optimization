import os
import json
import itertools
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import time
from transformers import AutoConfig

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# The 5-bit search space from the paper
SEARCH_SPACE = {
    "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],   # 2 bits (00,01,10,11)
    "dropout": [0.1, 0.3],                          # 1 bit
    "batch_size": [16, 32],                          # 1 bit
    "epochs": [3, 4],                                # 1 bit
}

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.encodings = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def train_and_evaluate(config, train_df, val_df, tokenizer_name, device):
    """Train FinBERT with given config and return validation loss and accuracy."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model_config = AutoConfig.from_pretrained(tokenizer_name)
    model_config.hidden_dropout_prob = config["dropout"]
    model_config.attention_probs_dropout_prob = config["dropout"]
    model_config.classifier_dropout = config["dropout"]
    model_config.num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        tokenizer_name, config=model_config
    )

    model.to(device)

    train_dataset = SentimentDataset(
        train_df["text"], train_df["label_id"], tokenizer, max_length=64
    )
    val_dataset = SentimentDataset(
        val_df["text"], val_df["label_id"], tokenizer, max_length=64
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    # Train
    for epoch in range(config["epochs"]):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            n_batches += 1
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / n_batches
    val_acc = accuracy_score(all_labels, all_preds)

    return val_loss, val_acc


def bitstring_for_config(lr, dropout, batch_size, epochs):
    """Convert hyperparameter values to 5-bit binary encoding."""
    lr_map = {1e-5: (0, 0), 2e-5: (0, 1), 3e-5: (1, 0), 5e-5: (1, 1)}
    dropout_map = {0.1: 0, 0.3: 1}
    batch_map = {16: 0, 32: 1}
    epoch_map = {3: 0, 4: 1}

    b1, b2 = lr_map[lr]
    b3 = dropout_map[dropout]
    b4 = batch_map[batch_size]
    b5 = epoch_map[epochs]

    return (b1, b2, b3, b4, b5)


def build_qubo_matrix(results):
    n_bits = 5
    n_combos = len(results)
    n_terms = n_bits + n_bits * (n_bits - 1) // 2

    A = np.zeros((n_combos, n_terms))
    y = np.zeros(n_combos)

    # Find the all-zeros baseline loss
    baseline_loss = None
    for r in results:
        if sum(r["bitstring"]) == 0:
            baseline_loss = r["val_loss"]
            break

    for idx, r in enumerate(results):
        bits = r["bitstring"]
        # Subtract baseline so (0,0,0,0,0) maps to target 0
        y[idx] = r["val_loss"] - baseline_loss

        col = 0
        for i in range(n_bits):
            A[idx, col] = bits[i]
            col += 1
        for i in range(n_bits):
            for j in range(i + 1, n_bits):
                A[idx, col] = bits[i] * bits[j]
                col += 1

    q, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)

    Q = np.zeros((n_bits, n_bits))
    col = 0
    for i in range(n_bits):
        Q[i, i] = q[col]
        col += 1
    for i in range(n_bits):
        for j in range(i + 1, n_bits):
            Q[i, j] = q[col]
            col += 1

    return Q


def run_qubo_sweep(track_name, data_dir, tokenizer_name):
    """Run all 32 combinations and build the QUBO matrix."""
    print(f"\n{'='*60}")
    print(f"QUBO sweep for: {track_name}")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))

    # Generate all 32 combinations
    combos = list(itertools.product(
        SEARCH_SPACE["learning_rate"],
        SEARCH_SPACE["dropout"],
        SEARCH_SPACE["batch_size"],
        SEARCH_SPACE["epochs"],
    ))

    print(f"Total combinations: {len(combos)}\n")

    results = []
    start_time = time.time()

    for i, (lr, dropout, bs, ep) in enumerate(combos):
        config = {
            "learning_rate": lr,
            "dropout": dropout,
            "batch_size": bs,
            "epochs": ep,
        }
        bits = bitstring_for_config(lr, dropout, bs, ep)

        print(f"[{i+1}/32] LR={lr}, DO={dropout}, BS={bs}, EP={ep} "
              f"bits={bits}", end=" ... ")

        combo_start = time.time()
        val_loss, val_acc = train_and_evaluate(
            config, train_df, val_df, tokenizer_name, device
        )
        combo_time = time.time() - combo_start

        print(f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({combo_time/60:.1f} min)")

        results.append({
            "config": config,
            "bitstring": bits,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

    total_time = time.time() - start_time
    print(f"\nSweep completed in {total_time/60:.1f} minutes")

    # Build QUBO matrix
    Q = build_qubo_matrix(results)

    print(f"\nQUBO Matrix Q (5x5):")
    print(np.array2string(Q, precision=4, suppress_small=True))

    # Verify: find the bitstring with minimum F(x) = x^T Q x
    best_loss = float("inf")
    best_bits = None
    best_config = None

    for r in results:
        bits = np.array(r["bitstring"])
        qubo_val = bits @ Q @ bits
        r["qubo_value"] = float(qubo_val)
        if r["val_loss"] < best_loss:
            best_loss = r["val_loss"]
            best_bits = r["bitstring"]
            best_config = r["config"]

    # Also find QUBO minimum
    qubo_min_val = float("inf")
    qubo_min_bits = None
    for r in results:
        if r["qubo_value"] < qubo_min_val:
            qubo_min_val = r["qubo_value"]
            qubo_min_bits = r["bitstring"]
            qubo_min_config = r["config"]

    print(f"\nBest by actual val_loss: {best_config} bits={best_bits} "
          f"loss={best_loss:.4f}")
    print(f"Best by QUBO objective:  {qubo_min_config} bits={qubo_min_bits} "
          f"qubo_val={qubo_min_val:.4f}")

    # Save everything
    output_dir = os.path.join("outputs", "metrics")
    os.makedirs(output_dir, exist_ok=True)

    # Convert results for JSON serialization
    save_results = []
    for r in results:
        save_results.append({
            "config": r["config"],
            "bitstring": list(r["bitstring"]),
            "val_loss": r["val_loss"],
            "val_accuracy": r["val_accuracy"],
            "qubo_value": r["qubo_value"],
        })

    output = {
        "track": track_name,
        "total_time_minutes": total_time / 60,
        "qubo_matrix": Q.tolist(),
        "results": save_results,
        "best_by_val_loss": {
            "config": best_config,
            "bitstring": list(best_bits),
            "val_loss": best_loss,
        },
        "best_by_qubo": {
            "config": qubo_min_config,
            "bitstring": list(qubo_min_bits),
            "qubo_value": qubo_min_val,
        },
    }

    save_path = os.path.join(output_dir, f"qubo_sweep_{track_name}.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")

    return Q, results


if __name__ == "__main__":
    tokenizer_name = "ahmedrachid/FinancialBERT"

    # Run on primary track first (more room for improvement)
    Q_primary, results_primary = run_qubo_sweep(
        track_name="primary",
        data_dir="data/processed/primary",
        tokenizer_name=tokenizer_name,
    )

    # Then AllAgree
    Q_allagree, results_allagree = run_qubo_sweep(
        track_name="allagree",
        data_dir="data/processed/allagree",
        tokenizer_name=tokenizer_name,
    )