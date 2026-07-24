import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

SEEDS = [42, 123, 456, 789, 1024]


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.encodings = tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
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


def train_and_evaluate_single(config, train_df, val_df, test_df, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained("ahmedrachid/FinancialBERT")
    model_config = AutoConfig.from_pretrained("ahmedrachid/FinancialBERT")
    model_config.hidden_dropout_prob = config["dropout"]
    model_config.attention_probs_dropout_prob = config["dropout"]
    model_config.classifier_dropout = config["dropout"]
    model_config.num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        "ahmedrachid/FinancialBERT", config=model_config
    )
    model.to(device)

    train_dataset = SentimentDataset(train_df["text"], train_df["label_id"], tokenizer)
    test_dataset = SentimentDataset(test_df["text"], test_df["label_id"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def run_repeated_trials(track_name, data_dir, configs, device):
    print(f"\n{'='*60}")
    print(f"Repeated trials for: {track_name}")
    print(f"{'='*60}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    all_results = {}

    for config_name, config in configs.items():
        print(f"\n--- {config_name}: {config} ---")
        run_results = []

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ")
            start = time.time()
            metrics = train_and_evaluate_single(config, train_df, val_df, test_df, seed, device)
            elapsed = time.time() - start
            print(f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} ({elapsed/60:.1f}min)")
            run_results.append(metrics)

        accs = [r["accuracy"] for r in run_results]
        f1s = [r["f1"] for r in run_results]

        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1)
        ci95_acc = 1.96 * std_acc / np.sqrt(len(accs))

        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s, ddof=1)
        ci95_f1 = 1.96 * std_f1 / np.sqrt(len(f1s))

        print(f"\n  Accuracy: {mean_acc*100:.1f}% ± {ci95_acc*100:.1f}% (95% CI)")
        print(f"  Macro F1: {mean_f1:.3f} ± {ci95_f1:.3f} (95% CI)")

        all_results[config_name] = {
            "config": config,
            "runs": run_results,
            "summary": {
                "accuracy_mean": mean_acc,
                "accuracy_std": std_acc,
                "accuracy_ci95": ci95_acc,
                "f1_mean": mean_f1,
                "f1_std": std_f1,
                "f1_ci95": ci95_f1,
            }
        }

    # Save
    os.makedirs("outputs/metrics", exist_ok=True)
    save_path = f"outputs/metrics/repeated_trials_{track_name}.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    primary_configs = {
        "baseline": {"learning_rate": 2e-5, "batch_size": 32, "epochs": 3, "dropout": 0.1},
        "qubo_optimized": {"learning_rate": 1e-5, "batch_size": 16, "epochs": 3, "dropout": 0.3},
        "sweep_best": {"learning_rate": 5e-5, "batch_size": 32, "epochs": 3, "dropout": 0.3},
    }

    allagree_configs = {
        "baseline": {"learning_rate": 2e-5, "batch_size": 32, "epochs": 3, "dropout": 0.1},
        "qubo_optimized": {"learning_rate": 2e-5, "batch_size": 16, "epochs": 4, "dropout": 0.1},
        "sweep_best": {"learning_rate": 5e-5, "batch_size": 16, "epochs": 4, "dropout": 0.1},
    }

    run_repeated_trials("primary", "data/processed/primary", primary_configs, device)
    run_repeated_trials("allagree", "data/processed/allagree", allagree_configs, device)