import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import time
from transformers import AutoConfig

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


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


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
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

    avg_loss = total_loss / n_batches
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }


def train_baseline(track_name, data_dir, output_dir, config):
    print(f"\n{'='*60}")
    print(f"Training baseline for: {track_name}")
    print(f"Config: {config}")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # Load tokenizer and model
    print("Loading FinBERT...")



    # THIS DATASET IS ALREADY FINE-TUNED -- INFLATES RESULTS
    # tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "ProsusAI/finbert", num_labels=3
    # )

    # INSTEAD USING THIS DATASET, WITH PRE-TRAINING BUT NOT FINT-TUNING
    # Change to this:
    # tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "ProsusAI/finbert", num_labels=3
    # )

    # # Reset the classifier head so it's untrained
    # model.classifier.weight.data.normal_(mean=0.0, std=0.02)
    # model.classifier.bias.data.zero_()

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

    # Create datasets
    train_dataset = SentimentDataset(
        train_df["text"], train_df["label_id"], tokenizer, max_length=config["max_length"]
    )
    val_dataset = SentimentDataset(
        val_df["text"], val_df["label_id"], tokenizer, max_length=config["max_length"]
    )
    test_dataset = SentimentDataset(
        test_df["text"], test_df["label_id"], tokenizer, max_length=config["max_length"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            epoch_loss += loss.item()
            n_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{config['epochs']} "
                      f"Batch {batch_idx+1}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        val_losses.append(val_metrics["loss"])

        print(f"\nEpoch {epoch+1}/{config['epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")
        print(f"  Val F1:     {val_metrics['f1']:.4f}")

        # Early stopping check
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            # Save best model
            model_dir = os.path.join(output_dir, "best_model")
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    # Load best model for test evaluation
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(output_dir, "best_model"), num_labels=3
    )
    model.to(device)

    # Test evaluation
    test_metrics = evaluate(model, test_loader, device)

    print(f"\n--- Test Results ({track_name}) ---")
    print(f"Accuracy:  {test_metrics['accuracy']*100:.1f}%")
    print(f"Precision: {test_metrics['precision']:.2f}")
    print(f"Recall:    {test_metrics['recall']:.2f}")
    print(f"F1:        {test_metrics['f1']:.2f}")
    print(f"Confusion Matrix:")
    for row in test_metrics["confusion_matrix"]:
        print(f"  {row}")

    # Save metrics
    metrics_dir = os.path.join("outputs", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    results = {
        "track": track_name,
        "config": config,
        "training_time_minutes": elapsed / 60,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
    }

    results_path = os.path.join(metrics_dir, f"baseline_{track_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {results_path}")

    # Save loss plot
    plots_dir = os.path.join("outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Baseline FinBERT Loss - {track_name}")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"baseline_loss_{track_name}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to {plot_path}")


if __name__ == "__main__":
    # Baseline config from the paper
    baseline_config = {
        "learning_rate": 2e-5,
        "batch_size": 32,
        "epochs": 3,
        "dropout": 0.1,
        "max_length": 64,
    }

    # Train on AllAgree track (paper-aligned)
    train_baseline(
        track_name="allagree",
        data_dir="data/processed/allagree",
        output_dir="models/baseline_allagree",
        config=baseline_config,
    )

    # Train on Primary CSV track
    train_baseline(
        track_name="primary",
        data_dir="data/processed/primary",
        output_dir="models/baseline_primary",
        config=baseline_config,
    )