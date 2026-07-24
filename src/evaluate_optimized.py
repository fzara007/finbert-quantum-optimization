import sys
sys.path.insert(0, "src")
from train_baseline import train_baseline


if __name__ == "__main__":
    # === PRIMARY TRACK ===

    # Baseline
    train_baseline(
        track_name="primary_baseline",
        data_dir="data/processed/primary",
        output_dir="models/baseline_primary",
        config={"learning_rate": 2e-5, "batch_size": 32, "epochs": 3,
                "dropout": 0.1, "max_length": 64},
    )

    # QUBO-optimized (SA/QAOA/VQE all found this)
    train_baseline(
        track_name="primary_optimized",
        data_dir="data/processed/primary",
        output_dir="models/optimized_primary",
        config={"learning_rate": 1e-5, "batch_size": 16, "epochs": 3,
                "dropout": 0.3, "max_length": 64},
    )

    # Sweep best (actual lowest val_loss from the 32 combos)
    train_baseline(
        track_name="primary_sweep_best",
        data_dir="data/processed/primary",
        output_dir="models/sweep_best_primary",
        config={"learning_rate": 5e-5, "batch_size": 32, "epochs": 3,
                "dropout": 0.3, "max_length": 64},
    )

    # === ALLAGREE TRACK ===

    # Baseline
    train_baseline(
        track_name="allagree_baseline",
        data_dir="data/processed/allagree",
        output_dir="models/baseline_allagree",
        config={"learning_rate": 2e-5, "batch_size": 32, "epochs": 3,
                "dropout": 0.1, "max_length": 64},
    )

    # QUBO-optimized (SA/QAOA/VQE all found this)
    train_baseline(
        track_name="allagree_optimized",
        data_dir="data/processed/allagree",
        output_dir="models/optimized_allagree",
        config={"learning_rate": 2e-5, "batch_size": 16, "epochs": 4,
                "dropout": 0.1, "max_length": 64},
    )

    # Sweep best (actual lowest val_loss from the 32 combos)
    train_baseline(
        track_name="allagree_sweep_best",
        data_dir="data/processed/allagree",
        output_dir="models/sweep_best_allagree",
        config={"learning_rate": 5e-5, "batch_size": 16, "epochs": 4,
                "dropout": 0.1, "max_length": 64},
    )