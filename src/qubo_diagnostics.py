import json
import numpy as np


def analyze_qubo(track_name):
    sweep_path = f"outputs/metrics/qubo_sweep_{track_name}.json"
    with open(sweep_path, "r") as f:
        data = json.load(f)

    Q = np.array(data["qubo_matrix"])
    results = data["results"]

    # Find baseline loss for centering
    baseline_loss = None
    for r in results:
        if sum(r["bitstring"]) == 0:
            baseline_loss = r["val_loss"]
            break

    print(f"\n{'='*60}")
    print(f"QUBO Surrogate Diagnostics: {track_name}")
    print(f"{'='*60}")
    print(f"\nBaseline val_loss (all-zeros): {baseline_loss:.4f}")

    # Compute predicted vs observed
    observed = []
    predicted = []
    configs = []

    for r in results:
        bits = np.array(r["bitstring"])
        obs = r["val_loss"] - baseline_loss
        pred = float(bits @ Q @ bits)
        observed.append(obs)
        predicted.append(pred)
        configs.append(r["config"])

    observed = np.array(observed)
    predicted = np.array(predicted)
    residuals = observed - predicted

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Mean absolute error
    mae = np.mean(np.abs(residuals))

    # Ranking comparison
    obs_rank = np.argsort(observed)
    pred_rank = np.argsort(predicted)

    # Spearman rank correlation
    from scipy.stats import spearmanr
    spearman_corr, spearman_p = spearmanr(observed, predicted)

    # Top-5 agreement
    obs_top5 = set(obs_rank[:5])
    pred_top5 = set(pred_rank[:5])
    top5_overlap = len(obs_top5 & pred_top5)

    # Best config agreement
    obs_best_idx = obs_rank[0]
    pred_best_idx = pred_rank[0]
    best_match = obs_best_idx == pred_best_idx

    print(f"\n--- Fit Quality ---")
    print(f"R-squared:              {r_squared:.4f}")
    print(f"Mean Absolute Error:    {mae:.4f}")
    print(f"Spearman correlation:   {spearman_corr:.4f} (p={spearman_p:.4f})")

    print(f"\n--- Ranking Agreement ---")
    print(f"Best config match:      {'Yes' if best_match else 'No'}")
    print(f"Top-5 overlap:          {top5_overlap}/5")

    # Print predicted vs observed table
    print(f"\n--- Predicted vs Observed (centered val_loss) ---")
    print(f"{'Bitstring':<15} {'LR':<8} {'DO':<6} {'BS':<5} {'EP':<5} "
          f"{'Observed':<10} {'Predicted':<10} {'Residual':<10}")
    print("-" * 75)

    # Sort by observed for readability
    sort_idx = np.argsort(observed)
    for i in sort_idx:
        r = results[i]
        bits_str = str(r["bitstring"])
        c = r["config"]
        print(f"{bits_str:<15} {c['learning_rate']:<8} {c['dropout']:<6} "
              f"{c['batch_size']:<5} {c['epochs']:<5} "
              f"{observed[i]:>+.4f}    {predicted[i]:>+.4f}    {residuals[i]:>+.4f}")

    # Save diagnostics
    diagnostics = {
        "track": track_name,
        "r_squared": r_squared,
        "mae": mae,
        "spearman_correlation": spearman_corr,
        "spearman_p_value": spearman_p,
        "best_config_match": best_match,
        "top5_overlap": top5_overlap,
        "predicted_vs_observed": [
            {
                "config": results[i]["config"],
                "bitstring": results[i]["bitstring"],
                "observed": float(observed[i]),
                "predicted": float(predicted[i]),
                "residual": float(residuals[i]),
            }
            for i in sort_idx
        ],
    }

    save_path = f"outputs/metrics/qubo_diagnostics_{track_name}.json"
    with open(save_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\nSaved to {save_path}")

    return diagnostics


if __name__ == "__main__":
    analyze_qubo("primary")
    analyze_qubo("allagree")