import json
import numpy as np
import time

SEED = 42
np.random.seed(SEED)


def simulated_annealing(Q, n_runs=10, n_iterations=500, T_init=1.0, cooling=0.995):
    n_bits = Q.shape[0]
    best_global_bits = None
    best_global_energy = float("inf")
    all_runs = []

    for run in range(n_runs):
        # Random initial bitstring
        x = np.random.randint(0, 2, size=n_bits)
        energy = x @ Q @ x
        best_bits = x.copy()
        best_energy = energy
        T = T_init

        for it in range(n_iterations):
            # Flip a random bit
            flip = np.random.randint(0, n_bits)
            x_new = x.copy()
            x_new[flip] = 1 - x_new[flip]
            new_energy = x_new @ Q @ x_new

            delta = new_energy - energy
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                x = x_new
                energy = new_energy

            if energy < best_energy:
                best_energy = energy
                best_bits = x.copy()

            T *= cooling

        all_runs.append({
            "run": run,
            "best_bits": best_bits.tolist(),
            "best_energy": float(best_energy),
        })

        if best_energy < best_global_energy:
            best_global_energy = best_energy
            best_global_bits = best_bits.copy()

    return best_global_bits, best_global_energy, all_runs


def bits_to_config(bits):
    lr_map = {(0, 0): 1e-5, (0, 1): 2e-5, (1, 0): 3e-5, (1, 1): 5e-5}
    dropout_map = {0: 0.1, 1: 0.3}
    batch_map = {0: 16, 1: 32}
    epoch_map = {0: 3, 1: 4}

    return {
        "learning_rate": lr_map[(bits[0], bits[1])],
        "dropout": dropout_map[bits[2]],
        "batch_size": batch_map[bits[3]],
        "epochs": epoch_map[bits[4]],
    }


def run_sa_for_track(track_name):
    sweep_path = f"outputs/metrics/qubo_sweep_{track_name}.json"
    with open(sweep_path, "r") as f:
        data = json.load(f)

    Q = np.array(data["qubo_matrix"])

    print(f"\n{'='*60}")
    print(f"Simulated Annealing for: {track_name}")
    print(f"{'='*60}")

    start = time.time()
    best_bits, best_energy, all_runs = simulated_annealing(Q)
    elapsed = time.time() - start

    config = bits_to_config(best_bits)

    print(f"\nBest bitstring: {best_bits.tolist()}")
    print(f"Best QUBO energy: {best_energy:.6f}")
    print(f"Config: {config}")
    print(f"Time: {elapsed:.2f}s")

    # Check consistency across runs
    unique_solutions = set()
    for r in all_runs:
        unique_solutions.add(tuple(r["best_bits"]))
    print(f"Unique solutions across 10 runs: {len(unique_solutions)}")
    for s in unique_solutions:
        count = sum(1 for r in all_runs if tuple(r["best_bits"]) == s)
        print(f"  {list(s)} -> found {count}/10 times")

    # Look up actual val_loss for this config
    for r in data["results"]:
        if r["bitstring"] == best_bits.tolist():
            print(f"Actual val_loss: {r['val_loss']:.4f}")
            print(f"Actual val_accuracy: {r['val_accuracy']:.4f}")
            break

    # Save
    result = {
        "track": track_name,
        "method": "simulated_annealing",
        "best_bitstring": best_bits.tolist(),
        "best_qubo_energy": best_energy,
        "config": config,
        "time_seconds": elapsed,
        "all_runs": all_runs,
    }

    save_path = f"outputs/metrics/sa_{track_name}.json"
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    run_sa_for_track("primary")
    run_sa_for_track("allagree")