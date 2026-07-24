import json
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, NELDER_MEAD
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import real_amplitudes, efficient_su2
import time


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


def run_vqe_for_track(track_name):
    sweep_path = f"outputs/metrics/qubo_sweep_{track_name}.json"
    with open(sweep_path, "r") as f:
        data = json.load(f)

    Q = np.array(data["qubo_matrix"])
    n_bits = Q.shape[0]

    print(f"\n{'='*60}")
    print(f"VQE for: {track_name}")
    print(f"{'='*60}")

    # Build QuadraticProgram
    qp = QuadraticProgram("hyperparameter_optimization")
    for i in range(n_bits):
        qp.binary_var(f"x{i}")

    linear = {f"x{i}": float(Q[i, i]) for i in range(n_bits)}
    quadratic = {}
    for i in range(n_bits):
        for j in range(i + 1, n_bits):
            if abs(Q[i, j]) > 1e-10:
                quadratic[(f"x{i}", f"x{j}")] = float(Q[i, j])

    qp.minimize(linear=linear, quadratic=quadratic)

    # Run SamplingVQE with different ansatze
    ansatze = [
        ("RealAmplitudes_2layers", real_amplitudes(n_bits, reps=2)),
        ("EfficientSU2_2layers", efficient_su2(n_bits, reps=2)),
    ]

    optimizers = [
        ("COBYLA", COBYLA(maxiter=200)),
        ("NELDER_MEAD", NELDER_MEAD(maxiter=200)),
    ]

    results_all = []

    for ansatz_name, ansatz in ansatze:
        for opt_name, optimizer in optimizers:
            print(f"\n--- {ansatz_name} + {opt_name} ---")
            start = time.time()

            sampler = StatevectorSampler()
            vqe = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=optimizer)
            vqe_optimizer = MinimumEigenOptimizer(vqe)

            result = vqe_optimizer.solve(qp)
            elapsed = time.time() - start

            bits = [int(result.variables_dict[f"x{i}"]) for i in range(n_bits)]
            energy = float(result.fval)
            config = bits_to_config(bits)

            print(f"  Bitstring: {bits}")
            print(f"  QUBO energy: {energy:.6f}")
            print(f"  Config: {config}")
            print(f"  Time: {elapsed:.2f}s")

            # Look up actual val_loss
            actual_loss = None
            actual_acc = None
            for r in data["results"]:
                if r["bitstring"] == bits:
                    actual_loss = r["val_loss"]
                    actual_acc = r["val_accuracy"]
                    break

            if actual_loss is not None:
                print(f"  Actual val_loss: {actual_loss:.4f}")
                print(f"  Actual val_accuracy: {actual_acc:.4f}")

            results_all.append({
                "ansatz": ansatz_name,
                "optimizer": opt_name,
                "bitstring": bits,
                "qubo_energy": energy,
                "config": config,
                "time_seconds": elapsed,
                "val_loss": actual_loss,
                "val_accuracy": actual_acc,
            })

    # Use paper-aligned config as primary: RealAmplitudes + Nelder-Mead
    primary_result = None
    for r in results_all:
        if r["ansatz"] == "RealAmplitudes_2layers" and r["optimizer"] == "NELDER_MEAD":
            primary_result = r
            break
    if primary_result is None:
        primary_result = results_all[0]

    output = {
        "track": track_name,
        "method": "VQE",
        "primary_result": primary_result,
        "all_results": results_all,
    }

    save_path = f"outputs/metrics/vqe_{track_name}.json"
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    run_vqe_for_track("primary")
    run_vqe_for_track("allagree")