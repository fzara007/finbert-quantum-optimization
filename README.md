# FinBERT Quantum Optimization

Hyperparameter tuning for FinBERT financial sentiment classification via QUBO formulation,
solved with Simulated Annealing, QAOA, and VQE.

## Setup

Requires Python 3.13+ and a Linux environment (WSL or native).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data

1. Place `all-data.csv` in `data/raw/`
2. Place `FinancialPhraseBank-v1.0/` folder (containing `Sentences_AllAgree.txt`) in `data/raw/`

## Execution Order

Run all scripts from the project root (`~/project/`):

```bash
# 1. Prepare data (both tracks)
python src/prepare_data.py

# 2. Run 32-combination hyperparameter sweep (GPU recommended)
python src/build_qubo.py

# 3. Run optimizers on the QUBO matrix
python src/run_sa.py
python src/run_qaoa.py
python src/run_vqe.py

# 4. Train and evaluate final models
python src/evaluate_optimized.py
```

## Project Structure
project/
data/
raw/ # Original datasets
processed/ # Cleaned train/val/test splits
src/ # All pipeline scripts
models/ # Saved model checkpoints
outputs/
metrics/ # JSON results for all experiments
plots/ # Loss curve plots

## Configuration

- Base checkpoint: `ahmedrachid/FinancialBERT`
- Tokenizer: `ahmedrachid/FinancialBERT`, max_length=64
- Random seed: 42 (data split, PyTorch, NumPy)
- Data split: 70/15/15 stratified
- Label mapping: negative=0, neutral=1, positive=2
- Quantum execution: Local noiseless simulation (Qiskit Aer)

## Tracks

- **Primary CSV**: 4,846 financial news sentences (all-data.csv)
- **AllAgree**: 2,264 sentences from Financial PhraseBank, 100% annotator agreement subset