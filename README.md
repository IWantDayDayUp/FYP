# ECG Foundation Model

This project focuses on building a **foundation model for ECG signals** using **Masked Autoencoding (MAE)** pre-training.

The model learns universal ECG representations from multiple datasets and can be transferred to downstream tasks such as **arrhythmia classification**.

---

## ✨ Project Goals

- Pre-train a generalizable ECG representation model
- Train on multiple PhysioNet databases
- Evaluate on downstream classification tasks
- Visualize reconstruction quality and learned features
- Provide a reproducible training pipeline for HPC and local environments

---

## 📂 Project Structure

```bash
├─ src/ecg_fm/ # Source code (Python package)
│ ├─ data/ # Dataset classes and preprocessing
│ ├─ models/ # MAE and downstream models
│ ├─ training/ # Training loops and pipelines
│ └─ utils/ # Logging, environment helpers, etc.
├─ notebooks/ # Interactive analysis and visualization
├─ scripts/ # SBATCH / helper scripts for HPC
├─ configs/ # Training configuration files
├─ train_mae.py # Entry point for single-DB MAE training
├─ .gitignore
└─ README.md
```

> Note: All raw ECG datasets and training logs are excluded from GitHub.
> See `data/README.md` for dataset installation instructions.

---

## 🚀 Quick Start

### HPC Training Example

```bash
sbatch --partition=gpu scripts/job_train_mae.sh
```

## 📊 Results Tracking

- `train.log` – training logs
- `metrics.csv` – epoch loss and performance
- `best.pt` & `last.pt` – checkpoints for model evaluation
- `summary.json` – run configuration and environment info

Visualizations and evaluation notebooks are in `notebooks/`.

## 🧱 Roadmap

- Single-dataset MAE pretraining
- Multi-dataset MAE (PhysioNet)
- Downstream arrhythmia classification
- Latent feature visualization
- Performance benchmarking across datasets
- Model deployment and inferencing interface

## ✦ Notes

This repository is under active development.

Contributions and feedback are welcome after publication of project results.
