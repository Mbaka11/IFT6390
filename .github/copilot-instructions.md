# Copilot Instructions — IFT6390: Introduction à l'apprentissage automatique

## Context

This is a university course repo for **IFT6390** (Introduction to Machine Learning) at **Université de Montréal**, part of the **Maitrise en apprentissage automatique (Mila)**. It contains course notes, TPs (labs), and graded projects.

## Language & Communication

- The course and assignments are in **French**. All markdown cells, comments in notebooks, and analysis text should be written in **French**.
- Code (variable names, function names, docstrings) can be in **English** for readability, but inline comments explaining ML concepts should be in **French** when they appear in deliverable notebooks.

## Repo Structure

```
Course_content/Markdown_content/  → Course notes (ch1–ch10), used as reference
Exams/                            → Exam material
TP_and_projects/TP/               → Lab notebooks (tp0–tp5)
TP_and_projects/Project_1/        → Completed: energy prediction (Ridge, KNN)
TP_and_projects/Project_2/        → Current: neural network project (molecules → Tc)
```

## Tech Stack & Environment

- **Python 3.10+** with Jupyter notebooks (designed for **Google Colab**)
- Core dependencies: `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- Project 2 additionally requires: `torch` (PyTorch), `rdkit`
- Pre-trained model: **SMI-TED** (SMILES-based molecular embeddings, 768-dim)
- Always use **reproducible seeds** (`SEED = 42` unless specified otherwise)

## Current Project: Devoir 2 — Prédiction de la température critique

### Objective
Predict the **critical temperature (Tc)** of molecules from their **SMILES** string representation using neural networks.

### Dataset
- Source: `chedl_thermo_properties.csv` (~24,600 molecules, ~13,100 with Tc)
- Train/Val split: 80/20 with `random_state=42`
- Target: `Tc` (critical temperature in Kelvin)
- Input: `SMILES` strings (text encoding of molecular structure)

### Project Structure (3 parts)
1. **Part 1 — MLP & Optimization**: Fixed character-frequency features → MLP, optimizer comparison (SGD, Momentum, Adam), deep MLP with activation/init/batchnorm/dropout ablation
2. **Part 2 — Sequential & Attention Models**: LSTM on character sequences, Transformer encoder with sinusoidal positional encoding, model comparison table
3. **Part 3 — Transfer Learning**: SMI-TED pre-trained embeddings, linear probe, sample efficiency curves across all 4 models

### Key Constraints
- Use **MSE** as the loss function and primary evaluation metric
- Also report **R²** for model comparison
- For each model, track: MSE val, R² val, parameter count, train-val gap
- Use **PyTorch** for all neural network implementations (no keras/tensorflow)
- Handle **variable-length SMILES** with proper padding and masking
- The notebook must be runnable end-to-end on Colab

## Coding Conventions

- Write clean, well-structured code with clear cell separation in notebooks
- Each experiment should produce **reproducible results** (set seeds for torch, numpy, random)
- Plotting: use `matplotlib`, label axes in French, include legends and titles
- When comparing models, use consistent training settings (epochs, batch size, learning rate) unless the experiment specifically varies them
- Follow the progression: predict before running → run → analyze results
- Keep training loops clean with proper train/eval mode switching in PyTorch

## Course Content Reference

When explaining ML concepts, align with the course material in `Course_content/Markdown_content/`:
- Neural networks: `ch7_neural_networks.md`
- Optimization & training: `ch8_optimization.md`
- ConvNets, RNNs, Autoencoders: `ch9_*.md`
- Transformers & attention: `ch10_transformers.md`
- Probabilistic models: `ch5_probabilistic.md`, `ch6_probabilistic_models.md`

## Lessons from Project 1

- Progress systematically from simple → complex models
- Watch for data leakage (don't use target-correlated columns as features)
- Carefully compare train vs. val distributions
- Document improvements and rationale at each step
