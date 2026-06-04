# SAE Feature Similarity

Investigating feature stability in Sparse Autoencoders (SAEs) by comparing learned representations across different random seeds.

## Overview

Sparse Autoencoders (SAEs) have emerged as a powerful tool for interpreting internal representations of large language models. A fundamental question is whether SAEs trained on the same activations but initialized with different random seeds learn the same underlying features.

This project explores the stability and reproducibility of SAE features by:

* Training multiple SAEs on identical GPT-2 activations.
* Comparing learned feature dictionaries across random seeds.
* Measuring alignment using cosine similarity.
* Analyzing feature recovery and matching behavior.

---

## Research Question

> If two Sparse Autoencoders are trained on the same dataset and activations but initialized differently, do they discover the same features?

Understanding feature stability is important for:

* Mechanistic Interpretability
* Feature Discovery
* Representation Learning
* Reproducibility of SAE-based analyses

---

## Experimental Setup

### Language Model

* GPT-2 (TransformerLens)

### Dataset

* TinyStories

### Activation Source

* Residual stream activations

### SAE Configuration

| Parameter                 | Value |
| ------------------------- | ----- |
| Input Dimension (`d_in`)  | 768   |
| Hidden Features (`d_sae`) | 3072  |
| Device                    | CUDA  |
| L1 Coefficient            | 0.001 |

---

## Training Runs

### SAE #1

* Random Seed: 42
* Final Reconstruction MSE: **0.00656**

### SAE #2

* Random Seed: 123
* Final Reconstruction MSE: **0.00894**

Both SAEs successfully converged and achieved low reconstruction error.

---

## Similarity Analysis

### Global Encoder Similarity

Encoder matrices were flattened and compared using cosine similarity.

| Metric                   | Value      |
| ------------------------ | ---------- |
| Global Cosine Similarity | **0.2015** |

A low global similarity suggests that feature ordering and exact weight placement differ significantly across runs.

---

### Feature-Level Matching

A feature-to-feature cosine similarity matrix was computed between the two encoder dictionaries.

For each feature in SAE #1, the most similar feature in SAE #2 was identified.

| Metric                 | Value      |
| ---------------------- | ---------- |
| Mean Best Similarity   | **0.7189** |
| Median Best Similarity | **0.8043** |
| Maximum Similarity     | **0.9474** |

---

## Key Findings

Although the encoder matrices appear quite different globally, many individual features are highly similar after matching.

Observations:

* Numerous features achieve cosine similarity above **0.9**.
* Several features are recovered almost identically across seeds.
* Feature ordering is not preserved between runs.
* Some features appear unstable and are not consistently recovered.

These preliminary results suggest that many SAE features are robust to random initialization.

---

## Visualizations

### Training Loss

The SAE reconstruction loss rapidly decreased during training and converged to a low final error.

### Feature Matching Histogram

A histogram of best-match cosine similarities revealed:

* A large concentration of highly similar features (0.8–0.95).
* A smaller group of poorly matched features.
* Evidence of both stable and unstable learned representations.

---

## Repository Structure

```text
.
├── notebooks/
│   └── train_sae_colab_v2.ipynb

├── results/
│   ├── w_enc_seed42.pt
│   └── w_enc_seed123.pt

├── src/

├── .gitignore

└── README.md
```

---

## Current Status

Completed:

* GPT-2 activation extraction
* SAE training pipeline
* Training with two random seeds
* Reconstruction evaluation
* Encoder weight export
* Global cosine similarity analysis
* Feature matching analysis
* Similarity histogram generation

In Progress:

* Multi-seed experiments
* Pairwise similarity matrix construction
* Heatmap visualization
* Feature stability statistics

---

## Future Work

* Train additional seeds (999, 2025, 7777, ...)
* Compare all seed pairs
* Build similarity heatmaps
* Implement CKA similarity analysis
* Study feature recovery rates
* Analyze highly stable features
* Scale experiments to larger datasets

---

## Tools & Libraries

* Python
* PyTorch
* SAE Lens
* TransformerLens
* Hugging Face Datasets
* Google Colab
* Matplotlib

---

## References

* Anthropic: Towards Monosemanticity
* SAE Lens
* TransformerLens
* TinyStories Dataset


