# L-Trail: Estimating macroscopic transition directions in scRNA-seq data via outlier-robust L-moments

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jamagiwa/L-Trail/blob/main/tutorials/Pancreas_dataset.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img width="541" height="350" alt="Figure 1" src="https://github.com/user-attachments/assets/26c0ce30-8643-4233-8767-4c34ebdbee8b" />
</p>

**L-Trail** is a statistical toolkit designed to infer the macroscopic transition trends of cell clusters in single-cell RNA sequencing (scRNA-seq) data.

Rather than modeling single-cell kinetics, L-Trail captures the distributional asymmetry (the "comet tail" effect) of cell populations within high-dimensional spaces (e.g., PCA) using L-moments. This provides a robust estimation of cluster-level trajectories based purely on static expression states.

## Key Features
* **Macroscopic Trajectory Inference**: Unlike RNA velocity, which models single-cell kinetics, L-Trail focuses on the macroscopic dynamics of cell populations. It infers the directional transition vectors for entire cell clusters based on their spatial distribution.
* **Splicing-Independent**: Because it relies entirely on the geometric asymmetry of the data (the "comet tail" effect) rather than mRNA splicing kinetics, L-Trail can be applied directly to standard, static gene expression matrices without the need for unspliced reads.
* **Robustness against Noise**: Single-cell data is inherently sparse and noisy. Instead of conventional central moments (e.g., Pearson's skewness) which are highly sensitive to outliers, L-Trail utilizes L-moments—linear combinations of order statistics. This provides a highly robust estimation of directional biases, effectively filtering out technical noise.

## Installation

Currently, L-Trail can be installed by cloning this repository.

```bash
!git clone [https://github.com/jamagiwa/L-Trail.git](https://github.com/jamagiwa/L-Trail.git)
%cd L-Trail
!pip install -r requirements.txt
```
