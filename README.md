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
* **Robustness against Noise**: Single-cell data is inherently sparse and noisy. Instead of conventional central moments (e.g., Pearson's skewness) which are susceptible to outliers, L-Trail utilizes L-moments—linear combinations of order statistics. This provides a robust estimation of directional biases, mitigating the impact of technical noise.

## Scope and Limitations
When applying L-Trail, consider the following algorithmic and structural constraints:

* **Sample Size Dependency**: L-moment estimation relies on sample statistics; therefore, larger cluster sizes yield more stable vectors. While a minimum of 30 cells is often used as a practical threshold, there is no strict mathematical minimum. Small clusters may produce unreliable directional vectors.
* **Clustering Resolution**: The algorithm evaluates the geometric shape of an entire cell cluster. Over-clustering or the use of algorithms that impose artificial boundaries can fragment the continuous "comet tail" distribution, altering the structural asymmetry and degrading estimation accuracy.
* **Linear Space Requirement**: Because L-Trail computes directional vectors using linear combinations of moments, calculations must be performed within a linear high-dimensional space (default: top 30 principal components). Computing vectors directly within non-linear embeddings (e.g., UMAP or t-SNE) introduces spatial distortions and is mathematically invalid.
* **Continuous Manifold Assumption**: The geometric inference assumes the presence of an underlying continuous data manifold. The algorithm estimates macroscopic trends within connected, progressing populations and cannot infer developmental trajectories across discrete or disconnected cell states.

## Installation

Currently, L-Trail can be installed by cloning this repository.

```bash
git clone [https://github.com/jamagiwa/L-Trail.git](https://github.com/jamagiwa/L-Trail.git)
cd L-Trail
pip install -r requirements.txt
```

## Dependencies
Python >= 3.8
numpy, pandas, scipy, scikit-learn, scanpy, anndata, scvelo, leidenalg

## Quick Start & Tutorial
We provide a Jupyter Notebook to demonstrate the basic usage of L-Trail, from data preprocessing to trajectory inference and visualization.
[Interactive Colab Tutorial] You can run the demonstration of the Pancreas dataset directly in your browser without any local setup:

Basic Usage Example
```python
import scanpy as sc
import sys
# Adjust to your cloned directory
sys.path.append('/path/to/L-Trail') 
from ltrail import tl, pl

# Load your AnnData object (assuming standard preprocessing and PCA/UMAP are already computed)
adata = sc.read_h5ad("your_data.h5ad")

# Calculate L-Trail similarities in high-dimensional space
df_results = tl.calc_knn_similarity(
    adata=adata,
    groupby='clusters',
    use_rep='X_pca',
    method='lmoment'
)

# Visualize vectors on embeddings
pl.plot_ltrail(
    adata,
    groupby='clusters',
    basis='X_umap',
    use_rep='X_pca',
    method='lmoment'
)
```

## Repository Structure
ltrail/tl.py: Core algorithms for L-moment computation and vector estimation.

ltrail/pl.py: Functions for visualizing L-Trail vectors on 2D embeddings (e.g., PCA, UMAP).

## Licence
This project is licensed under the MIT License - see the LICENSE file for details.
