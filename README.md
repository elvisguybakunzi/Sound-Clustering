# Clustering Unlabeled Sound Data

## Overview

This project focuses on clustering unlabeled sound data using dimensionality reduction techniques (PCA & t-SNE) and clustering algorithms (K-Means & DBSCAN). The goal is to explore high-dimensional data, extract meaningful features, and determine the most effective clustering method.

## Assignment

Complete the tasks in the notebook and document your observations in markdown cells.

## Libraries Used

```python
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
```

## Data Loading

The dataset consists of `.wav` files stored in a directory. The files are loaded and preprocessed using Librosa to extract Mel spectrogram features.

## Feature Extraction

Mel spectrograms are computed and converted to dB scale, then their mean values are taken to form feature vectors.

## Data Visualization

### Without Dimensionality Reduction

- 2D & 3D scatter plots showed overlapping clusters, making separation difficult.
- High-dimensional data complexity makes direct visualization challenging.

### With Dimensionality Reduction

- **PCA** (Principal Component Analysis) retains variance but struggles with non-linear patterns.
- **t-SNE** (t-distributed Stochastic Neighbor Embedding) preserves local relationships, leading to better-separated clusters.

## Why Dimensionality Reduction is Important

- **Avoids Cluster Overlap** by retaining essential variance.
- **Reduces Noise** by eliminating irrelevant features.
- **Improves Performance** by lowering computational cost.

## Clustering Techniques & Results

### **K-Means Clustering**

- Optimized using the **Elbow Method** (choosing k=3).
- Achieved **Silhouette Score: 0.2521** and **Davies-Bouldin Index: 1.4325**, indicating moderate cluster separation.

### **DBSCAN Clustering**

- Failed to form meaningful clusters, likely due to unsuitable density distribution.

## Final Visualization

- **t-SNE + K-Means** yielded the best visual separation of clusters.

## Conclusion

- **t-SNE outperformed PCA** in separating clusters by preserving local relationships.
- **K-Means was more effective** than DBSCAN due to the data’s density distribution.
- **Dimensionality reduction is crucial** for clustering high-dimensional sound data.

## Real-World Clustering Challenges

- High-dimensional data complicates clustering due to overlapping features.
- The choice of clustering method depends on data shape: K-Means works well for spherical clusters, while DBSCAN struggles with varying densities.
- **Dimensionality reduction (PCA/t-SNE) is essential** for extracting meaningful patterns.

