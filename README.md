# Spotify Music Clustering

Unsupervised machine learning project that groups songs by their Spotify audio features to uncover hidden structure in a music catalogue. The discovered segments can support music recommendation, playlist generation, and genre discovery.

## Project Overview

The dataset contains 10,000 tracks with nine numeric audio features:

- **popularity**
- **danceability** - how suitable a track is for dancing
- **valence** - the musical positiveness of a track
- **energy** - intensity and activity level of a song
- **key**
- **liveness** - detects the presence of a live audience
- **loudness** - the overall loudness of a track (dB)
- **speechiness** - the presence of spoken words in a track
- **tempo** - beats per minute (BPM)

## Workflow

1. **Load and explore** the dataset (`data/ML_spotify_data.csv`)
2. **Preprocess** - select nine audio features, drop duplicates (10,000 -> 9,998 rows), and standardise with `StandardScaler`
3. **Dimensionality reduction** with PCA for visualisation (PC1 + PC2 capture ~41% of the variance)
4. **Apply four clustering models:**
   - K-Means (elbow method + silhouette score, k = 4)
   - Hierarchical clustering (Ward linkage, dendrogram, k = 4)
   - DBSCAN (k-distance plot to tune `eps` and `min_samples`)
   - Gaussian Mixture Models (BIC / AIC and silhouette, g = 2)
5. **Evaluate** with silhouette scores, cluster profiles, PCA and t-SNE visualisations

## Results

### Model comparison

| Model | Clusters | Silhouette |
| --- | --- | --- |
| Hierarchical (Ward, k=4) | 4 | **0.172** |
| DBSCAN (eps=1.3) | 2 | 0.160 |
| K-Means (k=4) | 4 | 0.152 |
| GMM (g=2) | 2 | 0.101 |

### Final segments (hierarchical, Ward, k=4)

- **Cluster 0 - Mainstream (72%)** - near-average on every feature
- **Cluster 1 - Calm / acoustic (13%)** - lowest energy, quietest, slowest tempo
- **Cluster 2 - Spoken-word / rap (9%)** - highest speechiness and danceability, most popular tracks
- **Cluster 3 - Live recordings (6%)** - liveness near 0.78 with high energy

### Key takeaways

- All silhouette scores fall below 0.2, so audio features alone form overlapping, soft clusters rather than crisp groups.
- Distance-based methods (hierarchical and K-Means) perform best; DBSCAN and GMM underperform on this high-dimensional, near-uniform data.
- The four interpretable segments remain practical for playlist generation and recommendation.

## Getting Started

### Prerequisites

- Python 3.14+ with Jupyter
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`, `scipy`

### Usage

```bash
jupyter notebook Spotify_Music_Clustering.ipynb
```

The notebook runs all steps from data loading through to final cluster interpretation.

## Project Structure

```
spotify_music/
├── Spotify_Music_Clustering.ipynb   # Main analysis notebook
├── data/
│   └── ML_spotify_data.csv          # 10,000 tracks with audio features
└── README.md
```