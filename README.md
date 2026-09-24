# Spotify Music Clustering

Unsupervised machine learning project that groups songs by their Spotify audio features to uncover hidden structure in a music catalogue. The discovered segments can support music recommendation, playlist generation, and genre discovery.

## Project Overview

The dataset contains 10,000 tracks. The project description lists nine audio features, but this particular dataset export does **not** include `acousticness` or `instrumentalness`, so clustering uses the numeric features that are actually present:

- **popularity**
- **danceability** - how suitable a track is for dancing
- **valence** - the musical positiveness of a track
- **energy** - intensity and activity level of a song
- **key**
- **liveness** - detects the presence of a live audience
- **loudness** - the overall loudness of a track (dB)
- **speechiness** - the presence of spoken words in a track
- **tempo** - beats per minute (BPM)

The `explicit` column is present but deliberately excluded from clustering because it is a binary flag rather than an audio-quality measure and can dominate Euclidean distances; it is only used later as a descriptor.

## Workflow

1. **Load and explore** the dataset (`data/ML_spotify_data.csv`)
2. **Preprocess** - select available numeric features, drop duplicates (10,000 -> 9,998 rows), and standardise with `StandardScaler`
3. **Dimensionality reduction** with PCA for visualisation (PC1 + PC2 capture ~41% of the variance)
4. **Apply four clustering models:**
   - K-Means (elbow method + silhouette score, k = 4)
   - Hierarchical clustering (Ward linkage, dendrogram, k = 4)
   - DBSCAN (k-distance plot to tune `eps` and `min_samples`)
   - Gaussian Mixture Models (BIC / AIC and silhouette, g = 2)
5. **Evaluate** with silhouette scores, cluster profiles, PCA and t-SNE visualisations

## Results

### Model comparison

Comparison of the **selected configurations** (the metric is not the sole selection criterion):

| Model | Clusters | Silhouette |
| --- | --- | --- |
| Hierarchical (Ward, k=4) | 4 | **0.172** |
| DBSCAN (eps=1.3) | 2 | 0.160 |
| K-Means (k=4) | 4 | 0.152 |
| GMM (g=2) | 2 | 0.101 |

If every configuration were ranked by score alone, Hierarchical k = 2 (0.189) and K-Means k = 2 (0.173) would win; they were rejected because a two-cluster split is too coarse to be useful. k = 4 is chosen for its interpretable, actionable segments while staying within ~0.02 of the peak silhouette.

### Final segments (hierarchical, Ward, k=4)

- **Cluster 0 - Mainstream (72%)** - near-average on every feature
- **Cluster 1 - Low-energy / mellow (13%)** - lowest energy, quietest, slowest tempo (described cautiously: no `acousticness` feature exists to justify an "acoustic" label)
- **Cluster 2 - Spoken-word / rap (9%)** - highest speechiness and danceability, most popular tracks
- **Cluster 3 - Live / concert recordings (6%)** - liveness near 0.78 with high energy

### How the segments can be used

- **Playlist generation:** each cluster is a ready-made seed list ("mellow study soundtrack", "focus-friendly spoken word", "energetic live sets") that can be expanded automatically from cluster assignments.
- **Recommendations:** a recommender can score a user's recent listens, find their dominant cluster, and surface nearby tracks within it, generalising to tracks the user has never seen.
- **Catalogue organisation:** clusters give editorial staff tags, campaign targets (e.g. promoting the low-energy cluster during study weeks), and visibility into underserved niches.

### Key takeaways

- All silhouette scores fall below 0.2, so audio features alone form overlapping, soft clusters rather than crisp groups.
- Distance-based methods (hierarchical and K-Means) perform best; DBSCAN and GMM underperform on this high-dimensional, near-uniform data.
- Model selection balances the metric against utility: the highest-scoring k = 2 configurations were rejected as too coarse, and k = 4 was kept because its four segments are nameable, stable and actionable.
- The four interpretable segments remain practical for playlist generation, recommendation, and catalogue organisation.

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