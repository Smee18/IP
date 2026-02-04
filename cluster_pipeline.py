from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import adjusted_rand_score
import hdbscan
import pandas as pd
from sklearn.preprocessing import Normalizer
import umap
import matplotlib.pyplot as plt

label_indices = []
loaded_embeddings = []
label_info = []

target_classes = [0,1,2,3,4,5,6,7,8,9]
mask = np.isin(label_indices, target_classes)
required_indices = np.where(mask)[0]

def prepare_embedding(embeddings, is_scattering=False):
    if is_scattering:

        embeddings = np.log1p(embeddings)
    
    scaler = Normalizer(norm='l2')
    return scaler.fit_transform(embeddings)

def clustering_scorer(estimator, X, y):

    labels = estimator.fit_predict(X)
    return adjusted_rand_score(y, labels)

results = []
X_scaled = prepare_embedding(loaded_embeddings[required_indices], is_scattering=False)
y_true = label_indices[required_indices]

print("Starting Manual Parameter Sweep...")
for min_size in [25, 50, 75]:
    for min_samp in [1, 5, 10]:
        print(f"Trying min_size: {min_size}, min_samp: {min_samp}")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=min_samp)
        labels = clusterer.fit_predict(X_scaled)
        
        score = adjusted_rand_score(y_true, labels)
        noise_ratio = np.sum(labels == -1) / len(labels)
        
        results.append({
            'min_cluster_size': min_size,
            'min_samples': min_samp,
            'ARI': score,
            'noise': noise_ratio
        })

df_results = pd.DataFrame(results)
df_sorted = df_results.sort_values(by='ARI', ascending=False)
print("--- Top Performing Clusters ---")
print(df_sorted.head(5))

best_ari = df_sorted.iloc[0]['ARI']
current_dim = 2048 
efficiency = best_ari / np.sqrt(current_dim)

print(f"\nBest ARI: {best_ari:.4f}")
print(f"Model Dimension: {current_dim}")
print(f"Efficiency Score (ARI/√d): {efficiency:.6f}")

standard_embedding = umap.UMAP(random_state=42, min_dist=0.0, n_neighbors = 20).fit_transform(X_scaled)
plt.figure(figsize=(10, 8))


subset_labels = label_indices[required_indices]
cmap = plt.get_cmap('Spectral')
norm = plt.Normalize(vmin=min(target_classes), vmax=max(target_classes))

scatter = plt.scatter(
    standard_embedding[:, 0], 
    standard_embedding[:, 1], 
    c=subset_labels, 
    s=1.0, 
    cmap=cmap,
    norm=norm
)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', 
           label=label_info[cls],
           markerfacecolor=cmap(norm(cls)), 
           markersize=10) 
    for cls in target_classes
]

plt.legend(handles=legend_elements, loc='best', title="Galaxy Classes")
plt.title("UMAP Projection of Spiral Galaxy Features")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

plt.show()