from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import adjusted_rand_score
import hdbscan
import pandas as pd
from sklearn.preprocessing import Normalizer
import umap
import matplotlib.pyplot as plt
import h5py


def prepare_embedding(embeddings, is_scattering=False):
    if is_scattering:

        embeddings = np.log1p(embeddings)
    
    scaler = Normalizer(norm='l2')
    return scaler.fit_transform(embeddings)

def clustering_scorer(estimator, X, y):

    labels = estimator.fit_predict(X)
    return adjusted_rand_score(y, labels)

def perform_grid_search(X_data, y_true, dim):
    results = []

    print("Starting Manual Parameter Sweep...")
    for min_size in [25, 50, 75]:
        for min_samp in [1, 5, 10]:
            print(f"Trying min_size: {min_size}, min_samp: {min_samp}")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=min_samp)
            labels = clusterer.fit_predict(X_data)
            
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
    efficiency = best_ari / np.sqrt(dim)

    print(f"\nBest ARI: {best_ari:.4f}")
    print(f"Model Dimension: {dim}")
    print(f"Efficiency Score (ARI/√d): {efficiency:.6f}")

    return df_sorted
    

def plot_umap(X_data, target_classes):

    standard_embedding = umap.UMAP(random_state=42, min_dist=0.1, n_neighbors = 10).fit_transform(X_data)
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
            label=f"Class {cls}",
            markerfacecolor=cmap(norm(cls)), 
            markersize=10) 
        for cls in target_classes
    ]

    plt.legend(handles=legend_elements, loc='best', title="Galaxy Classes")
    plt.title("UMAP Projection of Spiral Galaxy Features")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    plt.show()

def load_cnn():
    output_filename = r"maps/cnn_embedding_gray.npz"
    print("Loading saved features...")
    data = np.load(output_filename)
    cnn_embeddings = data['cnn_embeddings']
    dim = 2048
    return cnn_embeddings, dim, False

def load_kymatio():
    output_filename = r"maps/kymatio_embedding_gray.npz"
    print("Loading saved features...")
    data = np.load(output_filename)
    kymatio_embeddings = data['kymatio_embeddings']
    dim = 611
    return kymatio_embeddings, dim, True

def load_rigid_motion():
    output_filename = r"maps/rigid_motion_embedding_gray.npz"
    print("Loading saved features...")
    data = np.load(output_filename)
    rm_embeddings = data['rm_embeddings']
    dim = 1664
    return rm_embeddings, dim, True


### MAIN CODE ###

data_path = 'data/Galaxy10_DECals.h5'

with h5py.File(data_path, 'r') as F:
    label_indices = np.array(F['ans'])

loaded_embeddings, dim, scatter_bool = load_cnn()

target_classes = [0,1,2,3,4,5,6,7,8,9]
mask = np.isin(label_indices, target_classes)
required_indices = np.where(mask)[0]
y_ground_truth = label_indices[required_indices]

results = []
X_scaled = prepare_embedding(loaded_embeddings[required_indices], is_scattering=scatter_bool)
results = perform_grid_search(X_scaled, y_ground_truth, dim)
plot_umap(X_scaled, target_classes)