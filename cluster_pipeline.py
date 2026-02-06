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
    best_labels = None
    max_ari = -np.inf

    print("Starting Manual Parameter Sweep...")
    for min_size in [100]:
        for min_samp in [20]:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=min_samp)
            labels = clusterer.fit_predict(X_data)
            
            score = adjusted_rand_score(y_true, labels)
            noise_ratio = np.sum(labels == -1) / len(labels)
            
            results.append({
                'min_cluster_size': min_size,
                'min_samples': min_samp,
                'ARI': score,
                'noise': noise_ratio,
                'labels': labels # Store labels to retrieve best ones
            })

            if score > max_ari:
                max_ari = score
                best_labels = labels

    df_results = pd.DataFrame(results)
    df_sorted = df_results.sort_values(by='ARI', ascending=False)
    
    print(f"\nBest ARI: {df_sorted.iloc[0]['ARI']:.4f}")
    
    # Return both the dataframe and the labels of the best run
    return df_sorted, best_labels

def plot_comparison(manifold_coords, y_true, y_pred, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Ground Truth
    scatter1 = ax1.scatter(manifold_coords[:, 0], manifold_coords[:, 1], 
                          c=y_true, s=2.0, alpha=0.6, cmap='Spectral')
    ax1.set_title(f"{model_name}: Ground Truth")
    plt.colorbar(scatter1, ax=ax1)

    # Plot 2: HDBSCAN Predictions
    # Note: c=y_pred will show clusters; -1 is noise (usually dark/grey)
    scatter2 = ax2.scatter(manifold_coords[:, 0], manifold_coords[:, 1], 
                          c=y_pred, s=2.0, alpha=0.6, cmap='tab20')
    ax2.set_title(f"{model_name}: HDBSCAN Clusters (Best ARI)")
    plt.colorbar(scatter2, ax=ax2)

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

loaded_embeddings, dim, scatter_bool = load_rigid_motion()

target_classes = [5,6]
mask = np.isin(label_indices, target_classes)
required_indices = np.where(mask)[0]
y_ground_truth = label_indices[required_indices]
results = []

X_prepared = prepare_embedding(loaded_embeddings[required_indices], is_scattering=scatter_bool)

print("Computing UMAP manifold...")
reducer = umap.UMAP(random_state=42, min_dist=0.0, n_neighbors=100, metric='correlation')
X_manifold = reducer.fit_transform(X_prepared)

# Run search and get best labels
results_df, best_hdbscan_labels = perform_grid_search(X_manifold, y_ground_truth, dim)

# Plot the comparison
plot_comparison(X_manifold, y_ground_truth, best_hdbscan_labels, "RM Scattering")