import numpy as np
from sklearn.metrics import adjusted_rand_score
import hdbscan
import pandas as pd
from sklearn.preprocessing import Normalizer
import umap
import matplotlib.pyplot as plt
import h5py
from sklearn.metrics import silhouette_score
import seaborn as sns
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import ColumnDataSource, HoverTool
from io import BytesIO
from PIL import Image
import base64
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10


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
    for min_size in [100, 200, 300]:
        for min_samp in [10, 20, 30]:
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

def plot_confusion(confusion_matrix, class_names):
    plt.figure(figsize=(15, 15))
    sns.heatmap(confusion_matrix, annot=True, cmap='viridis', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Morphological Separability Map (Silhouette Score)")
    plt.xlabel("Class B")
    plt.ylabel("Class A")
    plt.savefig('figures/umap_confusion.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
def array_to_base64(img_array):
    """Converts a numpy image array to a base64 string with proper scaling."""
    # 1. Handle dynamic range: Scale to 0-255
    img_min = img_array.min()
    img_max = img_array.max()
    
    # Avoid division by zero for empty images
    if img_max > img_min:
        scaled_img = (img_array - img_min) / (img_max - img_min) * 255
    else:
        scaled_img = img_array * 0
        
    # 2. Convert to uint8 for PIL
    img = Image.fromarray(scaled_img.astype('uint8'))
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

def save_interactive_umap(X_manifold, y_true, images, filename="umap_explorer.html"):
    # Convert labels to strings for categorical mapping
    label_strings = y_true.astype(str)
    unique_labels = sorted(list(set(label_strings)))
    
    df = pd.DataFrame({
        'x': X_manifold[:, 0],
        'y': X_manifold[:, 1],
        'label': label_strings,
        'image': [array_to_base64(img) for img in images]
    })

    source = ColumnDataSource(df)

    p = figure(title="Galaxy Morphology Continuum Explorer", 
               tools="pan,wheel_zoom,reset,save",
               width=900, height=700)

    # FIX: Use factor_cmap to map labels to colors
    mapper = factor_cmap(field_name='label', 
                         palette=Category10[10] if len(unique_labels) > 2 else Category10[3], 
                         factors=unique_labels)

    p.scatter('x', 'y', source=source, color=mapper, # Use the mapper here
              fill_alpha=0.6, size=5, line_color=None, legend_field='label')

    # 3. Add Hover Tool with Image HTML
    hover = HoverTool(tooltips="""
        <div>
            <div>
                <img src="@image" style="float: left; margin: 0px 15px 15px 0px; border: 2px solid #333;" width="128"/>
            </div>
            <div>
                <span style="font-size: 15px; font-weight: bold;">Class: @label</span>
            </div>
        </div>
    """)
    p.add_tools(hover)

    # 4. Save as standalone HTML
    output_file(f"{filename}18")
    save(p)
    print(f"Interactive manifold saved to {filename}")
    

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
    dim = 264
    return rm_embeddings, dim, True


### MAIN CODE ###

data_path = 'data/Galaxy10_ProcessedandCropped.h5'


loaded_embeddings, dim, scatter_bool = load_rigid_motion()
confusion_matrix = np.zeros((10, 10))
num_classes = 10
"""
for i in range(num_classes):
    for j in range(num_classes):
        print(f"Testing {i}-{j}")
        if i >= j: 
            continue 
            
        target_classes = [i, j]
        mask = np.isin(label_indices, target_classes)
        required_indices = np.where(mask)[0]
        
        y_ground_truth = label_indices[required_indices]
        X_prepared = prepare_embedding(loaded_embeddings[required_indices], is_scattering=scatter_bool)

        # UMAP step
        reducer = umap.UMAP(random_state=42, min_dist=0.0, n_neighbors=30, metric='correlation')
        X_manifold = reducer.fit_transform(X_prepared)

        if len(np.unique(y_ground_truth)) > 1:
            score = silhouette_score(X_manifold, y_ground_truth)
            normalized_score = (score + 1) / 2 
            confusion_matrix[i, j] = normalized_score
            confusion_matrix[j, i] = normalized_score
    
class_names= ["Disturbed","Merging","Round Smooth","In-between Round Smooth","Cigar Shaped Smooth","Barred Spiral","Unbarred Tight Spiral","Unbarred Loose Spiral","Edge-on without Bulge","Edge-on with Bulge"]
plot_confusion(confusion_matrix, class_names)
"""

target_classes = [1,8]
with h5py.File(data_path, 'r') as F:
    label_indices = np.array(F['ans']) # Copy to memory
    mask = np.isin(label_indices, target_classes)
    required_indices = np.where(mask)[0]
    # Ensure this is a numpy array in memory, not a H5 dataset pointer
    subset_images = np.array(F['images'][required_indices])


y_ground_truth = label_indices[required_indices]
X_prepared = prepare_embedding(loaded_embeddings[required_indices], is_scattering=scatter_bool)

# UMAP step
reducer = umap.UMAP(random_state=42, min_dist=0.0, n_neighbors=30, metric='correlation')
X_manifold = reducer.fit_transform(X_prepared)
save_interactive_umap(X_manifold, y_ground_truth, subset_images)

#plt.scatter(X_manifold[:, 0], X_manifold[:, 1], c=y_ground_truth, s=2.0, alpha=0.6, cmap='Spectral')
#plt.show()