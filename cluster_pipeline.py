import numpy as np
from sklearn.metrics import adjusted_rand_score
import hdbscan
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.decomposition import PCA
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
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output

J = 3
L= 12


def prepare_embedding(embeddings, is_scattering=False):
    if is_scattering:
        embeddings = np.log1p(embeddings)
    
    s1_s2_data = embeddings[:, 1:] 
    s1_end = J * L
    
    s1_data = s1_s2_data[:, :s1_end]
    s2_data = s1_s2_data[:, s1_end:]

    scaler = StandardScaler()
    s1_norm = scaler.fit_transform(s1_data)
    s2_norm = scaler.fit_transform(s2_data)

    return np.hstack([s1_norm, s2_norm])

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

def save_interactive_3d_umap(X_manifold, y_true, images, filename="umap_3d_explorer.html"):
    """Generates a 3D interactive UMAP visualization using Plotly."""
    
    # Convert labels to strings for the legend
    label_strings = y_true.astype(str)
    
    # Create DataFrame for Plotly
    df = pd.DataFrame({
        'UMAP 1': X_manifold[:, 0],
        'UMAP 2': X_manifold[:, 1],
        'UMAP 3': X_manifold[:, 2],
        'Class': label_strings,
        'image_base64': [array_to_base64(img) for img in images]
    })

    # Create the 3D Scatter plot
    fig = px.scatter_3d(
        df, 
        x='UMAP 1', y='UMAP 2', z='UMAP 3',
        color='Class',
        title="Galaxy Morphology 3D Continuum Explorer",
        opacity=0.7,
        template="plotly_dark"  # Dark theme often looks better for space data
    )

    # Update markers and add custom hover data
    fig.update_traces(
        marker=dict(size=3),
        hovertemplate="<b>Class: %{customdata[0]}</b><br><img src='%{customdata[1]}' width='128'><extra></extra>",
        customdata=df[['Class', 'image_base64']].values
    )

    # Save to HTML
    fig.write_html(filename)
    print(f"3D Interactive manifold saved to {filename}")

def run_dash_explorer(X_manifold, y_true, images):
    app = dash.Dash(__name__)

    local_indices = np.arange(len(y_true))

    # 1. Prepare Data
    df = pd.DataFrame({
        'UMAP 1': X_manifold[:, 0],
        'UMAP 2': X_manifold[:, 1],
        'UMAP 3': X_manifold[:, 2],
        'Class': y_true.astype(str),
        'point_index': np.arange(len(y_true)) 
    })

    # Light Mode Figure
    fig = px.scatter_3d(
        df, x='UMAP 1', y='UMAP 2', z='UMAP 3',
        color='Class', 
        opacity=0.8, 
        template="plotly_white", # Changed to white template
        height=800,
        custom_data=['point_index', 'Class']
    )
    
    fig.update_traces(
        marker=dict(size=4, line=dict(width=1, color='DarkSlateGrey')),
        hoverinfo='none', 
        hovertemplate=None
    )

    # 2. Layout (Clean White Aesthetic)
    app.layout = html.Div([
        # Left Panel: The Graph
        html.Div([
            dcc.Graph(id='3d-scatter', figure=fig, clear_on_unhover=False)
        ], style={'width': '70%', 'display': 'inline-block', 'padding': '10px'}),
        
        # Right Panel: The Inspector
        html.Div([
            html.H2("Galaxy Morphology Inspector", 
                    style={'color': '#2c3e50', 'fontFamily': 'serif', 'borderBottom': '2px solid #eee'}),
            html.Br(),
            html.Div(id='image-container', children=[
                html.P("Hover over a data point to inspect morphology", 
                       style={'color': '#7f8c8d', 'fontStyle': 'italic'})
            ]),
            html.Div(id='class-label', style={
                'fontSize': '22px', 
                'fontWeight': 'bold', 
                'marginTop': '20px',
                'color': '#2980b9',
                'fontFamily': 'serif'
            }),
            html.Hr(),
            html.P("Dissertation Sample View", style={'fontSize': '12px', 'color': '#bdc3c7'})
        ], style={
            'width': '25%', 
            'display': 'inline-block', 
            'vertical-align': 'top', 
            'padding': '30px', 
            'backgroundColor': '#f9f9f9', # Soft light grey background
            'height': '100vh',
            'boxShadow': '-2px 0px 5px rgba(0,0,0,0.05)'
        })
    ], style={'backgroundColor': 'white', 'display': 'flex', 'fontFamily': 'Arial'})

    # 3. Callback
    @app.callback(
        [Output('image-container', 'children'),
         Output('class-label', 'children')],
        [Input('3d-scatter', 'hoverData')]
    )
    def display_hover_data(hoverData):
        if hoverData is None:
            return html.Div("No selection"), ""
        
        try:
            point_idx = hoverData['points'][0]['customdata'][0]
            label = hoverData['points'][0]['customdata'][1]
            
            # Use the 'images' array (Pass the original color array here!)
            img_base64 = array_to_base64(images[point_idx])
            
            img_element = html.Img(
                src=img_base64, 
                style={
                    'width': '100%', 
                    'border': '1px solid #ddd', 
                    'borderRadius': '4px',
                    'boxShadow': '0px 4px 8px rgba(0,0,0,0.1)'
                }
            )
            return img_element, f"Morphology: {label}"
        except Exception as e:
            return html.Div(f"Error: {e}"), ""

    app.run(debug=True, use_reloader=False)
    

def load_cnn():
    output_filename = r"maps/cnn_embedding_gray_v4.npz"
    print("Loading saved features...")
    data = np.load(output_filename)
    cnn_embeddings = data['cnn_embeddings']
    dim = 2048
    return cnn_embeddings, dim, False

def load_kymatio():
    output_filename = r"maps/kymatio_embedding_gray_final.npz"
    print("Loading saved features...")
    data = np.load(output_filename)
    kymatio_embeddings = data['kymatio_embeddings']
    print(np.shape(kymatio_embeddings))
    dim = 469
    return kymatio_embeddings, dim, False

def load_rigid_motion():
    output_filename = r"maps/rigid_motion_embedding_gray_v4.npz"
    print("Loading saved features...")
    data = np.load(output_filename)
    rm_embeddings = data['rm_embeddings']
    dim = 112
    return rm_embeddings, dim, True

def plot_maps_dist(embds, J=3, L=12):
    # 1. Generate the two states of data
    # Raw includes S0 and no log/layer-norm
    X_basic = embds # Assuming this is (N, 469)
    # Preprocessed includes log1p, S0 removal, and layer-wise StandardScaler
    X_preprocessed = prepare_embedding(embds, is_scattering=True) # Result: (N, 468)

    # 2. Extract representative slices for visualization
    # We compare S0 (if exists), a sample of S1, and a sample of S2
    s1_end = J * L + 1 # +1 because X_basic still has S0
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

    # --- Plot A: Basic Scaling (The 'Drowning' Effect) ---
    sns.kdeplot(X_basic[:, 0], ax=axes[0], label='$S_0$ (Global Flux)', color='black', lw=2)
    sns.kdeplot(X_basic[:, 1:37].flatten(), ax=axes[0], label='$S_1$ (Edges)', color='blue', alpha=0.6)
    sns.kdeplot(X_basic[:, 37:].flatten(), ax=axes[0], label='$S_2$ (Texture)', color='red', alpha=0.4)
    
    axes[0].set_title(r"Standard $\log(1+x)$ Scaling (Dominant Nuisance Parameter)", fontsize=14)
    axes[0].set_xlabel("Coefficient Magnitude")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # --- Plot B: Proposed Structural Pipeline ---
    # In X_preprocessed, S0 is gone. S1 and S2 are Z-scored.
    sns.kdeplot(X_preprocessed[:, :36].flatten(), ax=axes[1], label='$S_1$ (Normalized)', color='blue', lw=2)
    sns.kdeplot(X_preprocessed[:, 36:].flatten(), ax=axes[1], label='$S_2$ (Normalized)', color='red', lw=2)
    
    axes[1].set_title("Proposed Structural Pipeline\n(Layer-wise Decorrelation)", fontsize=14)
    axes[1].set_xlabel("Z-Score (Standard Deviations)")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('figures/distribution_comparison_kymatio.pdf', dpi=300)
    plt.show()


### MAIN CODE ###

if __name__ == "__main__":

    data_path_pro = 'data/Galaxy10_ProcessedandCroppedFinal.h5'
    data_path_real = 'data/Galaxy10_DECals.h5'


    loaded_embeddings, dim, scatter_bool = load_rigid_motion()
    confusion_matrix = np.zeros((10, 10))
    num_classes = 10
    '''
    with h5py.File(data_path, 'r') as F:
        label_indices = np.array(F['ans']) # Copy to memory

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
    '''
    target_classes = [0,1,2,3,4,5,6,7,8,9]

    with h5py.File(data_path_pro, 'r') as F_pro:
        label_indices = F_pro['ans'][:]

        original_lookup = F_pro['original_indices'][:] 
        
        mask = np.isin(label_indices, target_classes)
        subset_processed_indices = np.where(mask)[0]
        
        raw_color_indices = original_lookup[subset_processed_indices]
        
        y_ground_truth = label_indices[subset_processed_indices]

    with h5py.File(data_path_real, 'r') as F_raw:

        subset_real_images = np.array([F_raw['images'][idx] for idx in raw_color_indices])
            
    X_prepared = prepare_embedding(loaded_embeddings[subset_processed_indices], is_scattering=scatter_bool)

    pca = PCA(n_components=0.95)
    PCA_X = pca.fit_transform(X_prepared)

    # UMAP step
    reducer = umap.UMAP(n_components=3, random_state=42, min_dist=0.1, n_neighbors=10, metric='correlation')
    X_manifold_3d = reducer.fit_transform(PCA_X)

    for i in range(3):
        idx_min = np.argmin(X_manifold_3d[:, i])
        idx_max = np.argmax(X_manifold_3d[:, i])
        print(f"Axis {i+1} Min: Index {idx_min}, Axis {i+1} Max: Index {idx_max}")
    
    run_dash_explorer(X_manifold_3d, y_ground_truth, subset_real_images)