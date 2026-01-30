import os
import umap

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from galaxy_mnist import GalaxyMNIST
import numpy as np
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras import utils
import h5py
import matplotlib.pyplot as plt

output_filename = r"maps/cnn_embedding.npz"

# To get the images and labels from file
with h5py.File('data/Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])

label_indices = labels.copy() 
labels_onehot = utils.to_categorical(labels, 10)

label_info = {
0: "Disturbed Galaxies",
1: "Merging Galaxies",
2: "Round Smooth Galaxies",
3: "In-between Round Smooth Galaxies",
4: "Cigar Shaped Smooth Galaxies",
5: "Barred Spiral Galaxies",
6: "Unbarred Tight Spiral Galaxies",
7: "Unbarred Loose Spiral Galaxies",
8: "Edge-on Galaxies without Bulge",
9: "Edge-on Galaxies with Bulge"
}

fig = plt.figure(figsize=(20, 8))

for i in range(10):

    all_indices_for_class = np.where(label_indices == i)[0]
    
    single_image_idx = all_indices_for_class[0]
    
    plt.subplot(2, 5, i + 1)
    
    plt.imshow(images[single_image_idx])
    plt.title(f"{label_info[i]}") 
    plt.axis('off') 

plt.tight_layout()
plt.show()


def get_features_in_batches(images, batch_size=32):

    model = ResNet50V2(
    weights='imagenet', 
    include_top=False, 
    pooling='avg', 
    input_shape=(256, 256, 3) 
    )
    
    n_samples = len(images)
    features = np.zeros((n_samples, 2048), dtype=np.float32)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        batch = images[start:end]
        batch = batch.astype(np.float32)
        batch = preprocess_input(batch)
        features[start:end] = model.predict(batch, verbose=0)
        
        if start % 100 == 0:
            print(f"Processed {start}/{n_samples}")
            
    return features


if os.path.exists(output_filename):
    print("Loading saved features...")
    data = np.load(output_filename)
    loaded_embeddings = data['cnn_embeddings']
else:
    cnn_embeddings = get_features_in_batches(images)
    # Save
    np.savez_compressed(output_filename, cnn_embeddings = cnn_embeddings)
    print(f"Extraction Complete. Shape: {cnn_embeddings.shape}")

mask = np.isin(label_indices, [5, 6, 7])
required_indices = np.where(mask)[0]

standard_embedding = umap.UMAP(random_state=42, n_jobs=1, min_dist=0.5, n_neighbors = 5).fit_transform(loaded_embeddings[required_indices])
plt.figure(figsize=(10, 8))
scatter = plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=label_indices[required_indices], s=0.1, cmap='Spectral')
plt.colorbar(scatter, ticks=[5, 6, 7], label='Galaxy Class')
plt.show()