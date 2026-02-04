import os
import numpy as np
import h5py
from kymatio import Scattering2D

output_filename = r"maps/kymatio_embedding_gray.npz"
data_path = 'data/Galaxy10_DECals.h5'

# To get the images and labels from file
with h5py.File('data/Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images'])

def get_features_in_batches(images, batch_size=32):

    model = Scattering2D(J=5, L=8, shape=(256, 256))
    
    n_samples = len(images)
    n_coefficients = 481 
    features = np.zeros((n_samples, n_coefficients), dtype=np.float32)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        batch = images[start:end].astype(np.float32) / 255.0
        
        S = model.scattering(batch)
        features[start:end] = S.mean(axis=(2, 3))
        
        if start % 100 == 0:
            print(f"Processed {start}/{n_samples}")


# Main Execution Logic
if os.path.exists(output_filename):
    print("Loading saved features...")
    data = np.load(output_filename)
    kymatio_embeddings = data['kymatio_embeddings']
else:
    print("Opening HDF5 file for lazy-loading...")
    with h5py.File(data_path, 'r') as F:
        # Pass the HDF5 dataset directly (don't convert to np.array here!)
        kymatio_embeddings = get_features_in_batches(F['images'])
    
    # Save results
    np.savez_compressed(output_filename, kymatio_embeddings=kymatio_embeddings)
    print(f"Extraction Complete. Shape: {kymatio_embeddings.shape}")

