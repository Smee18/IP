import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import h5py
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tqdm import tqdm

output_filename = r"maps/cnn_embedding_gray.npz"
data_path = 'data/Galaxy10_DECals.h5'

def get_features_in_batches(hdf5_dataset, batch_size=32):
    # Initialize model once
    model = ResNet50V2(
        weights='imagenet', 
        include_top=False, 
        pooling='avg', 
        input_shape=(256, 256, 3) 
    )
    
    n_samples = hdf5_dataset.shape[0]
    features = np.zeros((n_samples, 2048), dtype=np.float32)

    for start in tqdm(range(0, n_samples, batch_size), desc="Extracting CNN"):
        end = min(start + batch_size, n_samples)
        
        batch = hdf5_dataset[start:end].astype(np.float32)
        
        for i in range(end - start):
            img_input = batch[i] 
            features[start + i] = model.predict(img_input)
    
    return features

# Main Execution Logic
if os.path.exists(output_filename):
    print("Loading saved features...")
    data = np.load(output_filename)
    cnn_embeddings = data['cnn_embeddings']
else:
    print("Opening HDF5 file for lazy-loading...")
    with h5py.File(data_path, 'r') as F:
        # Pass the HDF5 dataset directly (don't convert to np.array here!)
        cnn_embeddings = get_features_in_batches(F['images'])
    
    # Save results
    np.savez_compressed(output_filename, cnn_embeddings=cnn_embeddings)
    print(f"Extraction Complete. Shape: {cnn_embeddings.shape}")