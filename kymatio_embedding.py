import sys
import os
import h5py
import scipy.special
import math
import numpy as np
from tqdm import tqdm

# 1. SciPy 1.15+ moved/removed these, but Kymatio's internal 3D code 
# (which is imported by the 2D frontend) needs them.
# We map them back so the import doesn't crash.

# Proxy for sph_harm (Kymatio 2D doesn't actually call it, just imports it)
if not hasattr(scipy.special, 'sph_harm'):
    scipy.special.sph_harm = lambda *args: None 

# Map factorial back to special
if not hasattr(scipy.special, 'factorial'):
    scipy.special.factorial = math.factorial

# 2. Inject into the system modules so Kymatio sees them
sys.modules['scipy.special'] = scipy.special

# NOW you can import Kymatio
from kymatio.torch import Scattering2D
import torch

output_filename = r"maps/kymatio_embedding_gray.npz"
data_path = 'data/Galaxy10_DECals.h5'


def get_features_in_batches(hdf5_dataset, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize Torch backend
    model = Scattering2D(J=5, L=8, shape=(256, 256)).to(device)
    
    n_samples = hdf5_dataset.shape[0]
    n_coefficients = 681 
    features = np.zeros((n_samples, n_coefficients), dtype=np.float32)
    
    for start in tqdm(range(0, n_samples, batch_size), desc="Extracting Kymatio Embeddings"):
        end = min(start + batch_size, n_samples)
        
        batch = hdf5_dataset[start:end].astype(np.float32)
        gray_batch = (0.299 * batch[..., 0] + 0.587 * batch[..., 1] + 0.114 * batch[..., 2]) / 255.0
        
        # Convert to Torch Tensor
        batch_t = torch.from_numpy(gray_batch).to(device)
        
        # Compute: returns (Batch, 681, 8, 8)
        S = model(batch_t)
        
        # Average Pool and move back to CPU/NumPy
        with torch.no_grad():  
            S = model(batch_t)
            features[start:end] = S.mean(dim=(2, 3)).cpu().numpy()
        
    return features

if os.path.exists(output_filename):
    print("Loading saved features...")
    data = np.load(output_filename)
    kymatio_embeddings = data['kymatio_embeddings']
else:
    print("Opening HDF5 for Lazy-Loading Scattering Extraction...")
    with h5py.File(data_path, 'r') as F:
        kymatio_embeddings = get_features_in_batches(F['images'])
    
    np.savez_compressed(output_filename, kymatio_embeddings=kymatio_embeddings)
    print(f"Extraction Complete. Shape: {kymatio_embeddings.shape}")