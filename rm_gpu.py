import numpy as np
from kymatio.scattering2d.filter_bank import filter_bank
import os 
import h5py
from tqdm import tqdm
import cupy as cp
import numpy as np

def prepare_gpu_filters(filters_set, wavelets_1d, phi_ang_filter):
    gpu_filters = {
        'phi': cp.array(filters_set['phi']["levels"][0], dtype=cp.complex64),
        'psi': [],
        'wavelets_1d': [cp.array(w, dtype=cp.complex64) for w in wavelets_1d],
        'phi_ang': cp.array(phi_ang_filter, dtype=cp.complex64)
    }
    
    for f_dict in filters_set['psi']:
        gpu_filters['psi'].append({
            'j': f_dict['j'],
            'theta': f_dict['theta'],
            'filter': cp.array(f_dict['levels'][0], dtype=cp.complex64)
        })
    return gpu_filters


def get_1d_wavelets(L, K=1):
    """Returns a list of K 1D wavelets for the angular axis."""
    psi_list = []
    qs = np.fft.fftfreq(L) * L
    
    for k in range(1, K + 1):
        # xi is the center frequency. It must stay below L/2.
        xi = 0.85 * (np.pi) * (2**(-k)) * (L / (2 * np.pi))
        sigma = 0.8 * (2**(-k)) * (L / (2 * np.pi))
        
        # Gaussian peak + DC correction (Morlet)
        gaussian_peak = np.exp(- (qs - xi)**2 / (2 * sigma**2))
        kappa = np.exp(- 0.5 * xi**2 / sigma**2)
        gaussian_correction = np.exp(- (qs**2) / (2 * sigma**2))
        
        psi_freq = gaussian_peak - kappa * gaussian_correction
        
        # Normalize to unit L-infinity norm for energy stability
        psi_freq /= np.max(np.abs(psi_freq))
        psi_list.append(psi_freq)
        
    return psi_list

def get_angular_phi(L):
    """Returns a 1D low-pass filter for the angular axis."""
    qs = np.fft.fftfreq(L) * L
    # Sigma must be wide enough to support the lowest xi of the wavelets
    sigma = 0.8 * (L / (2 * np.pi)) 
    
    # We define it in the Fourier domain
    phi_freq = np.exp(- (qs**2) / (2 * sigma**2))
    phi_freq[0] = 1.0 # Ensure DC is perfectly preserved
    
    return phi_freq

def get_scattering_maps_gpu(batch_images, filters_gpu, L, K):
     
    B, H, W = batch_images.shape
    coeff_maps_list = [] # Stores the flatenned coefficient maps for an image

    freq_channel = cp.fft.fft2(batch_images)       # Convert to fourier space
    phi_2d = filters_gpu['phi'] 

    # Order 0 map
    low_pass_gaussian = phi_2d[cp.newaxis, ...]                          # Extract low pass
    conv_0_freq = freq_channel * low_pass_gaussian                       # Convolve
    conv_0_spatial = cp.fft.ifft2(conv_0_freq)                           # Convert back to spatial domain
    s0 = cp.abs(conv_0_spatial)                                          # Take the modulus
    s0_gap = cp.mean(s0, axis=(-2, -1))                                  # Global average pooling
    coeff_maps_list.append(s0_gap[:, cp.newaxis])                        # Flatten and add to list
    del s0, conv_0_freq, conv_0_spatial                                  # Free memory immediately

    # Order 1 maps
    order1_cubes = []
    cube_stack   = []
    s1_maps      = []
    
    for f_dict in filters_gpu['psi']:
        j = f_dict['j']
        theta = f_dict['theta']

        conv_freq = freq_channel * f_dict['filter'][cp.newaxis, ...] # Convole with wavelet
        conv_spatial = cp.fft.ifft2(conv_freq)                       # Convert to spatial to apply Non-Linearity
        u1_spatial = cp.abs(conv_spatial)                            # Take the modulus


        u1_freq = cp.fft.fft2(u1_spatial)                       # Back to Fourier space
        s1_freq = u1_freq * low_pass_gaussian                   # Convolve with low pass
        s1_spatial = cp.real(cp.fft.ifft2(s1_freq))             # Keep only real part of spatial info
        s1_maps.append(s1_spatial)                              # Keep spatial S1 map

        cube_stack.append(u1_freq)                              # Add fourier layer to cube

        if theta == (L - 1): # When cube full

            s1_invariant = cp.sum(cp.stack(s1_maps), axis=0)                  # Stack all the maps of this scale in a cube and sum over them
            s1_gap = cp.mean(s1_invariant, axis=(-2, -1))                     # Global average pooling
            coeff_maps_list.append(s1_gap[:, cp.newaxis])                     # Flatten and add to list
            s1_maps = []                                                      # Reset stack
        
            cube_arr = cp.squeeze(cp.stack(cube_stack))                            # Convert cube to 3D
            order1_cubes.append({'j': f_dict['j'], 'cube': cp.stack(cube_stack)})  # Add to list
            cube_stack = []                                                        # Reset stack
    
    # Order 2 maps
    for parent_dict in order1_cubes:
    
        parent_cube = parent_dict['cube'] # Shape: (L, B, H, W)
        unique_j2 = sorted(list(set(f['j'] for f in filters_gpu['psi'] if f['j'] > parent_dict['j'])))
        
        for j2 in unique_j2:
            # Get wavelets for the second scale
            filters_at_j2 = cp.stack([f['filter'] for f in filters_gpu['psi'] if f['j'] == j2])
            f_expanded = filters_at_j2[:, cp.newaxis, cp.newaxis, :, :]
            p_expanded = parent_cube[cp.newaxis, :, :, :, :]
            # Spatial Convolution
            u2_full = cp.abs(cp.fft.ifft2(f_expanded * p_expanded))

            # Align by Relative Angle
            for t2 in range(L):
                # Shift theta1 axis to align with relative index
                u2_full[t2] = cp.roll(u2_full[t2], shift=-t2, axis=0)

            # Angular FFT across the Global orientation (axis 1)
            u2_ang_freq = cp.fft.fft(u2_full, axis=1)
            broadcast_shape = (1, L, 1, 1, 1)
            
            # Low-Pass Angular Filter
            phi_ang_b = filters_gpu['phi_ang'].reshape(broadcast_shape)
            s2_invariant_rel = cp.real(cp.fft.ifft(u2_ang_freq * phi_ang_b, axis=1)) # Convolve with 1D low pass angular filter
            s2_invariant_rel = cp.sum(s2_invariant_rel, axis=1)                      # Integrate out the global orientation
            
            for theta_rel_idx in range(L):
                s2_freq = cp.fft.fft2(s2_invariant_rel[theta_rel_idx]) * phi_2d[cp.newaxis, ...] # Convole with low pass
                s2 = cp.real(cp.fft.ifft2(s2_freq))                                              # Keep only real part of spatial info
                s2_gap = cp.mean(s2, axis=(-2, -1))                                              # Global average pooling
                coeff_maps_list.append(s2_gap[:, cp.newaxis])                                    # Flatten and add to list
            
            # High-Pass Angular Filters
            for w_1d in filters_gpu['wavelets_1d']:
                wavelet1d_b = w_1d.reshape(broadcast_shape)
                u2_ang_high_spatial = cp.abs(cp.fft.ifft(u2_ang_freq * wavelet1d_b, axis=1)) # Convolve with 1D high pass angular filter
                u2_invariant_rel = cp.sum(u2_ang_high_spatial, axis=1)                       # Integrate out the global orientation
                
                for theta_rel_idx in range(L):
                    s2_freq = cp.fft.fft2(u2_invariant_rel[theta_rel_idx]) * phi_2d[cp.newaxis, ...] # Convole with low pass
                    s2 = cp.real(cp.fft.ifft2(s2_freq))                                              # Keep only real part of spatial info
                    s2_gap = cp.mean(s2, axis=(-2, -1))                                              # Global average pooling
                    coeff_maps_list.append(s2_gap[:, cp.newaxis])                                    # Flatten and add to list

            del u2_full, u2_ang_freq, s2_invariant_rel # Free memory

    return cp.concatenate(coeff_maps_list, axis=1)

def get_features_gpu_batched_loop(hdf5_dataset, filters_gpu, n_features, L, K, batch_size=8):
    n_samples = hdf5_dataset.shape[0]
    features = np.zeros((n_samples, n_features), dtype=np.float32)

    for i in tqdm(range(0, n_samples, batch_size), desc="GPU Batch Extraction"):
        end = min(i + batch_size, n_samples)
        
        # 1. Transfer batch to GPU
        batch_cpu = hdf5_dataset[i:end].astype(np.float32)
        batch_gpu = cp.array(batch_cpu)
        
        # 2. Process
        feat_gpu = get_scattering_maps_gpu(batch_gpu, filters_gpu, L, K)
        
        # 3. Bring back to CPU
        features[i:end] = feat_gpu.get()
        
        # 4. Clear VRAM for next batch
        del batch_gpu, feat_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
    return features

M_input = 256
J = 5 # Number of scales
L = 8 # Number of angles
K = 2 # Number of 1d scales


n_order0 = 1 # Low pass filter
n_order1 = J # Each scale 
n_order2 = (J * (J - 1) // 2) * L * (K + 1)
n_features = n_order0 + n_order1 + n_order2

filters_set    = filter_bank(M_input, M_input, J, L=L) # Creates Morlet wavelets
wavelets_1d    = get_1d_wavelets(L=L, K = K)           # Creates High pass filter
phi_ang_filter = get_angular_phi(L)                    # Creates Low pass filter

output_filename = r"maps/rigid_motion_embedding_gray.npz"
data_path       = r'C:\IP_data\Galaxy10_ProcessedandCropped.h5'

gpu_filters = prepare_gpu_filters(filters_set, wavelets_1d, phi_ang_filter)

if __name__ == '__main__':

    if os.path.exists(output_filename):
        print("Loading saved features...")
        data = np.load(output_filename)
        rm_embeddings = data['rm_embeddings']
    else:
        print("Opening HDF5 file...")
        with h5py.File(data_path, 'r') as F:
            # We use the GPU loop instead of Parallel executor
            rm_embeddings = get_features_gpu_batched_loop(
                F['images'], 
                gpu_filters, 
                n_features, 
                L, 
                K
            )
        
        np.savez_compressed(output_filename, rm_embeddings=rm_embeddings)
        print(f"Extraction Complete. Shape: {rm_embeddings.shape}")


