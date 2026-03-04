import numpy as np
from kymatio.scattering2d.filter_bank import filter_bank
from scipy.fft import ifft2
import os 
import h5py
from tqdm import tqdm


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

def get_scattering_maps(image):
     
    coeff_maps_list = [] # Stores the flatenned coefficient maps for an image

    freq_channel = np.fft.fft2(image) # Convert to fourier space

    # Order 0 map
    low_pass_gaussian = filters_set['phi']["levels"][0][np.newaxis, ...] # Get low pass filter
    conv_0_freq = freq_channel * low_pass_gaussian                       # Convolve
    conv_0_spatial = ifft2(conv_0_freq)                                  # Convert back to spatial domain
    s0 = np.abs(conv_0_spatial)                                          # Take the modulus
    s0_gap = np.mean(s0, axis=(1, 2))                                    # Global average pooling
    coeff_maps_list.append(s0_gap.reshape(-1))                           # Flatten and add to list

    # Order 1 maps
    order1_cubes = []
    cube_stack   = []
    s1_maps      = []
    psi_filters = filters_set['psi']  # Get Morlet wavelets by scale
    
    for f_dict in psi_filters:
        j = f_dict['j']
        theta = f_dict['theta']
        f_freq = f_dict['levels'][0] [np.newaxis, ...] # Extract relevant wavelet angles

        conv_freq = freq_channel * f_freq                       # Convole with wavelet
        conv_spatial = np.fft.ifft2(conv_freq)                  # Convert to spatial to apply Non-Linearity
        u1_spatial = np.abs(conv_spatial)                       # Take the modulus


        u1_freq = np.fft.fft2(u1_spatial)                       # Back to Fourier space
        s1_freq = u1_freq * low_pass_gaussian                   # Convolve with low pass
        s1_spatial = np.real(np.fft.ifft2(s1_freq))             # Keep only real part of spatial info
        s1_maps.append(s1_spatial)                              # Keep spatial S1 map

        cube_stack.append(u1_freq)                              # Add fourier layer to cube

        if theta == (L - 1): # When cube full

            s1_invariant = np.sum(np.stack(s1_maps), axis=0)                  # Stack all the maps of this scale in a cube and sum over them
            s1_gap = np.mean(s1_invariant, axis=(1, 2))                       # Global average pooling
            coeff_maps_list.append(s1_gap.reshape(-1))                        # Flatten and add to list
            s1_maps = []                                                      # Reset stack
        
            cube_arr = np.squeeze(np.stack(cube_stack))      # Convert cube to 3D
            order1_cubes.append({'j': j, 'cube': cube_arr})  # Add to list
            cube_stack = []                                  # Reset stack
    
    # Order 2 maps
    for parent_dict in order1_cubes:
        parent_j = parent_dict['j']
        parent_cube = parent_dict['cube'] # Shape: (L, H, W)
        
        unique_j2 = sorted(list(set(f['j'] for f in filters_set['psi'])))
        
        for j2 in unique_j2:
            if j2 > parent_j:
                # Get wavelets for the second scale
                filters_at_j2 = np.stack([f['levels'][0] for f in filters_set['psi'] if f['j'] == j2])
                
                # Spatial Convolution
                u2_full = np.abs(ifft2(parent_cube[np.newaxis, ...] * filters_at_j2[:, np.newaxis, ...]))

                # Align by Relative Angle
                aligned_u2 = np.zeros_like(u2_full)
                for t2 in range(L):
                    slice_t2 = u2_full[t2]
                    # Shift theta1 axis to align with relative index
                    aligned_u2[t2] = np.roll(slice_t2, shift=-t2, axis=0)

                # Angular FFT across the Global orientation (axis 1)
                u2_ang_freq = np.fft.fft(aligned_u2, axis=1)
                
                broadcast_shape = (1, L, 1, 1)
                
                # Low-Pass Angular Filter
                phi_ang_b = phi_ang_filter.reshape(broadcast_shape)
                s2_invariant_rel = np.real(np.fft.ifft(u2_ang_freq * phi_ang_b, axis=1)) # Convolve with 1D low pass angular filter
                s2_invariant_rel = np.sum(s2_invariant_rel, axis=1)                      # Integrate out the global orientation
                
                for theta_rel_idx in range(L):
                    s2_freq = np.fft.fft2(s2_invariant_rel[theta_rel_idx]) * low_pass_gaussian # Convole with low pass
                    s2 = np.real(ifft2(s2_freq))                                               # Keep only real part of spatial info
                    s2_gap = np.mean(s2, axis=(1, 2))                                          # Global average pooling
                    coeff_maps_list.append(s2_gap.reshape(-1))                                 # Flatten and add to list
                
                # High-Pass Angular Filters
                for w_1d in wavelets_1d:
                    wavelet1d_b = w_1d.reshape(broadcast_shape)
                    u2_ang_high_spatial = np.abs(np.fft.ifft(u2_ang_freq * wavelet1d_b, axis=1)) # Convolve with 1D high pass angular filter
                    u2_invariant_rel = np.sum(u2_ang_high_spatial, axis=1)                       # Integrate out the global orientation
                    
                    for theta_rel_idx in range(L):
                        s2_freq = np.fft.fft2(u2_invariant_rel[theta_rel_idx]) * low_pass_gaussian # Convole with low pass
                        s2 = np.real(ifft2(s2_freq))                                               # Keep only real part of spatial info
                        s2_gap = np.mean(s2, axis=(1, 2))                                          # Global average pooling
                        coeff_maps_list.append(s2_gap.reshape(-1))                                 # Flatten and add to list

    final_maps = np.concatenate(coeff_maps_list)

    return final_maps

def get_features_in_batches(hdf5_dataset, batch_size=32):
    n_samples = hdf5_dataset.shape[0]
    n_coefficients = n_features # Pre-calculated based on J, L, K
    features = np.zeros((n_samples, n_coefficients), dtype=np.float32)

    for start in tqdm(range(0, n_samples, batch_size), desc="Extracting Rigid ScatNet"):
        end = min(start + batch_size, n_samples)
        
        batch = hdf5_dataset[start:end].astype(np.float32)
        
        for i in range(end - start):
            img_input = batch[i] 
            features[start + i] = get_scattering_maps(img_input)
    
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
data_path       = 'data/Galaxy10_ProcessedandCropped.h5'

if os.path.exists(output_filename):
    print("Loading saved features...")
    data = np.load(output_filename)
    rm_embeddings = data['rm_embeddings']
else:
    print("Opening HDF5 file for lazy-loading...")
    with h5py.File(data_path, 'r') as F:
        # Pass the HDF5 dataset directly to the batch function
        rm_embeddings = get_features_in_batches(F['images'], batch_size=16)
    
    # Save results
    np.savez_compressed(output_filename, rm_embeddings=rm_embeddings)
    print(f"Extraction Complete. Shape: {rm_embeddings.shape}")


