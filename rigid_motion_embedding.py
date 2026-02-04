import numpy as np
from kymatio.scattering2d.filter_bank import filter_bank
from scipy.fft import ifft2
import os 
import h5py
from tqdm import tqdm


def get_1d_wavelets(L, K=1):

    psi_list = []
    
    # Iterate over angular scales k < K
    for k in range(K):
        qs = np.fft.fftfreq(L) * L
        
        xi = 0.85 * (2 * np.pi) * (2**(-k)) * (L / (2*np.pi))
        sigma = 0.8 * (2**(-k)) * (L / (2*np.pi))
        
        # 1. The Gaussian Peak (shifted to center frequency xi)
        gaussian_peak = np.exp(- (qs - xi)**2 / (2 * sigma**2))
        
        gaussian_correction = np.exp(- (qs**2) / (2 * sigma**2))
        kappa = np.exp(- 0.5 * xi**2 / sigma**2)
        
        psi_freq = gaussian_peak - kappa * gaussian_correction
        

        psi_freq /= np.max(np.abs(psi_freq))
        
        psi_list.append(psi_freq)
        
    return psi_list


def get_angular_phi(L, sigma_scale=0.8):
    qs = np.fft.fftfreq(L) * L 
    

    sigma = sigma_scale * (L / 2) 

    phi_freq = np.exp(- (sigma**2 * qs**2) / 2)
    
    phi_freq[0] = 1.0 
    
    return phi_freq

def get_scattering_maps(image):
     
    coeff_maps_list = [] # Stores the flatenned coefficient maps for an image with three channels

    freq_channel = np.fft.fft2(image) # Convert to fourier space

    # Order 0 map
    low_pass_gaussian = filters_set['phi']["levels"][0][np.newaxis, ...] # Get low pass filter for each channel
    conv_0_freq = freq_channel * low_pass_gaussian                       # Convolve
    conv_0_spatial = ifft2(conv_0_freq)                                  # Convert back to spatial domain
    s0 = np.abs(conv_0_spatial)                                          # Take the modulus
    s0_down = s0[:, ::downscale_factor, ::downscale_factor]              # Downscale
    coeff_maps_list.append(s0_down.reshape(-1))                          # Flatten and add to list

    # Order 1 maps
    order1_cubes = []
    cube_stack = []
    psi_filters = filters_set['psi'] # Get Morlet wavelets by scale
    u1_sum_spatial = 0 
    
    for f_dict in psi_filters:
        j = f_dict['j']
        theta = f_dict['theta']
        f_freq = f_dict['levels'][0] [np.newaxis, ...] # Extract relevant wavelet angles

        conv_freq = freq_channel * f_freq                       # Convole with wavelet
        conv_spatial = np.fft.ifft2(conv_freq)                  # Convert to spatial to apply Non-Linearity
        u1_spatial = np.abs(conv_spatial)                       # Take the modulus
        u1_sum_spatial = u1_sum_spatial + u1_spatial            # Add to sum
        u1_freq = np.fft.fft2(u1_spatial)                       # Convert back to fourier space
        cube_stack.append(u1_freq)                              # Add layer to cube

        if theta == (L - 1): # When cube full

            u1_sum_freq = np.fft.fft2(u1_sum_spatial)               # FFT the sum of modulus
            s1_freq = u1_sum_freq * low_pass_gaussian               # Smooth with low pass filter
            s1 = np.real(np.fft.ifft2(s1_freq))                     # Convert back to spatial domain
            s1_down = s1[:, ::downscale_factor, ::downscale_factor] # Downscale
            coeff_maps_list.append(s1_down.reshape(-1))             # Flatten and add to list
        
            cube_arr = np.stack(cube_stack)                  # Convert cube to 3D
            order1_cubes.append({'j': j, 'cube': cube_arr})  # Add to list
            cube_stack = []                                  # Reset stack
            u1_sum_spatial =  0                              # Reset sum
    
    # Order 2 maps


    for parent_dict in order1_cubes:
        parent_j = parent_dict['j']
        parent_cube = parent_dict['cube']
        
        unique_j2 = sorted(list(set(f['j'] for f in filters_set['psi'])))
        
        for j2 in unique_j2:
            if j2 > parent_j:
            
                filters_at_j2 = [f['levels'][0][np.newaxis, ...] for f in filters_set['psi'] if f['j'] == j2]
                
                temp_spatial_convolutions = [] # New cube
                
                for angle_idx in range(L): # Loop over angles

                    # Spatial convolve
                    freq_slice = parent_cube[angle_idx]                 # Retrieve slice
                    conv_2_freq = freq_slice * filters_at_j2[angle_idx] # Convole with matching angle wavelet
                    temp_spatial_convolutions.append(conv_2_freq)       # Store in new cube
                
                spatial_pass_cube = np.stack(temp_spatial_convolutions)            # Stack up
                cube_spatial = np.fft.ifft2(spatial_pass_cube)                     # Back to spatial domain
                cube_L_freq = np.fft.fft(cube_spatial, axis=0)                     # FFT along orientation Z axis

                # Low - pass
                low_pass_response = cube_L_freq * phi_ang_filter[
                        :, np.newaxis, np.newaxis, np.newaxis] 
                u2_low = np.abs(np.fft.ifft(low_pass_response, axis=0))
                s2_low_map = np.sum(u2_low, axis=0)
                u2_freq_domain = np.fft.fft2(s2_low_map)                           # FFT for low pass
                s2_freq = u2_freq_domain * low_pass_gaussian                       # Convolve with low pass filter
                s2 = np.real(np.fft.ifft2(s2_freq))                                # Back to spatial domain
                s2_down = s2[:, ::downscale_factor, ::downscale_factor]            # Downscale
                coeff_maps_list.append(s2_down.reshape(-1))                        # Flatten and add to list

                # High - pass
                for wavelet1d in wavelets1d_list:
                    convolved_L = cube_L_freq * wavelet1d[
                        :, np.newaxis, np.newaxis, np.newaxis]                         # 1D convolution
                    u2_complex = np.fft.ifft(convolved_L, axis=0)                      # Back to spatial for orientation
                    u2 = np.abs(u2_complex)                                            # Take the modulus
                    u2_rotation_invariant = np.sum(u2, axis=0)                         # Sum over angular layers
                    u2_freq_domain = np.fft.fft2(u2_rotation_invariant)                # FFT for low pass
                    s2_freq = u2_freq_domain * low_pass_gaussian                       # Convolve with low pass filter
                    s2 = np.real(np.fft.ifft2(s2_freq))                                # Back to spatial domain
                    s2_down = s2[:, ::downscale_factor, ::downscale_factor]            # Downscale
                    coeff_maps_list.append(s2_down.reshape(-1))                        # Flatten and add to list

    final_maps = np.concatenate(coeff_maps_list)

    return final_maps

def get_features_in_batches(hdf5_dataset, batch_size=32):
    n_samples = hdf5_dataset.shape[0]
    n_coefficients = n_features # Pre-calculated based on J, L, K
    features = np.zeros((n_samples, n_coefficients), dtype=np.float32)

    for start in tqdm(range(0, n_samples, batch_size), desc="Extracting Rigid ScatNet"):
        end = min(start + batch_size, n_samples)
        
        batch_rgb = hdf5_dataset[start:end].astype(np.float32)
        
        batch_gray = (0.299 * batch_rgb[..., 0] + 
                      0.587 * batch_rgb[..., 1] + 
                      0.114 * batch_rgb[..., 2]) / 255.0
        
        for i in range(end - start):
            img_input = batch_gray[i][np.newaxis, ...] 
            features[start + i] = get_scattering_maps(img_input)
    
    return features

M_input = 256
J = 5 # Number of scales
L = 8 # Number of angles
K = 1 # Number of 1d scales
downscale_factor = 2**J

spatial_side = M_input // downscale_factor
n_pixels = spatial_side ** 2

n_order0 = 1 # Low pass filter
n_order1 = J # Each scale and angle pair
n_order2 = (J * (J - 1) // 2) * (K+1) # J2 > J1
n_maps_per_channel = n_order0 + n_order1 + n_order2

n_features = n_maps_per_channel * n_pixels

filters_set = filter_bank(M_input, M_input, J, L=L) # Creates the Morlet wavelets
wavelets1d_list = get_1d_wavelets(L=L, K = K)
phi_ang_filter = get_angular_phi(L)

output_filename = r"maps/rigid_motion_embedding_gray.npz"
data_path = 'data/Galaxy10_DECals.h5'

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


