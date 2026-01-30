import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump
from kymatio.scattering2d.filter_bank import filter_bank
from scipy.fft import ifft2
import os 
import h5py
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from sklearn.utils import shuffle
import umap
import umap.plot
from sklearn.neighbors import NearestNeighbors


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


file_path = r"data/HST_256x256_halfstellar32.hdf5"

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

output_filename = f"maps/{M_input}random_scattering_features_J{J}_L{L}.npz"

if os.path.exists(output_filename):
    print("Loading saved features...")
    data = np.load(output_filename)
    Sx_train = data['Sx_train']
    Sx_test = data['Sx_test']
    Y_train_classes = data['Y_train_classes'] 
    Y_test_classes = data['Y_test_classes']
    indices = data['all_indices']
    idx_train = data['idx_train']

else:

    with h5py.File(file_path, 'r') as hf:

        X_raw = hf["X"]
        Y_time = hf["Y_time"][:]
        Y_ratio = hf["Y_ratio"][:]

        Y_classes_full = []

        for time, ratio in zip(Y_time, Y_ratio): 
            
            # 1. Merger: Recent (0.5 Gyr) and Major (> 1:4 ratio)
            if time < 0.5 and 10 ** ratio > 0.25:
                Y_classes_full.append(1) 
            
            # 2. Non-Merger: Old (> 2 Gyr) OR very minor (< 1:100 ratio)
            if time >= 2 or 10 ** ratio <= 0.01:
                Y_classes_full.append(0) 

        Y_classes_full = np.array(Y_classes_full)

        unique, counts = np.unique(Y_classes_full, return_counts=True)
        print("Class Distribution:", dict(zip(unique, counts)))


        selected_indices = []
        max_per_class = np.min(counts)

        for class_id in [0, 1]:
            # Find indices where the class is class_id
            indices = np.where(Y_classes_full == class_id)[0]
            random_subset = np.random.choice(indices, max_per_class, replace=False)
            selected_indices.extend(random_subset)

        #selected_indices = np.sort(selected_indices)

        selected_indices = np.sort(np.random.choice(np.arange(0, len(Y_time)), 200, replace=False))

        X_new = X_raw[selected_indices]
        #Y_classes_new = Y_classes_full[selected_indices]

        indices = np.arange(len(selected_indices))
        idx_train, idx_test = train_test_split(indices, test_size=0.20, random_state=42)

        #Y_train_classes = np.repeat(Y_classes_new[idx_train], 3)
        #Y_test_classes = np.repeat(Y_classes_new[idx_test], 3)

        Y_train_classes = []
        Y_test_classes = []

        print("Processing Training Set...")
        
        Sx_train = np.zeros((len(idx_train)*3, n_features), dtype=np.float32)

        for row, sample in enumerate(tqdm(idx_train)):
            for i, projections in enumerate(X_new[sample]):
                feats = get_scattering_maps(projections)
                Sx_train[row * 3 + i] = feats

        print("Processing Test Set...")

        Sx_test = np.zeros((len(idx_test)*3, n_features), dtype=np.float32)

        for row, sample in enumerate(tqdm(idx_test)):
            for i, projections in enumerate(X_new[sample]):
                feats = get_scattering_maps(projections)
                Sx_test[row * 3 + i] = feats

        # Save
        np.savez_compressed(output_filename, 
                            Sx_train=Sx_train, 
                            Sx_test=Sx_test,
                            Y_train_classes=Y_train_classes,
                            Y_test_classes=Y_test_classes,
                            all_indices=selected_indices,
                            idx_train=idx_train)
        print("Saved features.")

Sx_train_log = np.log1p(Sx_train)
Sx_test_log = np.log1p(Sx_test)

print(Sx_train.shape)

kmeans = cluster.KMeans(n_clusters=2, random_state=42).fit(Sx_train_log)
labels = kmeans.labels_    # Get the labels
centroids = kmeans.cluster_centers_ # Get the centers (Shape: 2, n_features)


standard_embedding = umap.UMAP(random_state=42, n_jobs=1, min_dist=0.5, n_neighbors = 5).fit_transform(Sx_train_log)
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=labels, s=0.1, cmap='Spectral')
plt.show()


nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(Sx_train_log)

for i, center in enumerate(centroids):
    distances, neighbor_indices = nbrs.kneighbors([center])
    
    prototype_rows = neighbor_indices[0]
    
    print(f"\n=== Cluster {i} Prototypes (Indices: {prototype_rows}) ===")
    
    # Setup plot
    fig, axes = plt.subplots(1, 8, figsize=(20, 3))
    fig.suptitle(f'Cluster {i} Prototypes', fontsize=16)
    
    for ax_idx, row_idx in enumerate(prototype_rows):
        
        # 1. Calculate which galaxy in idx_train this row belongs to
        train_list_index = row_idx // 3
        projection_index = row_idx % 3
        
        # 2. Get the actual index in X_new
        gal_id_in_new = idx_train[train_list_index]
        
        # 3. Retrieve Image
        img = X_raw[gal_id_in_new][projection_index]
        
        # 4. Plot
        axes[ax_idx].imshow(img, cmap='gray_r', origin='lower')
        axes[ax_idx].axis('off')
        axes[ax_idx].set_title(f"Row: {row_idx}\nProj: {projection_index}")
        
    plt.show()
