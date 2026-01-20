import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
from sklearn.decomposition import PCA
from kymatio.scattering2d.filter_bank import filter_bank
from scipy.fft import ifft2
import os 
from sklearn.model_selection import GridSearchCV
import h5py
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, classification_report
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from scipy.stats import randint, uniform


def get_1d_wavelets(L, k_scale = 1, sigma_scale=0.8):

    psi_freq_list = []
    for k in range(k_scale):

        qs = np.fft.fftfreq(L) * L 
        xi = 2.5 / (2**k)   
        sigma = sigma_scale * (2**k)
        gaussian_peak = np.exp(- (qs - xi)**2 / (2 * sigma**2))
        gaussian_correction = np.exp(- (qs**2) / (2 * sigma**2)) 
        kappa = np.exp(- 0.5 * xi**2 / sigma**2)
        
        psi_freq = gaussian_peak - kappa * gaussian_correction
        psi_freq /= np.max(np.abs(psi_freq))
        psi_freq_list.append(psi_freq)
    
    return psi_freq_list


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

                    freq_slice = parent_cube[angle_idx]               # Retrieve slice
                    conv_2_freq = freq_slice * filters_at_j2[angle_idx] # Convole with matching angle wavelet
                    temp_spatial_convolutions.append(conv_2_freq)       # Store in new cube
                
                spatial_pass_cube = np.stack(temp_spatial_convolutions)            # Stack up
                cube_spatial = np.fft.ifft2(spatial_pass_cube)                     # Back to spatial domain
                cube_L_freq = np.fft.fft(cube_spatial, axis=0)                     # FFT along orientation Z axis

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



    return np.concatenate(coeff_maps_list)


def create_labels(y_ratio, y_time):
    labels = np.zeros(len(y_ratio), dtype=int)
    
    for i in range(len(y_ratio)):
        r = y_ratio[i]
        t = y_time[i]
        
        is_large = r >= RATIO_CUTOFF
        is_recent = t <= TIME_CUTOFF
        
        if is_large and is_recent:
            labels[i] = 0 # Large Recent
        elif not is_large and is_recent:
            labels[i] = 0 # Small Recent
        elif is_large and not is_recent:
            labels[i] = 1 # Large Old 
        elif not is_large and not is_recent:
            labels[i] = 1 # Small Old

    for i in range(0, 2):
        print(f"{len([x for x in labels if x == i])} in class {i}")    
    return labels, ["Recent Merger", "Old Merger"]

file_path = "HST_256x256_halfstellar32.hdf5"

M_input = 33
M_crop = 32
J = 2 # Number of scales
L = 8 # Number of angles
K = 2 # Number of 1d scales
channel = 2 # Channel index
downscale_factor = 2**J

spatial_side = M_input // downscale_factor
n_pixels = spatial_side ** 2

n_order0 = 1 # Low pass filter
n_order1 = J # Each scale and angle pair
n_order2 = (J * (J - 1) // 2) * K # J2 > J1
n_maps_per_channel = n_order0 + n_order1 + n_order2

n_features = n_maps_per_channel * n_pixels

filters_set = filter_bank(M_crop, M_crop, J, L=L) # Creates the Morlet wavelets
wavelets1d_list = get_1d_wavelets(L=L, k_scale = K)

RATIO_CUTOFF = -0.6  # Boundary between Major and Minor
TIME_CUTOFF = 2.0    # Boundary between Recent and Old

data = np.load("preprocessed-galaxy-classification-36x36.npz")
X = data['X']   
#X = X[:, np.newaxis, ...]
Y_is_merger = data['Y_is_merger']

print(f"Original X shape: {X.shape}")

margin = (M_input - M_crop) // 2
X_cropped = X[:, :, margin:margin+M_crop, margin:margin+M_crop]

print(f"Cropped X shape: {X_cropped.shape}") 

X_cropped = X_cropped.astype('float32') / 255.0 

'''
with h5py.File(file_path, 'r') as hf:

    X_raw = hf["X"]
    Y_time = hf["Y_time"][:]
    Y_ratio = hf["Y_ratio"][:]

    Y_classes, names = create_labels(Y_ratio, Y_time)
    
    indices = np.arange(len(Y_classes))
    idx_train, idx_test, Y_train, Y_test = train_test_split(indices, Y_classes, test_size=0.25, random_state=42)
'''

X_train, X_test, Y_train, Y_test = train_test_split(X_cropped, Y_is_merger, test_size=0.20, random_state=42)
output_filename = f"{M_crop}scattering_features_J{J}_L{L}.npz"

if os.path.exists(output_filename):
    print("Loading saved features...")
    data = np.load(output_filename)
    Sx_train = data['Sx_train']
    Sx_test = data['Sx_test']
else:
    print("Processing Training Set...")
    
    Sx_train = np.zeros((len(X_train), n_features), dtype=np.float32)

    for row, img in enumerate(tqdm(X_train)):
        feats = get_scattering_maps(img)
        Sx_train[row] = feats

    print("Processing Test Set...")

    Sx_test = np.zeros((len(X_test), n_features), dtype=np.float32)

    for row, img in enumerate(tqdm(X_test)):    
        feats = get_scattering_maps(img)
        Sx_test[row] = feats

    # Save
    np.savez_compressed(output_filename, Sx_train=Sx_train, Sx_test=Sx_test, Y_train=Y_train, Y_test=Y_test)
    print("Saved features.")

Sx_train_log = np.log1p(Sx_train)
Sx_test_log = np.log1p(Sx_test)

pipeline = Pipeline([
    ('rf', RandomForestClassifier( 
        n_estimators = 300,
        max_depth = None,
        min_samples_leaf = 5,
        random_state=42))
], verbose=True)


param_grid = {
    
    'rf__min_samples_leaf': [5, 8, 10], 
    'rf__max_features': ["sqrt", "log2"] 

}

#print("Starting Grid Search...")
#grid = GridSearchCV(pipeline, param_grid, cv=2, verbose=2, scoring='balanced_accuracy')

pipeline.fit(Sx_train, Y_train)

#print(f"Best Accuracy: {grid.best_score_}")
#print(f"Best Params: {grid.best_params_}")

preds = pipeline.predict(Sx_test)

'''
mae = mean_absolute_error(Y_test, preds)
rmse = root_mean_squared_error(Y_test, preds)

print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
'''

dump(pipeline, 'galaxy_rf_model.joblib') 
print("Model saved")

print(classification_report(Y_test, preds, target_names=["Non Merger", "Merger"]))

'''
plt.figure(figsize=(6, 6))
plt.scatter(Y_test, preds, alpha=0.3, color='blue', label='Predictions')

min_val = min(Y_test.min(), preds.min())
max_val = max(Y_test.max(), preds.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')

plt.xlabel("True Log Mass Ratio (Ground Truth)")
plt.ylabel("Predicted Log Mass Ratio")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
'''
