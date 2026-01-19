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

file_path = "HST_256x256_halfstellar32.hdf5"

M_input = 256
J = 6
L = 8
downscale_factor = 2**J

spatial_side = M_input // downscale_factor
n_pixels = spatial_side ** 2

n_order0 = 1
n_order1 = J
n_order2 = (J * (J - 1) // 2) * L 
n_maps_per_channel = n_order0 + n_order1 + n_order2

n_features = n_maps_per_channel * n_pixels * 3

filters_set = filter_bank(M_input, M_input, J, L=L)

'''
data = np.load("preprocessed-galaxy-classification-36x36.npz")
X = data['X']           
Y_is_merger = data['Y_is_merger']

print(f"Original X shape: {X.shape}")

margin = (M_input - M_crop) // 2
X_cropped = X[:, :, margin:margin+M_crop, margin:margin+M_crop]

print(f"Cropped X shape: {X_cropped.shape}") 

X_cropped = X_cropped.astype('float32') / 255.0 
'''


def get_scattering_maps(image):
     
    coeff_maps_list = []

    freq_channel = np.fft.fft2(image)

    # Order 0 map
    low_pass_gaussian = filters_set['phi']["levels"][0][np.newaxis, ...]
    conv_0_freq = freq_channel * low_pass_gaussian
    conv_0_spatial = ifft2(conv_0_freq)
    s0 = np.real(conv_0_spatial)
    s0_down = s0[:, ::downscale_factor, ::downscale_factor]
    coeff_maps_list.append(s0_down.reshape(-1))

    # Order 1 maps
    order1_cubes = []
    cube_stack = []
    psi_filters = filters_set['psi']
    
    for f_dict in psi_filters:
        j = f_dict['j']
        theta = f_dict['theta']
        f_freq = f_dict['levels'][0] [np.newaxis, ...]

        convolution = freq_channel * f_freq
        conv_spatial = ifft2(convolution)
        cube_stack.append(conv_spatial)

        if theta == (L - 1):
        
            cube_arr = np.stack(cube_stack)
            
            order1_cubes.append({'j': j, 'cube': cube_arr})

            cube_modulus = np.abs(cube_arr)
            s1_invariant = np.sum(cube_modulus, axis=0)
            s1_down = s1_invariant[:, ::downscale_factor, ::downscale_factor]
            coeff_maps_list.append(s1_down.reshape(-1))
            
            cube_stack = []
    
    # Order 2 maps

    for parent_dict in order1_cubes:
        parent_j = parent_dict['j']
        parent_cube = parent_dict['cube'] # Shape: (8, 32, 32)
        
        unique_j2 = sorted(list(set(f['j'] for f in filters_set['psi'])))
        
        for j2 in unique_j2:
            if j2 > parent_j:
            
                filters_at_j2 = [f['levels'][0][np.newaxis, ...] for f in filters_set['psi'] if f['j'] == j2]
                
                orbit_maps = np.zeros((L, 3, M_input, M_input))
                
                for angle_idx in range(L):

                    freq_slice = np.fft.fft2(np.abs(parent_cube[angle_idx]))

                    for filter_idx, filter in enumerate(filters_at_j2):
                        conv_freq = freq_slice * filter
                        conv_spatial = ifft2(conv_freq)
                        s2_map = np.abs(conv_spatial)
                        alpha = (filter_idx - angle_idx) % L
                        orbit_maps[alpha] += s2_map
                
                for alpha in range(L):
                    final_map = orbit_maps[alpha]
                    final_down = final_map[:, ::downscale_factor, ::downscale_factor]
                    coeff_maps_list.append(final_down.reshape(-1))


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

RATIO_CUTOFF = -0.6  # Boundary between Major and Minor
TIME_CUTOFF = 2.0    # Boundary between Recent and Old

with h5py.File(file_path, 'r') as hf:

    X_raw = hf["X"]
    Y_time = hf["Y_time"][:]
    Y_ratio = hf["Y_ratio"][:]

    Y_classes, names = create_labels(Y_ratio, Y_time)
    
    indices = np.arange(len(Y_classes))
    idx_train, idx_test, Y_train, Y_test = train_test_split(indices, Y_classes, test_size=0.25, random_state=42)

    output_filename = f"255scattering_features_J{J}_L{L}.npz"
    
    if os.path.exists(output_filename):
        print("Loading saved features...")
        data = np.load(output_filename)
        Sx_train = data['Sx_train']
        Sx_test = data['Sx_test']
    else:
        print("Processing Training Set...")
        
        Sx_train = np.zeros((len(idx_train), n_features), dtype=np.float32)

        for row, img_idx in enumerate(tqdm(idx_train)):
            img = X_raw[img_idx] 
            feats = get_scattering_maps(img)
            Sx_train[row] = feats

        print("Processing Test Set...")

        Sx_test = np.zeros((len(idx_test), n_features), dtype=np.float32)

        for row, img_idx in enumerate(tqdm(idx_test)):    
            img = X_raw[img_idx]
            feats = get_scattering_maps(img)
            Sx_test[row] = feats

        # Save
        np.savez_compressed(output_filename, Sx_train=Sx_train, Sx_test=Sx_test, Y_train=Y_train, Y_test=Y_test)
        print("Saved features.")

Sx_train_log = np.log1p(Sx_train)
Sx_test_log = np.log1p(Sx_test)

'''
selector = SelectFromModel(
    BalancedRandomForestClassifier(n_estimators=100, random_state=42), 
    threshold='median'
)

Sx_train_sel = selector.fit_transform(Sx_train_log, Y_train)
Sx_test_sel = selector.transform(Sx_test_log)
'''

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    #('pca', PCA(n_components=100)), 
    ('rf', BalancedRandomForestClassifier(
        n_estimators=200,       
        sampling_strategy='all', 
        replacement=True,     
        random_state=42))
], verbose=True)

param_grid = {

    'rf__min_samples_leaf': [2, 4, 8],
    'rf__max_features': ['sqrt', 0.5, 'log2']
}

print("Starting Grid Search...")
grid = GridSearchCV(pipeline, param_grid, cv=3, verbose=2, scoring='balanced_accuracy')

grid.fit(Sx_train, Y_train)

print(f"Best Accuracy: {grid.best_score_}")
print(f"Best Params: {grid.best_params_}")

preds = grid.predict(Sx_test)

'''
mae = mean_absolute_error(Y_test, preds)
rmse = root_mean_squared_error(Y_test, preds)

print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
'''

dump(grid, 'galaxy_regressor_model.joblib') 
print("Model saved")

print(classification_report(Y_test, preds, target_names=names))

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
