import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
from sklearn.decomposition import PCA
from kymatio.scattering2d.filter_bank import filter_bank
from scipy.fft import ifft2
import os 
from sklearn.model_selection import GridSearchCV
import h5py
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

file_path = "HST_256x256_halfstellar32.hdf5"

M_input = 256
J = 5
L = 8
downscale_factor = 2**J

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


def get_scattering_maps(data):
     
    coeff_maps = []
    for image in data:
        coeff_map_image = []
        for channel in image:

            freq_channel = np.fft.fft2(channel)

            # Order 0 map
            low_pass_gaussian = filters_set['phi']["levels"][0]
            conv_0_freq = freq_channel * low_pass_gaussian
            conv_0_spatial = ifft2(conv_0_freq)
            s0 = np.real(conv_0_spatial)
            s0_down = s0[::downscale_factor, ::downscale_factor]
            coeff_map_image.append(s0_down)

            # Order 1 maps
            order1_cubes = []
            cube_stack = []
            psi_filters = filters_set['psi']
            
            for f_dict in psi_filters:
                j = f_dict['j']
                theta = f_dict['theta']
                f_freq = f_dict['levels'][0] 

                convolution = freq_channel * f_freq
                conv_spatial = ifft2(convolution)
                cube_stack.append(conv_spatial)

                if theta == (L - 1):
                
                    cube_arr = np.stack(cube_stack)
                    
                    order1_cubes.append({'j': j, 'cube': cube_arr})

                    cube_modulus = np.abs(cube_arr)
                    s1_invariant = np.sum(cube_modulus, axis=0)
                    s1_down = s1_invariant[::downscale_factor, ::downscale_factor]
                    coeff_map_image.append(s1_down)
                    
                    cube_stack = []
            
            # Order 2 maps

            for parent_dict in order1_cubes:
                parent_j = parent_dict['j']
                parent_cube = parent_dict['cube'] # Shape: (8, 32, 32)
                
                unique_j2 = sorted(list(set(f['j'] for f in filters_set['psi'])))
                
                for j2 in unique_j2:
                    if j2 > parent_j:
                    
                        filters_at_j2 = [f['levels'][0] for f in filters_set['psi'] if f['j'] == j2]
                        
                        orbit_maps = np.zeros((L, M_input, M_input))
                        
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
                            coeff_map_image.append(final_map[::downscale_factor, ::downscale_factor])



        coeff_maps.append(coeff_map_image)
    coeff_maps_arr = np.array(coeff_maps)
    #print(f"Output Maps: {coeff_maps_arr.shape}")
    coeff_maps_arr.reshape(coeff_maps_arr.shape[0], -1)
    return coeff_maps_arr


with h5py.File(file_path, 'r') as hf:

    X_raw = hf["X"]
    Y_ratio = hf["Y_ratio"][:]
    
    indices = np.arange(len(Y_ratio))
    idx_train, idx_test, Y_train, Y_test = train_test_split(indices, Y_ratio, test_size=0.25, random_state=42)

    # Check for saved file
    output_filename = f"255scattering_features_J{J}_L{L}.npz"
    
    if os.path.exists(output_filename):
        print("Loading saved features...")
        data = np.load(output_filename)
        Sx_train = data['Sx_train']
        Sx_test = data['Sx_test']
    else:
        # --- TRAINING SET ---
        print("Processing Training Set...")
        Sx_train_list = []
        for i in tqdm(idx_train):
            img = X_raw[i] 
            img = np.log1p(img)[np.newaxis, ...] 
            feats = get_scattering_maps(img)
            Sx_train_list.append(feats.reshape(-1))
            
        Sx_train = np.array(Sx_train_list) 

        # --- TEST SET ---
        print("Processing Test Set...")
        Sx_test_list = []
        for i in tqdm(idx_test):
            img = X_raw[i]
            img = np.log1p(img)[np.newaxis, ...]
            feats = get_scattering_maps(img)
            Sx_test_list.append(feats.reshape(-1))
            
        Sx_test = np.array(Sx_test_list)

        # Save
        np.savez_compressed(output_filename, Sx_train=Sx_train, Sx_test=Sx_test, Y_train=Y_train, Y_test=Y_test)
        print("Saved features.")

print(f"Feature Shape: {Sx_train.shape}")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('svm', LinearSVR(random_state=42))
])

param_grid = {
    'svm__C': [0.01, 0.1, 1],
}

print("Starting Grid Search...")
grid = GridSearchCV(pipeline, param_grid, cv=3, verbose=2, n_jobs=None)
grid.fit(Sx_train, Y_train)

print(f"Best Accuracy: {grid.best_score_}")
print(f"Best Params: {grid.best_params_}")

preds = grid.predict(Sx_test)

mae = mean_absolute_error(Y_test, preds)
r2 = r2_score(Y_test, preds)

print(f"Test MAE: {mae:.4f}")
print(f"Test R2: {r2:.4f}")

dump(grid, 'galaxy_regressor_model.joblib') 
print("Model saved")
