import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  

file_path = r"data/HST_256x256_halfstellar32.hdf5"

with h5py.File(file_path, 'r') as hf:
    # Load data
    X = hf["X"][:]
    Y_time = hf["Y_time"][:]
    Y_ratio = hf["Y_ratio"][:]
    
    # 1. Setup Bins 
    n_bins = 20
    # Calculate edges based on data range
    x_edges = np.linspace(Y_ratio.min(), Y_ratio.max(), n_bins + 1)
    y_edges = np.linspace(Y_time.min(), Y_time.max(), n_bins + 1)

    # 2. Assign every data point to a 2D bin
    x_bin_idxs = np.digitize(Y_ratio, x_edges) - 1 # -1 for 0 indexed bins
    y_bin_idxs = np.digitize(Y_time, y_edges) - 1
    
    # Clip indices to handle edge cases
    x_bin_idxs = np.clip(x_bin_idxs, 0, n_bins - 1)
    y_bin_idxs = np.clip(y_bin_idxs, 0, n_bins - 1)

    # 3. Group indices by their bin coordinate
    bin_map = {}
    for i in range(len(Y_ratio)):

        bin_key = (x_bin_idxs[i], y_bin_idxs[i])
        
        if bin_key not in bin_map:
            bin_map[bin_key] = []
        bin_map[bin_key].append(i)

    # 4. Undersample: Select indices
    target_cap = 10 
    
    selected_indices = []
    
    for bin_key, indices in bin_map.items():
        if len(indices) > target_cap:
            # If bin is too tall, sample randomly
            chosen = np.random.choice(indices, target_cap, replace=False)
            selected_indices.extend(chosen)
        else:
            # If bin is short, keep everything
            selected_indices.extend(indices)

    # Convert to array for slicing
    selected_indices = np.array(selected_indices)
    
    # 5. Create the undersampled dataset
    X_new = X[selected_indices]
    Y_time_new = Y_time[selected_indices]
    Y_ratio_new = Y_ratio[selected_indices]

    print(f"Original size: {len(Y_ratio)}")
    print(f"Undersampled size: {len(Y_ratio_new)}")


### VIEW A SAMPLE AND ITS PROJECTIONS ###

fig = plt.figure(figsize=(10, 7))

plt.subplot(1, 3, 1)  
plt.imshow((X[0]), cmap='gray_r')  
plt.axis('off') 
 
plt.subplot(1, 3, 2)  
plt.imshow((X[1]), cmap='gray_r')  
plt.axis('off') 

plt.subplot(1, 3, 3)  
plt.imshow((X[2]), cmap='gray_r') 
plt.axis('off')    

plt.show()

### VIEW DISTRIBUTIONS ###

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Subplot 1: Time
sns.histplot(Y_time, color="red", label="Original", kde=True, stat="density", linewidth=0, alpha=0.4, ax=axes[0])
sns.histplot(Y_time_new, color="blue", label="Undersampled", kde=True, stat="density", linewidth=0, alpha=0.4, ax=axes[0])
axes[0].set_title("Distribution of Time Since Merger")
axes[0].legend()

# Subplot 2: Ratio
sns.histplot(Y_ratio, color="red", label="Original", kde=True, stat="density", linewidth=0, alpha=0.4, ax=axes[1])
sns.histplot(Y_ratio_new, color="blue", label="Undersampled", kde=True, stat="density", linewidth=0, alpha=0.4, ax=axes[1])
axes[1].set_title("Distribution of Log Mass Ratio")
axes[1].legend()

plt.tight_layout()
plt.show()

