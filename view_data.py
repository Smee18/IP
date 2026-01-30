import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import numpy as np
import os
import seaborn as sns  

file_path = r"data/HST_256x256_halfstellar32.hdf5"

with h5py.File(file_path, 'r') as hf:
    # Load data
    X = hf["X"][275]
    Y_time = hf["Y_time"][:]
    Y_ratio = hf["Y_ratio"][:]


with h5py.File(file_path, 'r') as hf:
    X_full = hf["X"][:]
    Y_time_full = hf["Y_time"][:]
    Y_ratio_log = hf["Y_ratio"][:] # Rename to indicate it is log scale

    # --- CRITICAL FIX: Convert Log Ratio to Linear Ratio ---
    Y_ratio_linear = 10 ** Y_ratio_log 

    # --- STEP 1: CLASSIFY ---
    Y_classes_full = []

    for time, ratio in zip(Y_time_full, Y_ratio_linear): # Use the linear version here!
        
        # 1. Merger: Recent (0.5 Gyr) and Major (> 1:4 ratio)
        if time < 0.5 and ratio > 0.25:
            Y_classes_full.append(1) 
        
        # 2. Non-Merger: Old (> 2 Gyr) OR very minor (< 1:100 ratio)
        elif time >= 2 or ratio <= 0.01:
            Y_classes_full.append(0) 
            
        # 3. In-between: Everything else
        else:
            Y_classes_full.append(2)

    Y_classes_full = np.array(Y_classes_full)

    # Check the counts now
    unique, counts = np.unique(Y_classes_full, return_counts=True)
    print("Class Distribution:", dict(zip(unique, counts)))

    # --- STEP 2: BALANCED UNDERSAMPLING ---
    # Find the size of the smallest class to balance everything to that level
    min_class_size = counts.min() 
    print(f"Undersampling to {min_class_size} samples per class.")
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

selected_indices = []

for class_id in [0, 1, 2]:
    # Find indices where the class is class_id
    indices = np.where(Y_classes_full == class_id)[0]
    
    # Randomly pick 'min_class_size' indices
    if len(indices) >= min_class_size:
        chosen = np.random.choice(indices, min_class_size, replace=False)
        selected_indices.extend(chosen)
    else:
        # Fallback if a class is empty (prevents crash)
        selected_indices.extend(indices)

selected_indices = np.array(selected_indices)
np.random.shuffle(selected_indices) # Shuffle so classes are mixed

X_new = X_full[selected_indices]
Y_classes_new = Y_classes_full[selected_indices]
Y_time_new = Y_time_full[selected_indices]
Y_ratio_new = Y_ratio_log[selected_indices]

print(f"New Dataset Size: {len(Y_classes_new)}")
print(f"New Class Counts: 0:{list(Y_classes_new).count(0)}, 1:{list(Y_classes_new).count(1)}, 2:{list(Y_classes_new).count(2)}")


plt.figure(figsize=(8,6))
colors = {0: 'blue', 1: 'red', 2: 'green'}
labels = {0: 'Non-Merger', 1: 'Merger', 2: 'In-Between'}

for c in [0, 1, 2]:
    mask = Y_classes_new == c
    plt.scatter(Y_time_new[mask], Y_ratio_new[mask], 
                c=colors[c], label=labels[c], s=10, alpha=0.6)

plt.xlabel("Time since merger [Gyr]")
plt.ylabel("Mass Ratio")
plt.title("Balanced Dataset Distribution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


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


### VIEW S1 MAPS ###
base_path = r'figures\s1'

fig = plt.figure(figsize=(10, 7))

# Loop from 0 to 7 (8 images total)
for i in range(8):
    # Construct the file path dynamically
    # matching your pattern: 4_0.png, 4_1.png, etc.
    filename = f'4_{i}.png'
    full_path = os.path.join(base_path, filename)
    
    # 1. Load the image (The step you missed)
    try:
        img_data = mpimg.imread(full_path)
    except FileNotFoundError:
        print(f"Error: Could not find file {full_path}")
        continue

    # 2. Create the subplot
    # (rows, cols, index) -> index starts at 1, so we use i + 1
    plt.subplot(2, 4, i + 1)
    
    # 3. Display the image data
    plt.imshow(img_data)
    
    # 4. Styling
    plt.axis('off')
    # Use f-string to insert the variable into the LaTeX string
    plt.title(rf"$\theta = {i}$")

plt.tight_layout() # distinct separation between subplots
plt.show()



# Subplot 2: Ratio
sns.histplot(Y_ratio, color="red", label="Original", kde=True, stat="density", linewidth=0, alpha=0.4, ax=axes[1])
sns.histplot(Y_ratio_new, color="blue", label="Undersampled", kde=True, stat="density", linewidth=0, alpha=0.4, ax=axes[1])
axes[1].set_title("Distribution of Log Mass Ratio")
axes[1].legend()

plt.tight_layout()
plt.show()

