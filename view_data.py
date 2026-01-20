import numpy as np
import matplotlib.pyplot as plt
import cv2
'''
M_input = 36
M_crop = 32


data = np.load(r"data/preprocessed-galaxy-classification-36x36.npz")
X = data['X']
Y_ratio = data['Y_ratio']
Y_time = data['Y_time']
Y_is_merger = data['Y_is_merger']
Y_stellar_mass = data['Y_stellar_mass']

print(len(X))

print(f"{np.count_nonzero(Y_is_merger)} mergers | {len(Y_is_merger) - np.count_nonzero(Y_is_merger)} non mergers")
print(f"Shape of entry: {X[42].shape}")
print(f"Example is a {Y_is_merger[42]}")
merged = cv2.merge(X[42])
#print(merged)
margin = (M_input - M_crop) // 2
X_cropped = X[:, :, margin:margin+M_crop, margin:margin+M_crop]
print(f"Cropped shape: {X_cropped.shape}")
fig = plt.figure(figsize=(10, 7))

# Add the first image to the figure (top-left position)
plt.subplot(1, 3, 1)  # 2 rows, 2 columns, first position
plt.imshow((X_cropped[42, 0]))  
plt.axis('off')  # Hide the axis labels
plt.title("G channel (green)") 

# Add the second image to the figure (top-right position)
plt.subplot(1, 3, 2)  # 2 rows, 2 columns, second position
plt.imshow((X_cropped[42, 1]))  
plt.axis('off')  # Hide the axis labels
plt.title("R channel (red)") 

# Add the third image to the figure (bottom-left position)
plt.subplot(1, 3, 3)  # 2 rows, 2 columns, third position
plt.imshow((X_cropped[42, 2])) 
plt.axis('off')  # Hide the axis labels
plt.title("I Channel (infrared)")  

plt.show()

'''
import h5py
import matplotlib.pyplot as plt
import numpy as np

def make_rgb(img_data):
    """
    Convert (3, H, W) Magnitude data into (H, W, 3) RGB Flux-like image.
    Mapping:
        Red   <- H Channel (Index 2, Infrared)
        Green <- I Channel (Index 1, Red/NIR)
        Blue  <- B Channel (Index 0, Blue)
    """
    # 1. Select channels and transpose to (Height, Width, Channels)
    # We map H->Red, I->Green, B->Blue to mimic physical color
    rgb = np.stack([img_data[2], img_data[1], img_data[0]], axis=-1)
    
    # 2. Invert Magnitudes to mimic Flux (Visual Brightness)
    # Assuming data is normalized roughly [0, 1] where 1 is faint background
    rgb = 1.0 - rgb 
    
    # 3. Clip to [0, 1] to avoid matplotlib warnings
    rgb = np.clip(rgb, 0, 1)
    
    return rgb

file_path = r"data/HST_256x256_halfstellar32.hdf5"

with h5py.File(file_path, 'r') as hf:
    X = hf["X"][35]
    Y_time = hf["Y_time"][:]
    Y_ratio = hf["Y_ratio"][:]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')

print(len(Y_time), len(Y_ratio))

hist, xedges, yedges = np.histogram2d(Y_ratio, Y_time, bins=50)

x_width = (xedges[1] - xedges[0])
y_width = (yedges[1] - yedges[0])

xpos, ypos = np.meshgrid(xedges[:-1] + x_width/2, yedges[:-1] + y_width/2, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)

# Dimensions of bars
dx = x_width * 0.9 * np.ones_like(zpos) # 0.9 makes gaps between bars
dy = y_width * 0.9 * np.ones_like(zpos)
dz = hist.ravel()

# Color by height (optional, makes it easier to see)
cm = plt.get_cmap('plasma')
col = [cm(h/dz.max()) for h in dz]

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=col, zsort='average')

ax.set_xlabel('Log Mass Ratio')
ax.set_ylabel('Time Since Merger')
ax.set_zlabel('Count')
ax.set_title(f'3D Histogram')

plt.show()


fig = plt.figure(figsize=(10, 7))

# Add the first image to the figure (top-left position)
plt.subplot(1, 4, 1)  # 2 rows, 2 columns, first position
plt.imshow((X[0]), cmap='gray_r')  
plt.axis('off')  # Hide the axis labels
plt.title("B channel (Blue)") 

# Add the second image to the figure (top-right position)
plt.subplot(1, 4, 2)  # 2 rows, 2 columns, second position
plt.imshow((X[1]), cmap='gray_r')  
plt.axis('off')  # Hide the axis labels
plt.title("I channel (Red)") 

# Add the third image to the figure (bottom-left position)
plt.subplot(1, 4, 3)  # 2 rows, 2 columns, third position
plt.imshow((X[2]), cmap='gray_r') 
plt.axis('off')  # Hide the axis labels
plt.title("H Channel (Infrared)")  

plt.subplot(1, 4, 4)  # 2 rows, 2 columns, third position
plt.imshow((make_rgb(X)), cmap='gray_r') 
plt.axis('off')  # Hide the axis labels
plt.title("Merged")

plt.show()


N_SAMPLES = 64
GRID_SIZE = 8 # 10x10 grid

with h5py.File(file_path, 'r') as hf:
    # Load first 100 images
    # Shape is likely (100, 3, 256, 256)
    X_batch = hf["X"][:N_SAMPLES]


fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(20, 20))
fig.suptitle(f"First {N_SAMPLES} Galaxy Samples (RGB Composite)", fontsize=16)

# Flatten axes array for easy iteration
axes_flat = axes.flatten()

for i in range(N_SAMPLES):
    ax = axes_flat[i]
    
    # Process image
    img_rgb = make_rgb(X_batch[i])
    
    ax.imshow(img_rgb)
    ax.axis('off')
    
    # Optional: Add small label for mass ratio (mu) or time (t)
    # ax.text(5, 25, f"t={Y_time_batch[i]:.1f}", color='white', fontsize=8, backgroundcolor='black')

plt.tight_layout()
plt.subplots_adjust(top=0.95, wspace=0.05, hspace=0.05)
plt.show()