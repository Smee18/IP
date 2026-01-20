import numpy as np
import matplotlib.pyplot as plt
import cv2

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

file_path = "HST_256x256_halfstellar32.hdf5"

with h5py.File(file_path, 'r') as hf:
    Y_time = hf["Y_time"][:]
    Y_ratio = hf["Y_ratio"][:]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')

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
'''