from keras import utils
import h5py
import matplotlib.pyplot as plt
import numpy as np
from photutils.psf.matching import TukeyWindow

# To get the images and labels from file
with h5py.File('data/Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])

label_indices = labels.copy() 
labels_onehot = utils.to_categorical(labels, 10)

label_info = {
0: "Disturbed Galaxies",
1: "Merging Galaxies",
2: "Round Smooth Galaxies",
3: "In-between Round Smooth Galaxies",
4: "Cigar Shaped Smooth Galaxies",
5: "Barred Spiral Galaxies",
6: "Unbarred Tight Spiral Galaxies",
7: "Unbarred Loose Spiral Galaxies",
8: "Edge-on Galaxies without Bulge",
9: "Edge-on Galaxies with Bulge"
}

fig = plt.figure(figsize=(20, 8))

for i in range(10):

    all_indices_for_class = np.where(label_indices == i)[0]
    
    single_image_idx = all_indices_for_class[0]
    
    plt.subplot(2, 5, i + 1)

    image_color = images[single_image_idx]
    gray_img = np.dot(image_color[..., :3], [0.299, 0.587, 0.114]) / 255.0
    stretch = np.arcsinh(0.06 * 3.5 * stretch) // 3.5
    taper = TukeyWindow(alpha=0.4)
    final_img = stretch * taper
    
    plt.imshow(gray_img, cmap='gray')
    plt.title(f"{label_info[i]}") 
    plt.axis('off') 

plt.tight_layout()
plt.show()