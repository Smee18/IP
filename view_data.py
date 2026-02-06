from keras import utils
import h5py
import matplotlib.pyplot as plt
import numpy as np
from photutils.psf.matching import TukeyWindow
from sky import sigma_clip

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

Q = 3.5
alpha = 0.06
taper_2d = TukeyWindow(alpha=0.6)((256, 256))

fig = plt.figure(figsize=(20, 8))

for i in range(10):

    all_indices_for_class = np.where(label_indices == i)[0]
    
    single_image_idx = all_indices_for_class[0]
    
    plt.subplot(2, 5, i + 1)

    image_color = images[single_image_idx]
    raw_flux = image_color[..., 1].astype(np.float32) / 255.0
    mask = sigma_clip(raw_flux, alpha=3) 
    sky_level = np.nanmedian(raw_flux[mask]) 
    sky_std = np.nanstd(raw_flux[mask]) 
    clean_flux = np.maximum(raw_flux - sky_level, 0)
    clean_flux[clean_flux < (2 * sky_std)] = 0 # Silent background
    tapered_flux = clean_flux * taper_2d
    final_img = np.arcsinh(alpha * Q * tapered_flux) / Q 
    
    plt.imshow(final_img, cmap='gray')
    plt.title(f"{label_info[i]}") 
    plt.axis('off') 

plt.tight_layout()
plt.show()
