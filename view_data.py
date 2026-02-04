from keras import utils
import h5py
import matplotlib.pyplot as plt
import numpy as np

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
    
    plt.imshow(images[single_image_idx])
    plt.title(f"{label_info[i]}") 
    plt.axis('off') 

plt.tight_layout()
plt.show()