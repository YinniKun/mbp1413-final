'''
Draws the PCA for all images in a folder
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage import io
from skimage.transform import resize

# Path to the folder containing the images
folder_path = '/Users/ric/mbp_temp/datasets/train/images'

# List of images in the folder
image_list = os.listdir(folder_path)

# load all images
images = []
for image in image_list:
    img = io.imread(os.path.join(folder_path, image))
    img = resize(img, (100, 100))
    images.append(img.flatten())

# Convert the list of images to a numpy array
images = np.array(images)

# Perform PCA
pca = PCA(n_components=2)
pca.fit(images)

# Project the images to the new space
images_pca = pca.transform(images)

# compute variance explained
variance_explained = pca.explained_variance_ratio_

# Plot the PCA
plt.scatter(images_pca[:, 0], images_pca[:, 1])
# label some points
plt.xlabel(f'PCA 1')
plt.ylabel(f'PCA 2')
plt.title('PCA for all images')
plt.savefig('pca.png')