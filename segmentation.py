"""Compute segmentation maps for images in the input folder.
"""

import numpy as np
from skimage import io
from skimage.segmentation import felzenszwalb

from split_and_merge import connected_component


def run(img_name):

    img = io.imread(img_name)
    prediction = felzenszwalb(img, scale=1000, sigma=0.5, min_size=100)
    prediction = connected_component(prediction)

    areas= []
    labels =[]

    for i in range(1 , 1000):
        a = prediction == i
        area = np.sum(a)
        if area > 0:
            areas.append(area)
            labels.append(i)

    return prediction

    # mask_dset = dbo_mask.create_dataset(os.path.basename(img_name), data=prediction,dtype= np.uint8)
    # mask_dset.attrs['area'] = np.array(areas)
    # mask_dset.attrs['label'] = np.array(labels)