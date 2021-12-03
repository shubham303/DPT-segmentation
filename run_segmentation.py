"""Compute segmentation maps for images in the input folder.
"""
import os
import glob
import cv2
import argparse

import h5py
import torch
import torch.nn.functional as F

import util.io

from torchvision.transforms import Compose
from dpt.models import DPTSegmentationModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
import numpy as np
import matplotlib as plt

from split_and_merge import connected_component


def run(input_path, output_path, model_path, model_type="dpt_hybrid", optimize=True):
    """Run segmentation network

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    # get input
    img_names = glob.glob(os.path.join(input_path, "*.jpg"))
    num_images = len(img_names)

    # create output folder
    #os.makedirs(output_path, exist_ok=True)

    print("start processing")
    seg_db = h5py.File("./seg.h5", "w")
    dbo_mask = seg_db.create_group("mask")
    
    for ind, img_name in enumerate(img_names):
        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        from skimage import io
        from skimage.segmentation import felzenszwalb, flood_fill
        from skimage.segmentation import mark_boundaries

        img = io.imread(img_name)
        prediction = felzenszwalb(img, scale=1000, sigma=0.5, min_size=100)
        prediction = connected_component(prediction)
        
        areas= []
        labels =[]
        
        for i in range(1 , 1000):
            a = prediction==i
            area= np.sum(a)
            if area >0:
                areas.append(area)
                labels.append(i)
        
        mask_dset = dbo_mask.create_dataset(os.path.basename(img_name), data=prediction,dtype= np.uint8)
        mask_dset.attrs['area'] = np.array(areas)
        mask_dset.attrs['label'] = np.array(labels)
        
       
        #util.io.write_segm_img(filename, img, prediction, alpha=0.5)

    
        
    
    seg_db.close()
    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="./images", help="folder with input images"
    )

    parser.add_argument(
        "-o", "--output_path", default="./", help="folder for output images"
    )

    parser.add_argument(
        "-m",
        "--model_weights",
        default=None,
        help="path to the trained weights of model",
    )

    # 'vit_large', 'vit_hybrid'
    parser.add_argument("-t", "--model_type", default="dpt_large", help="model type")
    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")
    parser.set_defaults(optimize=True)

    args = parser.parse_args()

    default_models = {
        "dpt_large": "weights/dpt_large-ade20k-b12dca68.pt",
        "dpt_hybrid": "weights/dpt_hybrid-ade20k-53898607.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute segmentation maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
    )
