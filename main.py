import argparse
import pickle

import torch
from PIL import Image
import os.path
import numpy as np

import cv2

import monodepth
import segmentation

from torchvision.transforms import Compose

import lmdb

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="./input", help="folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="./",
        help="folder for output images",
    )

    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_large",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")
    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    default_models = {
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }

    if args.model_weights is None:
        print(args.model_type)
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if args.model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=args.model_weights,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif args.model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=args.model_weights,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif args.model_type == "dpt_hybrid_kitti":
        net_w = 1216
        net_h = 352

        model = DPTDepthModel(
            path=args.model_weights,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif args.model_type == "dpt_hybrid_nyu":
        net_w = 640
        net_h = 480

        model = DPTDepthModel(
            path=args.model_weights,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif args.model_type == "midas_v21":  # Convolutional model
        net_w = net_h = 384

        model = MidasNet_large(args.model_weights, non_negative=True)
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        assert (
            False
        ), f"model_type '{args.model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    print(args.model_weights)

    model.eval()

    if args.optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    img_env = lmdb.open("img_lmdb", map_size=int(1e9))
    dep_env = lmdb.open("dep_lmdb", map_size=int(1e9))
    seg_env = lmdb.open("seg_lmdb", map_size=int(1e9))

    if os.path.isdir(args.input_path):
        for i, file in enumerate(os.listdir(args.input_path)):

            if os.path.isdir(file):
                continue

            print("Processing: " + file)

            img = np.asarray(Image.open(os.path.join(args.input_path, file)).convert('RGB'))
            depth_map = monodepth.run(os.path.join(args.input_path, file), model, device, transform, args)
            segmentation_map, areas, labels = segmentation.run(os.path.join(args.input_path, file))

            # print(img.shape)
            # print(depth_map.shape)
            # print(segmentation_map.shape)

            # print("Image:", img)
            # print("Depth Map:", depth_map)
            # print("Segmentation Map:", segmentation_map)

            with img_env.begin(write=True) as txn:
                txn.put(file.encode("ascii"), pickle.dumps(img))

            with dep_env.begin(write=True) as txn:
                txn.put(file.encode("ascii"), pickle.dumps(depth_map))

            with seg_env.begin(write=True) as txn:
                txn.put(file.encode("ascii"), pickle.dumps(segmentation_map))
                txn.put((file + "_areas").encode("ascii"), pickle.dumps(areas))
                txn.put((file + "_labels").encode("ascii"), pickle.dumps(labels))

