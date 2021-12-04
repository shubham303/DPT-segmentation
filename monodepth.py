"""Compute depth maps for images in the input folder.
"""

import torch
import util.io

def run(img_name, model, device, transform, args):
    # input
    img = util.io.read_image(img_name)

    # if img is None:
    #     continue

    if args.kitti_crop is True:
        height, width, _ = img.shape
        top = height - 352
        left = (width - 1216) // 2
        img = img[top: top + 352, left: left + 1216, :]

    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

        if args.optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
                .squeeze()
                .cpu()
                .numpy()
        )

        if args.model_type == "dpt_hybrid_kitti":
            prediction *= 256

        if args.model_type == "dpt_hybrid_nyu":
            prediction *= 1000.0

    return prediction
