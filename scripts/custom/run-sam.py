from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import glob
import os
import json


CHECKPOINT = os.path.expanduser("~/third_party/segment-anything/ckpts/sam_vit_h_4b8939.pth")
MODEL = "vit_h"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    sam = sam_model_registry[MODEL](checkpoint=CHECKPOINT)
    sam.to("cuda")
    predictor = SamPredictor(sam)

    root = args.data_dir
    img_path_list = sorted(glob.glob(f"{root}/images/*"))
    keypoints_path_list = sorted(glob.glob(f"{root}/keypoints/*"))

    os.makedirs(f"{root}/masks_sam", exist_ok=True)
    os.makedirs(f"{root}/masks_sam_images", exist_ok=True)
    for image_path, keypoints_path in zip(img_path_list, keypoints_path_list):
        image = cv2.imread(image_path)
        predictor.set_image(image)

        keypoints = json.load(open(keypoints_path))
        for i in range(len(keypoints)):
            pts = np.array(keypoints[i]['keypoints'])
            masks, _, _ = predictor.predict(pts, np.ones_like(pts[:, 0]))
            mask = masks.sum(axis=0) > 0
            cv2.imwrite(image_path.replace("images", "masks_sam"), mask * 255)

            image[~mask] = 0
            cv2.imwrite(image_path.replace("images", "masked_sam_images"), image)
            break
