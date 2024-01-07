import os
import glob
import pickle
from pathlib import Path
import numpy as np
import cv2
import h5py
import tqdm


def load_pkl(fpath):
    with open(fpath, "rb") as f:
        return pickle.load(f, encoding="latin")

def load_h5py(fpath):
    return h5py.File(fpath, "r")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../datasets/zju_mocap/", help="path to the PeopleSnapshotData")
    parser.add_argument("--subject", type=str, default="394", help="sequence to process")
    args = parser.parse_args()

    dirpath = os.path.join(args.root, args.subject)
    assert os.path.exists(dirpath), f"Cannot open {dirpath}"
    dirpath = Path(dirpath)

    outdir = Path(f"./data/ZJUMoCap/{args.subject}/")
    os.makedirs(outdir, exist_ok=True)

    breakpoint()

    # # load camera
    # camera = load_pkl(dirpath / "cameras.pkl")

    # K = camera['frame_000000']['intrinsics']
    # w2c = np.eye(4)
    # dist_coeffs = camera['frame_000000']['distortions']

    # H, W = None, None

    # # load images
    # image_dir = outdir / "images"
    # os.makedirs(image_dir, exist_ok=True)

    # print("Write images to", image_dir)
    # image_files = sorted(glob.glob(str(dirpath / "images/*")))
    # frame_cnt = len(image_files)
    # for i in tqdm.trange(frame_cnt):
    #     img_path = f"{image_dir}/image_{i:04d}.png"
    #     frame = cv2.imread(image_files[i])

    #     H, W = frame.shape[0], frame.shape[1]

    #     frame = cv2.undistort(frame, K, dist_coeffs)
    #     cv2.imwrite(img_path, frame)

    # # load masks
    # mask_dir = outdir / "masks"
    # os.makedirs(mask_dir, exist_ok=True)

    # print("Write mask to", mask_dir)
    # mask_files = sorted(glob.glob(str(dirpath / "masks/*")))
    # frame_cnt = len(mask_files)
    # for i in tqdm.trange(frame_cnt):
    #     mask_path = f"{mask_dir}/mask_{i:04d}.npy"
    #     mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
    #     mask = cv2.undistort(mask, K, dist_coeffs)
    #     np.save(mask_path, mask)


    # camera_path = outdir / "cameras.npz"
    # np.savez(str(camera_path), **{
    #     "intrinsic": K,
    #     "extrinsic": w2c,
    #     "height": H,
    #     "width": W,
    # })
    # print("Write camera to", camera_path)

    smpl_param_files = sorted(glob.glob(f"../ZJU-Mocap/CoreView_{args.subject}/new_params/*"))

    betas = None
    thetas = None
    transl = None

    for i in range(len(smpl_param_files)):
        smpl_params = np.load(f"../ZJU-Mocap/CoreView_{args.subject}/new_params/{i}.npy", allow_pickle=True).item()
        if betas is None:
            betas = smpl_params['shapes'][0]

        if thetas is None:
            thetas = smpl_params['poses']
            thetas[i, :3] = smpl_params['Rh'][0]
        else:
            thetas = np.concatenate((thetas, smpl_params['poses']), axis=0)
            thetas[i, :3] = smpl_params['Rh'][0]

        if transl is None:
            transl = smpl_params['Th']
        else:
            transl = np.concatenate((transl, smpl_params['Th']), axis=0)

    smpl_params = {
        "betas": betas.astype(np.float32),
        "thetas": thetas.astype(np.float32),
        "transl": transl.astype(np.float32),
    }
    np.savez(str(outdir / "poses.npz"), **smpl_params)
