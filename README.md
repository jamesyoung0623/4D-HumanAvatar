# 4D-HumanAvatar

## Install the dependencies
```
conda create -n 4D-HumanAvatar python=3.10
conda activate 4D-HumanAvatar
bash install.sh
```

## Prepare Data
```bash
# Step 1: Download data from: https://graphics.tu-bs.de/people-snapshot
# Step 2: Preprocess using our script
python scripts/peoplesnapshot/preprocess_PeopleSnapshot.py --root <PATH_TO_PEOPLESNAPSHOT> --subject male-3-casual

# Step 3: Download SMPL from: https://smpl.is.tue.mpg.de/ and place the model in ./data/SMPLX/smpl/
# └── SMPLX/smpl/
#         ├── SMPL_FEMALE.pkl
#         ├── SMPL_MALE.pkl
#         └── SMPL_NEUTRAL.pkl
```

## Acknowledge
We would like to acknowledge the following third-party repositories we used in this project:
- [[Tinycuda-nn]](https://github.com/NVlabs/tiny-cuda-nn)
- [[Segment-anything]](https://github.com/facebookresearch/segment-anything)

Besides, we used code from:
- [[InstantAvatar]](https://github.com/tijiang13/InstantAvatar)
- [[Anim-NeRf]](https://github.com/JanaldoChen/Anim-NeRF)
- [[SelfRecon]](https://github.com/jby1993/SelfReconCode)
- [[lpips]](https://github.com/richzhang/PerceptualSimilarity)
- [[SMPLX]](https://github.com/vchoutas/smplx)
- [[pytorch3d]](https://github.com/facebookresearch/pytorch3d)

We are grateful to the developers and contributors of these repositories for their hard work and dedication to the open-source community. Without their contributions, our project would not have been possible.

