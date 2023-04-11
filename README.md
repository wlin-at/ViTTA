# Video Test-Time Adaptation for Action Recognition (CVPR 2023)

**ViTTA** is the first approach of test-time adaptation of video action recognition models against common distribution shifts. ViTTA is tailored to saptio-temporal models and capable of adaptation on a single video sample at a step. It consists in a feature distribution alignment technique that aligns online estimates of test set statistics towards the training statistics. It further enforces prediction consistency over temporally augmented views of the same test video sample. 

Official implementation of ViTTA [[`arXiv`](https://arxiv.org/abs/2211.15393)]  
Author [HomePage](https://wlin-at.github.io/)

## Requirements
* Our experiments run on Python 3.6, PyTorch 1.7, mmcv-full 1.3.12. Other versions should work but are not tested. 
* Dependency of mmaction2 (for [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)):  
 $ `pip install mmcv-full==1.3.12`  
 $ `git clone https://github.com/SwinTransformer/Video-Swin-Transformer.git && cd Video-Swin-Transformer`  
 $ `pip install -v -e . --user`  
* Other relevant dependencies can be found in `requirements.txt`

---
## Data Preparation
* Download
Download required data for Experiments on UCF101 from [here](https://files.icg.tugraz.at/d/3551df694e3d4d6b89da/)  
`list_video_perturbations_ucf`: list of files for corrupted videos of UCF101 validation set (in 12 corruption types)  
`model_swin_ucf`: Video Swin Transformer trained on UCF101 training set  
`model_tanet_ucf`: TANet trained on UCF101 training set  
`model_tanet_ucf`: TANet trained on UCF101 training set  
`source_statistics_tanet_ucf`: precomputed source (UCF101 training set) statistics on TANet  
`source_statistics_swin_ucf`: precomputed source (UCF101 training set) statistics on Video Swin Transformer  
`ucf_corrupted_videos.zip`: Corrupted videos of UCF validation set (in 12 corruption types)  
* Data structure
lines in file list are in format 
`video_path n_frames class_id`  
video dataset structure
    ```
    level_5_ucf_val_split_1_/
      gauss/
        ApplyEyeMakeup/
          v_ApplyEyeMakeup_g01_c01.mp4
          v_ApplyEyeMakeup_g01_c02.mp4
          ...
        ApplyLipstick/
        ...
      contrast/
        ApplyEyeMakeup/
          v_ApplyEyeMakeup_g01_c01.mp4
          v_ApplyEyeMakeup_g01_c02.mp4
          ...
        ApplyLipstick/
      ...
    ```
## Citation
Thanks for citing our paper:
```bibtex
@inproceedings{lin2023video,
  title={Video Test-Time Adaptation for Action Recognition},
  author={Lin, Wei and Mirza, Muhammad Jehanzeb and Kozinski, Mateusz and Possegger, Horst and Kuehne, Hilde and Bischof, Horst},
  booktitle={CVPR},
  year={2023},
}
```
