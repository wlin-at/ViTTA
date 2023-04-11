# Video Test-Time Adaptation for Action Recognition (CVPR 2023)

**ViTTA** is the first approach of test-time adaptation of video action recognition models against common distribution shifts. ViTTA is tailored to saptio-temporal models and capable of adaptation on a single video sample at a step. It consists in a feature distribution alignment technique that aligns online estimates of test set statistics towards the training statistics. It further enforces prediction consistency over temporally augmented views of the same test video sample. 

Official implementation of ViTTA [[`arXiv`](https://arxiv.org/abs/2211.15393)]  
Author [HomePage](https://wlin-at.github.io/)

## Requirements
* Our experiments run on Python 3.6, PyTorch 1.7, mmcv-full 1.3.12. Other versions should work but are not tested. 
* Relevant dependencies are provided in `requirements.txt`

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
