# EMR-Merging

This is the official implementation of our NeurIPS 2024 Spotlight Paper: **EMR-Merging: Tuning-Free High-Performance Model Merging** ([arxiv](https://arxiv.org/abs/2405.17461)).
We realize tuning-free and high-performance model merging.

We provide the code for merging [ViT models](./merge_vit), [language models](./merge_lm) (including RoBERTa and GPT-2), and [BEiT-3 models](./merge_beit3).

<img src='./png/method-main.jpg'>

**Method Framework**: In the (a) Merging Procedure, we merge task-specific vectors into a unified task vector and lightweight task-specific modulators to modulate direction and amplitude. During the (b) Inference Procedure, we apply the corresponding mask and rescaler to the unified task vector to obtain a specific task vector. The process of (c)Task-specific Direction and Amplitude Modulation includes obtaining task-specific masks and scalers.



## Citation
If you find this project helpful for you, feel free to cite our paper:
```
@inproceedings{huang_emrmerging,
 author = {Huang, Chenyu and Ye, Peng and Chen, Tao and He, Tong and Yue, Xiangyu and Ouyang, Wanli},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {122741--122769},
 publisher = {Curran Associates, Inc.},
 title = {EMR-Merging: Tuning-Free High-Performance Model Merging},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/dda5cac5272a9bcd4bc73d90bc725ef1-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}

```


## Acknowledgement
Our implementation references the code below, thanks to them.

- DARE: https://github.com/yule-BUAA/MergeLM

- FusionBench: https://github.com/tanganke/fusion_bench

- AdaMerging: https://github.com/EnnengYang/AdaMerging

- Task Arithmetic: https://github.com/mlfoundations/task_vectors

- TIES-MERGING: https://github.com/prateeky2806/ties-merging/tree/main

- Model Soups: https://github.com/mlfoundations/model-soups

- BEiT-3: https://github.com/microsoft/unilm/tree/master/beit3

  
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=harveyhuang18/EMR_Merging&type=Timeline)](https://star-history.com/#harveyhuang18/EMR_Merging&Timeline)

