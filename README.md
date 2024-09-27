# EMR-Merging

This is the official implementation of our NeurIPS 2024 Spotlight Paper: **EMR-Merging: Tuning-Free High-Performance Model Merging** ([arxiv](https://arxiv.org/abs/2405.17461)).

We realize tuning-free and high-performance model merging.

We provide the code for merging ViT models. We will release the code under different settings including NLP, PEFT, and multi-modal.

EMR-Merging requires no additional training. We merge models finetuned on different tasks and evaluate the merged model.

<img src='./png/method_main.png'>

**Method Framework**: In the (a) Merging Procedure, we merge task-specific vectors into a unified task vector and lightweight task-specific modulators to modulate direction and amplitude. During the (b) Inference Procedure, we apply the corresponding mask and rescaler to the unified task vector to obtain a specific task vector. The process of (c)Task-specific Direction and Amplitude Modulation includes obtaining task-specific masks and scalers.




## Checkpoints for other benchmarks (Code coming soon)

We release the [checkpoints](https://drive.google.com/drive/folders/1KZv7RHIuNGzvjaVBJ7zkUxP1tG-4bGlI?usp=sharing) when merging 30 ViT-B/16 models. All the models are finetuned from [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224). All the datasets for these checkpoints are open-source and can be found in the paper's references. 

The RoBERTa [checkpoints](https://huggingface.co/vanillaOVO/roberta_base_glue_ckpts) finetuned on the GLUE benchmark are from DARE ([arxiv](https://arxiv.org/abs/2311.03099), [code](https://github.com/yule-BUAA/MergeLM)). Please follow [DARE](https://github.com/yule-BUAA/MergeLM) to download the GLUE dataset.

The IA3 [checkpoints](https://drive.google.com/drive/folders/1V2-SLOgK248TQBMP2i_cEdQnxB2jM2E1?usp=sharing) are released by TIES-Merging ([arxiv](https://arxiv.org/abs/2306.01708), [code](https://github.com/prateeky2806/ties-merging/tree/main)).

The BEiT3 [checkpoints](https://arxiv.org/abs/2208.10442) are released by Microsoft [here](https://github.com/microsoft/unilm/tree/master/beit3). We merge models finetuned on **BEiT3-base**. The finetuned checkpoints include *beit3_base_patch16_480_vqa.pth*, *beit3_base_patch16_480_coco_captioning.pth*, *beit3_base_patch16_384_coco_retrieval.pth*, *beit3_base_patch16_224_nlvr2.pth*, and *beit3_base_patch16_224_in1k.pth*. You can follow the doc and download the corresponding pre-trained and finetuned checkpoints.

The GPT-2 [checkpoints](https://huggingface.co/collections/tanganke/gpt-2-models-fine-tuned-on-tasks-from-glue-benchmark-664ab37d9e33e622679f541b) and their corresponding datasets are from FusionBench ([arxiv](https://arxiv.org/abs/2406.03280), [code](https://github.com/tanganke/fusion_bench)), a benchmark for model merging. 



## Citation
If you find this project helpful for you, feel free to cite our paper:
```
@article{huang2024emr,
  title={EMR-Merging: Tuning-Free High-Performance Model Merging},
  author={Huang, Chenyu and Ye, Peng and Chen, Tao and He, Tong and Yue, Xiangyu and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2405.17461},
  year={2024}
}
```


# Acknowledgement
Our implementation references the code below, thanks to them.

- DARE: https://github.com/yule-BUAA/MergeLM

- FusionBench: https://github.com/tanganke/fusion_bench

- AdaMerging: https://github.com/EnnengYang/AdaMerging

- Task Arithmetic: https://github.com/mlfoundations/task_vectors

- TIES-MERGING: https://github.com/prateeky2806/ties-merging/tree/main

- Model Soups: https://github.com/mlfoundations/model-soups


