# EMR-Merging

## What's New:

We release the checkpoints when merging 30 ViT-B/16 models [here](https://drive.google.com/drive/folders/1KZv7RHIuNGzvjaVBJ7zkUxP1tG-4bGlI?usp=sharing). All the models are finetuned from [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224). All the datasets for these checkpoints are open-source and can be found in the paper's references. We will also release them soon.

The checkpoints of RoBERTa models finetuned on the GLUE benchmark are from DARE ([arxiv](https://arxiv.org/abs/2311.03099), [code](https://github.com/yule-BUAA/MergeLM)). You can download them [here](https://huggingface.co/vanillaOVO/roberta_base_glue_ckpts). Please follow [DARE](https://github.com/yule-BUAA/MergeLM) to download the GLUE dataset.

The IA3 checkpoints are released by TIES-Merging ([arxiv](https://arxiv.org/abs/2306.01708), [code](https://github.com/prateeky2806/ties-merging/tree/main)). You can download them [here](https://drive.google.com/drive/folders/1V2-SLOgK248TQBMP2i_cEdQnxB2jM2E1?usp=sharing).

The [BEiT3](https://arxiv.org/abs/2208.10442) checkpoints are released by Microsoft [here](https://github.com/microsoft/unilm/tree/master/beit3). We merge models finetuned on **BEiT3-base**. The finetuned checkpoints include *beit3_base_patch16_480_vqa.pth*, *beit3_base_patch16_480_coco_captioning.pth*, *beit3_base_patch16_384_coco_retrieval.pth*, *beit3_base_patch16_224_nlvr2.pth*, and *beit3_base_patch16_224_in1k.pth*. You can follow the doc and download the corresponding pre-trained and finetuned checkpoints.

The GPT-2 checkpoints and their corresponding datasets can be found in [FusionBench](https://github.com/tanganke/fusion_bench), a benchmark for model merging. 

## Get Started

This repository is the official implementation of EMR-Merging.

We realize tuning-free and high-performance model merging.

We provide the code for merging ViT models. In the future, we will release the code under different settings including NLP, PEFT, and multi-modal.

EMR-Merging requires no additional training. We merge models finetuned on different tasks and evaluate the merged model.

### Dependencies

Please follow [task_vectors](https://github.com/mlfoundations/task_vectors) to install the dependencies.

### Checkpoints and Datasets

You can download the fine-tuned checkpoints from the [task_vectors#checkpoints](https://github.com/mlfoundations/task_vectors#checkpoints).
The Google Drive folder is: [task_vectors_checkpoints](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw)
Refer to dataset processing in the [task_vectors](https://github.com/mlfoundations/task_vectors) and [AdaMerging](https://github.com/EnnengYang/AdaMerging).



## Eval

Run Task Arithmetic [paper](https://arxiv.org/abs/2212.04089)
> python main_task_arithmetic.py

Run TIES-MERGING [paper](https://arxiv.org/abs/2306.01708)
> python main_ties_merging.py

Run Layer-wise AdaMerging++ [paper](https://arxiv.org/abs/2310.02575)
> python main_layer_wise_adamergingpp.py

Check [here](https://github.com/EnnengYang/AdaMerging) if you want to load the trained merge coefficients for AdaMerging.

Run EMR-Merging (Ours)
> python main_emr_merging.py


# Acknowledgement
Our implementation references the code below, thanks to them.

- DARE: https://github.com/yule-BUAA/MergeLM

- Fusion Bench: https://github.com/tanganke/fusion_bench

- AdaMerging: https://github.com/EnnengYang/AdaMerging

- Task Arithmetic: https://github.com/mlfoundations/task_vectors

- TIES-MERGING: https://github.com/prateeky2806/ties-merging/tree/main

- Model Soups: https://github.com/mlfoundations/model-soups


