# Merging Vision Transformers (ViTs)

## Get Started

### Dependencies

Please follow [task_vectors](https://github.com/mlfoundations/task_vectors) to install the dependencies.

### Checkpoints 

You can download the fine-tuned checkpoints from the [task_vectors#checkpoints](https://github.com/mlfoundations/task_vectors#checkpoints).
The Google Drive folder is: [task_vectors_checkpoints](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw)

Please follow [doc](./checkpoints/README.md) to place these checkpoints.

### Datasets

Please follow [Adamerging](https://github.com/EnnengYang/AdaMerging?tab=readme-ov-file#datasets) to download the datasets.

Please follow [doc](./data/README.md) to place these datasets.


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


## Results

Results for our EMR-Merging are shown in [logs](./logs). 

## Extension: Merging 30 ViTs

We release the [checkpoints](https://drive.google.com/drive/folders/1KZv7RHIuNGzvjaVBJ7zkUxP1tG-4bGlI?usp=sharing) when merging 30 ViT-B/16 models. All the models are finetuned from [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224). All the datasets for these checkpoints are open-source and can be found in the paper's references. 
