# EMR-Merging
This repository is the official implementation of EMR-Merging.

We realize data-less and high-performance model merging.

We provide the code for merging ViT models. In the future, we will release the code under different settings including NLP, PEFT, and multi-modal.

# Get Started

EMR-Merging requires no additional training. We merge models finetuned on different tasks and evaluate the merged model.

### Checkpoints

You can download the fine-tuned checkpoints from the [task_vectors#checkpoints](https://github.com/mlfoundations/task_vectors#checkpoints).
The Google Drive folder is: [task_vectors_checkpoints](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw)


### Datasets
Refer to dataset processing in the [task_vectors](https://github.com/mlfoundations/task_vectors) and [AdaMerging](https://github.com/EnnengYang/AdaMerging).



## Eval

Run Task Atithmetic [paper](https://arxiv.org/abs/2212.04089)
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
- AdaMerging: https://github.com/EnnengYang/AdaMerging

- Task Arithmetic: https://github.com/mlfoundations/task_vectors

- TIES-MERGING: https://github.com/prateeky2806/ties-merging/tree/main

- Model Soups: https://github.com/mlfoundations/model-soups


