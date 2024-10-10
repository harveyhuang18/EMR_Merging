# Merging BEiT-3 Models

We merge BEiT-3 models on 5 multi-modal tasks using EMR-Merging.

## Get Started

### Dependencies

Please follow [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3) to install dependencies and download datasets.

We merge models on ImageNet-1k, COCO-Captioning, NLVR2, VQAv2, and COCO-Retrieval.

We use checkpoints including *beit3-base (pretrained model)*, *beit3_base_patch16_480_vqa.pth*, *beit3_base_patch16_480_coco_captioning.pth*, *beit3_base_patch16_384_coco_retrieval.pth*, *beit3_base_patch16_224_nlvr2.pth*, and *beit3_base_patch16_224_in1k.pth*. 

You can follow the doc and download the checkpoints. Please put them in [./model_weights](./model_weights).

## Eval

run script_{in1k, nlvr2, retrieval, vqav2, captioning}.sh

