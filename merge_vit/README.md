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
We also release them here.

| Dataset       | Lib                                | URL                                                          |
| ------------- | ---------------------------------- | ------------------------------------------------------------ |
| MNIST         | torchvision.datasets.MNIST         | https://pytorch.org/vision/0.18/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST |
| CIFAR10       | torchvision.datasets.CIFAR10       | https://pytorch.org/vision/0.18/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10 |
| Vegetables    | torchvision.datasets.ImageFolder   | https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset |
| Food-101      | torchvision.datasets.Food101       | https://pytorch.org/vision/0.18/generated/torchvision.datasets.Food101.html#torchvision.datasets.Food101 |
| Kvasir-V2     | torchvision.datasets.ImageFolder   | https://dl.acm.org/do/10.1145/3193289/full/packageFiles/kvasir-dataset-v2-1606843201547.zip |
| Intel-Images  | torchvision.datasets.ImageFolder   | https://www.kaggle.com/datasets/puneet6060/intel-image-classification |
| Cars          | torchvision.datasets.StanfordCars  | https://pytorch.org/vision/0.18/generated/torchvision.datasets.StanfordCars.html#torchvision.datasets.StanfordCars |
| EuroSAT       | torchvision.datasets.EuroSAT       | https://pytorch.org/vision/0.18/generated/torchvision.datasets.EuroSAT.html#torchvision.datasets.EuroSAT |
| Weather       | torchvision.datasets.ImageFolder   | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/M8JQCR |
| Cats & Dogs   | torchvision.datasets.ImageFolder   | https://www.kaggle.com/competitions/dogs-vs-cats             |
| MangoLeafBD   | torchvision.datasets.ImageFolder   | https://data.mendeley.com/datasets/hxsnvwty3r/1              |
| beans         | torchvision.datasets.ImageFolder   | https://huggingface.co/datasets/AI-Lab-Makerere/beans        |
| CIFAR100      | torchvision.datasets.CIFAR100      | https://pytorch.org/vision/0.18/generated/torchvision.datasets.CIFAR100.html#torchvision.datasets.CIFAR100 |
| GTSRB         | torchvision.datasets.GTSRB         | https://pytorch.org/vision/0.18/generated/torchvision.datasets.GTSRB.html#torchvision.datasets.GTSRB |
| SVHN          | torchvision.datasets.SVHN          | https://pytorch.org/vision/0.18/generated/torchvision.datasets.SVHN.html#torchvision.datasets.SVHN |
| Dogs          | torchvision.datasets.ImageFolder   | http://vision.stanford.edu/aditya86/ImageNetDogs/            |
| FashionMNIST  | torchvision.datasets.FashionMNIST  | https://pytorch.org/vision/0.18/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST |
| OxfordIIITPet | torchvision.datasets.OxfordIIITPet | https://pytorch.org/vision/0.18/generated/torchvision.datasets.OxfordIIITPet.html#torchvision.datasets.OxfordIIITPet |
| Landscape     | torchvision.datasets.ImageFolder   | https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images |
| Flowers       | torchvision.datasets.ImageFolder   | https://www.kaggle.com/datasets/alxmamaev/flowers-recognition |
| STL10         | torchvision.datasets.STL10         | https://pytorch.org/vision/0.18/generated/torchvision.datasets.STL10.html#torchvision.datasets.STL10 |
| CUB-200-2011  | torchvision.datasets.ImageFolder   | http://www.vision.caltech.edu/datasets/cub_200_2011/         |
| EMNIST        | torchvision.datasets.EMNIST        | https://pytorch.org/vision/0.18/generated/torchvision.datasets.EMNIST.html#torchvision.datasets.EMNIST |
| DTD           | torchvision.datasets.DTD           | https://pytorch.org/vision/0.18/generated/torchvision.datasets.DTD.html#torchvision.datasets.DTD |
| RESISC45      | torchvision.datasets.ImageFolder   | https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp |
| SUN397        | torchvision.datasets.SUN397        | https://pytorch.org/vision/0.18/generated/torchvision.datasets.SUN397.html#torchvision.datasets.SUN397 |
| KenyanFood13  | torchvision.datasets.ImageFolder   | https://www.dropbox.com/scl/fi/hk1llnnv6bpjw153epfxo/Food13.zip?rlkey=o7iq83g4g0xjeif45ibxd9kkb&e=1&dl=0 |
| Animal-10N    | torchvision.datasets.ImageFolder   | http://dm.kaist.ac.kr/datasets/animal-10n/                   |
| Garbage       | torchvision.datasets.ImageFolder   | https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification |
| Fruits-360    | torchvision.datasets.ImageFolder   | https://www.kaggle.com/datasets/moltean/fruits               |
