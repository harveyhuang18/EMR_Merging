from datasets import RetrievalDataset
from transformers import XLMRobertaTokenizer
from datasets import CaptioningDataset
from transformers import XLMRobertaTokenizer
from datasets import ImageNetDataset
from datasets import VQAv2Dataset

if __name__ == '__main__':
    tokenizer = XLMRobertaTokenizer("/remote-home/yepeng2/beit3/model_weights/beit3.spm")

    # RetrievalDataset.make_coco_dataset_index(
    #     data_path="/remote-home/yepeng2/ds/COCO",
    #     tokenizer=tokenizer,
    # )

    # CaptioningDataset.make_coco_captioning_dataset_index(
    #     data_path="/remote-home/yepeng2/ds/COCO",
    #     tokenizer=tokenizer,
    # )

    # ImageNetDataset.make_dataset_index(
    #     train_data_path="/remote-home/yepeng2/PycharmProjects/Datasets/ImageNet_1k/train",
    #     val_data_path="/remote-home/yepeng2/PycharmProjects/Datasets/ImageNet_1k/val",
    #     index_path="/remote-home/yepeng2/PycharmProjects/Datasets/ImageNet_1k"
    # )




    # tokenizer = XLMRobertaTokenizer("/your_beit3_model_path/beit3.spm")

    VQAv2Dataset.make_dataset_index(
        data_path="/remote-home/yepeng2/ds/COCO",
        tokenizer=tokenizer,
        annotation_data_path="/remote-home/yepeng2/ds/COCO/vqa",
    )