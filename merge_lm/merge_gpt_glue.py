import datasets.arrow_dataset
from tqdm import tqdm
import numpy
from datasets import load_dataset, load_from_disk
import copy
import os
import sys

import transformers
from utils.utils import set_random_seed
from model_merging_methods.merging_methods import MergingMethod

import sys
import json
import argparse
from torch.utils.data import DataLoader
import time
import logging
from functools import partial
from torchmetrics import Accuracy, MeanMetric
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, GPT2Tokenizer#, GPT2ForSequenceClassification
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.glue_data_loader import GLUEDataLoader
from utils.metrics import compute_metrics
from utils.customized_trainers import CustomizedTrainer
from model_merging_methods.mask_weights_utils import mask_model_weights
from utils.load_config import cache_dir

from transformers import (
    GPT2ForSequenceClassification,
    GPT2Model,
    GPT2Tokenizer,
    default_data_collator,
    AutoConfig
)


def get_emr_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list,
                          tokenizer: transformers.AutoTokenizer, logger):
    logger.info(f"configuration is {args}")
    merging_method = MergingMethod(merging_method_name='emr_merging')

    merged_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to(args.device)
    pretrained_model = copy.deepcopy(merged_model)
    pretrained_model.to('cpu')
    pretrained_param_dict = {param_name: param_value for param_name, param_value in
                             pretrained_model.named_parameters()}
    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    # exclude parameter whose name matches "classifier"
    Vector_unified, masks, rescales = merging_method.get_emr_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*score.*"],
                                                   models_use_deepcopy=True)

    for idx, (dataset_name, model_to_merge) in enumerate(zip(args.dataset_names, models_to_merge)):
        # merged_model.config =
        merged_model.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.ckpt_path+f"/gpt2_{dataset_name}") # load the config
        task_vector_recon = {}
        for n in Vector_unified:
            task_vector_recon[n] = Vector_unified[n] * masks[n][idx] * rescales[idx]
        with torch.no_grad():
            merged_params = {}
            for param_name in task_vector_recon:
                merged_params[param_name] = pretrained_param_dict[param_name] + task_vector_recon[param_name]
        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_params:
                param_value.data.copy_(merged_params[param_name])
        merged_model.score = model_to_merge.score
        merged_model.to(args.device)
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)

        try:
            ds_val = ds['validation']
        except:
            ds_val = ds['validation_mismatched']
        with torch.no_grad():
            accuracy = Accuracy("multiclass", num_classes=num_labels[
                dataset_name])  # len(ds['validation'].unique('label')))#, num_classes=num_labels[dataset_name])
            loader = DataLoader(
                ds_val,
                collate_fn=default_data_collator,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
            )
            for batch in (
                    pbar := tqdm(
                        loader, desc="Evaluating", leave=False, dynamic_ncols=True
                    )
            ):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)

                outputs = merged_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
                pbar.set_postfix({"accuracy": acc.item()})

            acc = accuracy.compute().item()
            logger.info(f"acc on {dataset_name}: {acc}")

def mrpc_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples['sentence1'],#, 'sentence2'],
        examples["sentence2"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def mnli_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def cola_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def qnli_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["question"],
        examples["sentence"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def qqp_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["question1"],
        examples["question2"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs

class TokenizedGLUE:
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def load_dataset(
        self, name
    ):
        glue_dataset_loaders = {
            "mrpc": self.load_mrpc_dataset,
            "mnli": self.load_mnli_dataset,
            "cola": self.load_cola_dataset,
            "sst2": self.load_sst2_dataset,
            "qnli": self.load_qnli_dataset,
            "qqp": self.load_qqp_dataset,
            "rte": self.load_rte_dataset,
            # "wnli": load_wnli_dataset,
        }
        return glue_dataset_loaders[name]()


    def load_mrpc_dataset(self):
        dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/mrpc')
        dataset = dataset.map(
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=['sentence1', 'sentence2'],
        )
        return dataset


    def load_rte_dataset(self):
        dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/rte')
        # dataset = load_dataset("glue", "rte", cache_dir=cache_dir)
        dataset = dataset.map(
            # RTE has the same format as MRPC
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset


    def load_wnli_dataset(self):
        dataset = load_dataset("glue", "wnli", cache_dir=cache_dir)
        # dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/wnli')
        dataset = dataset.map(
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset


    def load_qqp_dataset(self):
        dataset = load_dataset("glue", "qqp", cache_dir=cache_dir)
        # dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/qqp')
        dataset = dataset.map(
            partial(qqp_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=['question1', 'question2'],
        )
        return dataset


    def load_mnli_dataset(self):
        dataset = load_dataset("glue", "mnli",  cache_dir=cache_dir)
        # dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/mnli')
        dataset = dataset.map(
            partial(mnli_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["premise", "hypothesis"],
        )
        return dataset


    def load_cola_dataset(self):
        dataset = load_dataset("glue", "cola", cache_dir=cache_dir)
        # dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/cola')
        dataset = dataset.map(
            partial(cola_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence"],
        )
        return dataset


    def load_sst2_dataset(self):
        dataset = load_dataset("glue", "sst2", cache_dir=cache_dir)
        # dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/sst2')
        print(dataset.column_names)
        dataset = dataset.map(
            partial(cola_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence"],
        )
        return dataset


    def load_qnli_dataset(self):
        # dataset = load_from_disk('/remote-home/yepeng2/cache/GLUE_DOWNLOAD/qnli')
        dataset = load_dataset("glue", "qnli", cache_dir=cache_dir)
        dataset = dataset.map(
            partial(qnli_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["question", "sentence"],
        )
        return dataset


num_labels = {
        'cola': 2,
        'sst2': 2,
        'mrpc': 2,
        'stsb': 5,
        'qqp': 2,
        'mnli': 3,
        'qnli': 2,
        'rte': 2
    }
dataset_names = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Interface for inference PLMs on glue")
    parser.add_argument("--language_model_name", type=str, default="gpt2", help="name of the language model", choices=["gpt2"])
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--merging_method_name", type=str, default="emr_merging",
                        help="name of the method to merge models",
                        choices=["emr_merging"])
    parser.add_argument("--gpu", type=int, default=2, help="number of gpu to use")
    parser.add_argument('--ckpt_path', type=str, default='/remote-home/yepeng2/Mario/ckpts/gpt2',help="ckpt path")

    try:
        args = parser.parse_args()
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    except:
        parser.print_help()
        sys.exit()
    args.dataset_names = dataset_names


    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2')


    tokenizer.model_max_length = 512
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)
    pretrained_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.ckpt_path+'/gpt2').to('cpu')


    models = []
    loaders = []
    for dataset_name in dataset_names:
        args.dataset_name = dataset_name
        load_model_path = args.ckpt_path+f"/gpt2_{dataset_name}"
        finetuned_model = GPT2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=load_model_path).to('cpu')
        models.append(finetuned_model)
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    save_merge_log_path = f"./save_merge_logs/{args.merging_method_name}/{args.language_model_name}"
    os.makedirs(save_merge_log_path, exist_ok=True)

    fh = logging.FileHandler(f"{save_merge_log_path}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")
    logger.info(f"configuration is {args}")
    performance = get_emr_merge_performance(args, models, loaders, tokenizer, logger)