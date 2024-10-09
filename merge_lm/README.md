# Merging Language Models (LMs)

## Get Started

### Dependencies

Please follow [DARE](https://github.com/yule-BUAA/MergeLM) to install the dependencies.

Additionally, install scipy, sklearn, torchmetrics, evaluate.

### Checkpoints

**RoBERTa**: You can download the fine-tuned checkpoints from huggingface [here](https://huggingface.co/vanillaOVO/roberta_base_glue_ckpts/tree/main).

**GPT-2**: You can download the fine-tuned checkpoints from huggingface [here](https://huggingface.co/collections/tanganke/gpt-2-models-fine-tuned-on-tasks-from-glue-benchmark-664ab37d9e33e622679f541b).

Place the checkpoints as follows:

```
│cktps/
├──roberta/
│  ├── cola/
│  │  ├── roberta-base_lr1e-05
│  │  │  ├── config.json
│  │  │  ├──......
│  ├── sst2/
│  │  ├── roberta-base_lr1e-05
│  │  │  ├── config.json
│  │  │  ├──......
│  ├── ......
├──gpt2/
│  ├── gpt2/
│  │  ├── config.json
│  │  ├──......
│  ├── gpt2_cola/
│  │  ├── config.json
│  │  ├──......
│  ├── ......
```

#### 



You can modify the `cache_dir` in the `utils/load_config.py` file to specify your own path to save the datasets.

## Eval

#### Merge RoBERTa models

> python merge_roberta_glue.py

#### Merge GPT-2 models

> python merge_gpt_glue.py

## Results

Results for our EMR-Merging will be saved in ./save_merge_logs.

