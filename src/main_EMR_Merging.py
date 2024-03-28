import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import time
import sys
sys.path.append('/remote-home/yepeng2')

from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def apply_vector(vector, pretrained_checkpoint):#, scaling_coef=1.0):
    """Apply a task vector to a pretrained model."""
    with torch.no_grad():
        pretrained_model = torch.load(pretrained_checkpoint)
        new_state_dict = {}
        pretrained_state_dict = pretrained_model.state_dict()
        for key in pretrained_state_dict:
            if key not in vector:
                print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                continue
            new_state_dict[key] = pretrained_state_dict[key] + vector[key]
    pretrained_model.load_state_dict(new_state_dict, strict=False)
    return pretrained_model
def EMR_merge(task_vectors):
    sum_param = {}
    n2p = []
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].vector
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])
    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}
    Vector_unified = {}
    scales = torch.zeros(len(task_vectors))
    masks = {}
    for n in sum_param:
        masks[n] = []
        flag = (sum_param[n]>0)*2-1
        param_max = torch.zeros_like(n2p[0][n])
        for m in range(len(task_vectors)):
            param = task_vectors[m].vector[n]
            mask = (param * flag)>0
            masks[n].append(mask)
            param_abs = torch.abs(mask*param)
            param_max = torch.where(param_abs>param_max, param_abs, param_max)
            scales[m] += torch.mean(torch.abs(param))
            pass
        Vector_unified[n] =  param_max * flag

    new_scales = torch.zeros(len(task_vectors))
    for m in range(len(task_vectors)):
        for n in Vector_unified:
            p = Vector_unified[n] * masks[n][m]
            new_scales[m] += torch.mean(torch.abs(p))
    rescales = scales / new_scales

    return Vector_unified, masks, rescales



exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
model = 'ViT-L-14'
args = parse_arguments()
args.data_location = '/remote-home/yepeng2/ADAMerging/data'
args.model = model
args.save = '/remote-home/yepeng2/ADAMerging/checkpoints/' + model
args.logs_path = '/remote-home/yepeng2/ADAMerging/logs/' + model
pretrained_checkpoint = '/remote-home/yepeng2/ADAMerging/checkpoints/'+model+'/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))

task_vectors = [
    TaskVector(pretrained_checkpoint, '/remote-home/yepeng2/ADAMerging/checkpoints/'+model+'/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets
]

# task_vector_sum = sum(task_vectors)

# scaling_coef_ = 0.3
Vector_unified, masks, rescales = EMR_merge(task_vectors)

# log.info('*'*20 + 'scaling_coef:' + str(scaling_coef_) + '*'*20)

accs = []
for i in range(len(exam_datasets)):
    dataset = exam_datasets[i]
    task_vector_recon = {}
    args.batch_size=32
    for n in Vector_unified:
        task_vector_recon[n] =  Vector_unified[n] * masks[n][i] * rescales[i]
    image_encoder = apply_vector(task_vector_recon, pretrained_checkpoint)
    metrics = eval_single_dataset(image_encoder, dataset, args)
    log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
    accs.append(metrics.get('top1')*100)
log.info('Avg ACC:' + str(np.mean(accs)) + '%')
