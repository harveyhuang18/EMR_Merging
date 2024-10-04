import torch.optim.optimizer
from transformers.models.vit.modeling_vit import *

from torchvision.datasets import MNIST, ImageFolder, CIFAR10, CIFAR100, EuroSAT, Food101, GTSRB, SVHN, FashionMNIST, \
    OxfordIIITPet, FGVCAircraft, FER2013, STL10, EMNIST, DTD, SUN397, StanfordCars

from torch.utils.data.dataloader import DataLoader
import imageio

import time
from PIL import Image
from transformers import ViTForImageClassification
from tqdm import tqdm
from torch.utils.data import DataLoader
from ties_merging_utils import *
from args import parse_arguments

from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    Grayscale,
    ToTensor
)

import torch

trans_gray = Compose([
    Resize(size=(224, 224)),
    Grayscale(3),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
trans_rgb = Compose([
    Resize(size=(224, 224)),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path + '/' + filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def eval_single_dataset(image_encoder, dataset, clf, bz, device, ds_name='None'):
    image_encoder.classifier = clf
    model = image_encoder.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=bz)
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm(dataloader)):
            x = data[0].to(device)
            y = data[1].to(device)
            logits = model(x).logits
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
        top1 = correct / n
    metrics = {'top1': top1}
    print(f'Done evaluating on {ds_name}. Accuracy: {100 * top1:.2f}%')
    return metrics


def emr_merge(flat_task_vectors):
    sum_param = []
    n2p = []
    for m in range(len(flat_task_vectors)):
        n2p_temp = flat_task_vectors[m]
        n2p.append(n2p_temp)
    sum_param = torch.mean(flat_task_vectors, dim=0)
    vector_unified = {}
    scales = torch.zeros(len(flat_task_vectors))
    masks = []
    flag = (sum_param > 0) * 2 - 1
    param_max = torch.zeros_like(n2p[0])
    for m in range(len(flat_task_vectors)):
        param = flat_task_vectors[m]
        mask = (param * flag) > 0
        masks.append(mask)
        param_abs = torch.abs(mask * param)
        param_max = torch.where(param_abs > param_max, param_abs, param_max)
        scales[m] += torch.mean(torch.abs(param))
    vector_unified = param_max * flag
    new_scales = torch.zeros(len(flat_task_vectors))
    for m in range(len(flat_task_vectors)):
        p = vector_unified * masks[m]
        new_scales[m] += torch.mean(torch.abs(p))
    rescalers = scales / new_scales
    return vector_unified, masks, rescalers


if __name__ == '__main__':
    args = parse_arguments()
    args.model = 'ViT-B/16'
    args.home = '' # type your home path here
    args.save = args.home + '/checkpoints/' + args.model
    args.logs_path = args.home + '/logs/' + args.model
    args.batch_size = 32
    args.pretrained_checkpoint = args.home + '/model_weights/google_base_model'
    args.device = 'cuda:0'
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.logs_path, 'log_{}_emr_merging_30.txt'.format(str_time_))
    ckpts = [
        # Put your checkpoints Here
        "model_weights/mnist_finetuned",
        "model_weights/cifar10_finetuned_2",
        # ...
    ]
    exam_datasets = [
        # Put your dataset Here
        "MNIST(root='/remote-home/yepeng2/ds/mnist', train=False, transform=trans_gray)",
        "CIFAR10(root='/remote-home/yepeng2/ds/cifar10', train=False, transform=trans_rgb)",
        # ...
    ]
    pretrained_model = ViTForImageClassification.from_pretrained(args.pretrained_checkpoint).to('cpu')
    ft_checks = []
    ptm_check = pretrained_model.vit.state_dict()
    classifiers = []
    for ckpt in tqdm(ckpts, 'loading_model_weights'):
        ft_check = ViTForImageClassification.from_pretrained(args.home + '/' + ckpt).to('cpu').vit.state_dict()
        ft_checks.append(ft_check)
        classifiers.append(ViTForImageClassification.from_pretrained(args.home + '/' + ckpt).to('cpu').classifier)

    remove_keys = []
    print(f"Flattening out Checkpoints")
    flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
    tv_flat_checks = flat_ft - flat_ptm
    vector_uni, masks, rescalers = emr_merge(tv_flat_checks)
    print("Evaluating:")
    Total_ACC = 0.
    for i, ds_name in enumerate(exam_datasets):
        merged_state_dict = vector_to_state_dict(vector_uni * masks[i] * rescalers[i] + flat_ptm, ptm_check,
                                                 remove_keys=remove_keys)
        image_encoder = ViTForImageClassification.from_pretrained(args.pretrained_checkpoint)
        image_encoder.vit.load_state_dict(merged_state_dict, strict=False)

        dataset = eval(ds_name)
        metrics = eval_single_dataset(image_encoder, dataset, classifiers[i], args.batch_size, args.device, ds_name)
        Total_ACC += metrics['top1']
        log.info(str(metrics))
    log.info('Avg ACC:' + str(Total_ACC / len(exam_datasets)))

