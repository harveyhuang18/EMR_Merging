import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
import numpy as np
from tqdm import tqdm
import utils
from modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
from run_beit3_finetuning import *
def filt_param_to_merge(pt_weight, ft_weights):
    names = []
    for n in pt_weight['model']:
        flag = True
        for ft_weight in ft_weights:
            if n not in ft_weight['model'] or  pt_weight['model'][n].shape != ft_weight['model'][n].shape:
                flag = False
                break
        if flag:
            names.append(n)
    return names

def TA_merge(task_vectors, lamda=1):
    vector = {}
    for n in task_vectors[0]:
        for i in range(len(task_vectors)):
            if n not in vector:
                vector[n] = task_vectors[i][n] * lamda
            else:
                vector[n] += task_vectors[i][n] * lamda

    return vector


def EMR_merge(task_vectors):
    sum_param = {}
    n2p = []
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m]#.vector
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
            param = task_vectors[m][n]
            mask = (param * flag)>0
            masks[n].append(mask)
            param_abs = torch.abs(mask*param)
            param_max  += param_abs#= torch.where(param_abs>param_max, param_abs, param_max)
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

def Task_Vectors(pt_weight, ft_weights):#, param_names):
    param_names_to_merge = filt_param_to_merge(pt_weight, ft_weights)
    vectors = []
    for ft_weight in ft_weights:
        vector = {}
        for n in param_names_to_merge:

            vector[n] = ft_weight['model'][n] - pt_weight['model'][n]
        vectors.append(vector)
    return vectors




def val(ckpt, ds_init, args):
    utils.init_distributed_mode(args)
    if ds_init is not None:
        utils.create_ds_config(args)
    if args.task_cache_path is None:
        args.task_cache_path = args.output_dir
    print(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    if utils.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None
    data_loader_train, data_loader_val = create_downstream_dataset(args)

    if not args.model.endswith(args.task):
        if args.task in ("flickr30k", "coco_retrieval"):
            model_config = "%s_retrieval" % args.model
        elif args.task in ("coco_captioning", "nocaps"):
            model_config = "%s_captioning" % args.model
        elif args.task in ("imagenet"):
            model_config = "%s_imageclassification" % args.model
        else:
            model_config = "%s_%s" % (args.model, args.task)
    else:
        model_config = args.model
    print("model_config = %s" % model_config)
    model = create_model(
        model_config,
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations,
    )
    utils.load_model_from_ckpt(ckpt, model, args.model_key, args.model_prefix)
    model.to(device)

    task_handler = get_handler(args)

    data_loader_test = create_downstream_dataset(args, is_eval=True)
    if args.task in ["nlvr2", "flickr30k", "coco_retrieval", "imagenet"]:
        ext_test_stats, task_key = evaluate(data_loader_test, model, device, task_handler)
        print(
            f"Accuracy of the network on the {len(data_loader_test.dataset)} test images: {ext_test_stats[task_key]:.3f}%")
        exit(0)
    elif args.task == "vqav2":
        result, _ = evaluate(data_loader_test, model, device, task_handler)
        utils.dump_predictions(args, result, "vqav2_test")
        exit(0)
    elif args.task in ["coco_captioning", "nocaps"]:
        predictions, _ = evaluate(data_loader_test, model, device, task_handler)
        prediction_file = utils.dump_predictions(args, predictions, "{}_test".format(args.task))
        if utils.is_main_process() and args.task == "coco_captioning":
            captioning_result = utils.coco_caption_eval(args.output_dir, prediction_file,
                                                        "{}_test".format(args.task))
            result_file = os.path.join(args.output_dir, f"{args.task}_result.json")
            print(json.dumps(captioning_result))
            utils.write_result_to_jsonl(captioning_result, result_file)
        exit(0)


def main():
    ckpt_path = './model_weights/' # enter your ckpt path here
    ft_ckpt_name = ['beit3_base_patch16_480_vqa.pth', 'beit3_base_patch16_480_coco_captioning.pth', 'beit3_base_patch16_384_coco_retrieval.pth', 'beit3_base_patch16_224_nlvr2.pth', 'beit3_base_patch16_224_in1k.pth']
    tasks = ['vqav2', 'coco_captioning', 'coco_retrieval', 'nlvr2', 'imagenet']
    ft_model_weights = []
    for ckpt in ft_ckpt_name:
        ft_model_weights.append(torch.load(ckpt_path+ckpt))
    pt_model_name = "beit3_base_patch16_224.pth"
    pt_model_weight = torch.load(ckpt_path + pt_model_name)
    # base_model_weight
    vectors = Task_Vectors(pt_model_weight, ft_model_weights)
    Vector_unified, masks, rescales = EMR_merge(vectors)
    param_recon = {}
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    try:
        task_num = tasks.index(opts.task)
    except:
        raise ValueError('Task not applicable!!!')
    for n in Vector_unified:
        param_recon[n] = Vector_unified[n] * masks[n][task_num] * rescales[task_num] + pt_model_weight['model'][n]
    ft_model = ft_model_weights[task_num]
    for n in param_recon:
        ft_model['model'][n] = param_recon[n]
    val(ft_model, ds_init, opts)


if __name__ == '__main__':
    main()

