import os
import torch
from loguru import logger


def print_trainable_params(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def save_checkpoint(model, optimizer, cur_epoch, args, cfg, is_best=False):
    """
    Save the checkpoint at the current epoch.
    """
    os.makedirs(f'weights/{args.dataset_name}/{args.exp_name}', exist_ok=True)

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "name": args.exp_name,
        "epoch": cur_epoch,
    }
    path = f'weights/{args.dataset_name}/{args.exp_name}/{"best" if is_best else cur_epoch}.pth'
    logger.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, path))
    torch.save(save_obj, path)

def reload_model(model, args):
    """
    Load checkpoint for evaluation.
    """
    if not args.ckpt_path:
        checkpoint_path = f'weights/{args.dataset_name}/{args.exp_name}/best.pth'
    else:
        checkpoint_path = args.ckpt_path
    logger.info("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model
