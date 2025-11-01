import math


def adjust_learning_rate(param_group, LR, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = LR / 2
    if epoch < args.warmup_epochs:
        lr = LR * epoch / args.warmup_epochs
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs + 1e-10)))
    param_group["lr"] = lr
    return lr
