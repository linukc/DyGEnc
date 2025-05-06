import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")
import argparse

import torch
import wandb
from tqdm import tqdm
from loguru import logger
from omegaconf import OmegaConf
from huggingface_hub import whoami
from hydra import initialize, compose
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from hydra.core.config_store import ConfigStore

from src.model import DyGEnc
from src.cfgs import set_config
from src.utils.seed import set_seed
from src.datasets import load_dataset
from src.utils.collate import collate_fn
from src.utils.ckpt import save_checkpoint
from src.utils.lr_schedule import adjust_learning_rate


def train_step(cfg, train_loader, model, optimizer, epoch, progress_bar):
    epoch_loss, accum_loss = 0., 0.

    for step, batch in enumerate(train_loader):

        optimizer.zero_grad()
        loss = model(batch)
        if not torch.isnan(loss):
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % cfg.adj_lr_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], cfg.lr, step / len(train_loader) + epoch, cfg)

            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

            if (step + 1) % cfg.adj_lr_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({'Lr': lr})
                wandb.log({'Accum Loss': accum_loss / cfg.adj_lr_steps})
                accum_loss = 0.
        else:
            logger.warning(f"{loss.item()} for step {step}" )

        progress_bar.update(1)

    logger.info(f"Epoch: {epoch}|{cfg.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
    wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

def main(args, cfg):
    set_seed(seed=cfg.seed)
    logger.info(cfg)

    # Step 1: Set up wandb
    wandb.init(project=f"DyGEnc_{args.dataset_name}",
               name=args.exp_name,
               config=OmegaConf.to_container(cfg, resolve=True))

    # Step 2: Load dataset
    train_dataset = load_dataset[args.dataset_name](split="train",
                                                    seq_limit=cfg.train_seq_limit,
                                                    lm_model=cfg.train_lm_model)
    val_dataset = load_dataset[args.dataset_name](split=cfg.val_split,
                                                  seq_limit=cfg.val_seq_limit,
                                                  lm_model=cfg.val_lm_model)

    train_loader = DataLoader(train_dataset, num_workers=cfg.train_num_workers, batch_size=cfg.train_batch_size,
                              drop_last=True, pin_memory=True, shuffle=True, collate_fn=collate_fn)
    
    val_loader = DataLoader(val_dataset, num_workers=cfg.val_num_workers, batch_size=cfg.val_batch_size,
                            drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # Step 3: Build Model
    model = DyGEnc(cfg)

    # Step 4 Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': cfg.lr, 'weight_decay': cfg.wd}, ],
        betas=(0.9, 0.95)
    )
    trainable_params, all_param = model.print_trainable_params()
    logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    # Step 5. Training
    num_training_steps = cfg.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        train_step(cfg, train_loader, model, optimizer, epoch, progress_bar)
        save_checkpoint(model, optimizer, epoch, args, cfg, is_best=False)

        val_loss = 0.
        model.eval()
        torch.cuda.empty_cache() 
        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                loss = model(batch)
                if not torch.isnan(loss):
                    val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            logger.info(f"Epoch: {epoch}|{cfg.num_epochs}: Val Loss: {val_loss}")
            wandb.log({'Val Loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, args, cfg, is_best=True)
            best_epoch = epoch

        logger.info(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

        if epoch - best_epoch >= cfg.patience:
            logger.warning(f'Early stop at epoch {epoch}')
            break
        torch.cuda.empty_cache()
    progress_bar.close()


if __name__ == "__main__":
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    assert whoami(), "Please, login in your HF account to access llama models!"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', help='one of star|agqa', required=True, choices=["agqa", "star"])
    parser.add_argument('--exp_name', help='experiment_name', required=True)
    args, remaining_argv = parser.parse_known_args()

    cs = ConfigStore.instance()
    cs.store(name="config", node=set_config[args.dataset_name])

    with initialize(config_path=None, version_base=None):
        cfg = compose(config_name="config", overrides=remaining_argv)

    main(args, cfg)
