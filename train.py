import math
import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")
import argparse

import torch
import trackio as wandb
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


def train_step(cfg, args, train_loader, model, optimizer, epoch, progress_bar):
    epoch_loss, accum_loss = 0., 0.
    save_step = math.ceil(len(train_loader.dataset) / train_loader.batch_size) // 10
    for step, batch in enumerate(train_loader):

        optimizer.zero_grad()
        prepared_inputs = model.prepare_train_input(batch)
        loss = model(*prepared_inputs)
        if not torch.isnan(loss):
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % cfg.adj_lr_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], cfg.lr, step / len(train_loader) + epoch, cfg)

            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

            if (step + 1) % cfg.adj_lr_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                try:
                    wandb.log({'Lr': lr})
                    wandb.log({'Accum Loss': accum_loss / cfg.adj_lr_steps})
                except:
                    logger.warning("trackio failed")
                accum_loss = 0.
        else:
            logger.warning(f"{loss.item()} for step {step}" )
        
        if step != 0 and step % save_step == 0:
            logger.info(f"Saving ckpt each {save_step} for current step {step}/{len(train_loader.dataset)}")
            save_checkpoint(model, optimizer, step, args, cfg, is_best=False)

        progress_bar.update(1)

    logger.info(f"Epoch: {epoch}|{cfg.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
    try:
        wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})
    except:
        logger.warning("trackio failed")

def train_step_with_accum(cfg, args, train_loader, model, optimizer, epoch, progress_bar):
    ### Monkey-patch to swap default CE mean reduction to sum reduction
    from transformers.loss.loss_utils import fixed_cross_entropy
    original_fixed_cross_entropy = fixed_cross_entropy

    def patched_fixed_cross_entropy(*args, **kwargs):
        # Force reduction to "sum"
        kwargs['num_items_in_batch'] = 1
        # see source code https://github.com/huggingface/transformers/blob/
        # 565dd0bad74a46d85c41e2d870f803d9e7a1a94e/src/transformers/loss/loss_utils.py#L28

        # Call original function with modified args
        loss = original_fixed_cross_entropy(*args, **kwargs)
        return loss
    fixed_cross_entropy = patched_fixed_cross_entropy

    logger.warning("For grad. accum. training cfg.adj_lr_steps is set to cfg.accumulation_steps manually.")
    epoch_loss, accum_loss = 0., 0.

    num_samples_in_epoch = len(train_loader)
    remainder = num_samples_in_epoch % cfg.accumulation_steps
    remainder = remainder if remainder != 0 else cfg.accumulation_steps
    total_gradient_updates = math.ceil(num_samples_in_epoch / cfg.accumulation_steps)
    train_loader = iter(train_loader)
    save_step = total_gradient_updates // 10

    total_batched_samples = 0
    for update_step in range(total_gradient_updates):
        optimizer.zero_grad()

        # In order to correctly estimate the total number of non-padded tokens on which we'll compute the cross-entropy loss
        # we need to pre-load the full local batch - i.e the next batch_size * accumulation_steps samples
        batch_samples = []
        num_batches_in_step = cfg.accumulation_steps if update_step != (total_gradient_updates - 1) else remainder
        for _ in range(num_batches_in_step):
            batch_samples.append(model.prepare_train_input(next(train_loader)))
        # get local num items in batch
        items_in_batch = [(batch[2].ne(-100)).sum().item() for batch in batch_samples]
        num_items_in_batch = sum(items_in_batch)

        for i, batch in enumerate(batch_samples):
            loss = model(*batch)
            if torch.isnan(loss):
                num_items_in_batch -= items_in_batch[i]
                logger.warning(f"{loss.item()} for accum update step {update_step} / step {i}" )
                continue
            else:
                if num_items_in_batch == 0 :
                    logger.warning("invalid batch")
                    continue
                loss /= num_items_in_batch
                epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()
                loss.backward()
                progress_bar.update(1)
                total_batched_samples += 1

        clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
        adjust_learning_rate(optimizer.param_groups[0], cfg.lr, total_batched_samples / num_samples_in_epoch + epoch, cfg)
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        try:
            wandb.log({'Lr': lr})
            wandb.log({'Accum Loss': accum_loss})
        except:
            logger.warning("trackio failed")
        accum_loss = 0.0

        if update_step != 0 and update_step % save_step == 0:
            logger.info(f"Saving ckpt each {save_step} for step {update_step}/{total_gradient_updates}")
            save_checkpoint(model, optimizer, update_step, args, cfg, is_best=False)

    logger.info(f"Epoch: {epoch}|{cfg.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / num_samples_in_epoch}")
    try:
        wandb.log({'Train Loss (Epoch Mean)': epoch_loss / num_samples_in_epoch})
    except:
        logger.warning("trackio failed")

def main(args, cfg):
    set_seed(seed=cfg.seed)
    logger.info(cfg)
  
    # Step 1: Set up wandb
    wandb.init(project=f"{args.dataset_name}",
               name=args.exp_name,
               config=OmegaConf.to_container(cfg, resolve=True),
               space_id=f"{args.dataset_name}_gradio",
               dataset_id=f"{args.exp_name}_{args.dataset_name}")

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

    try:
        for epoch in range(cfg.num_epochs):
            model.train()
            if cfg.accumulation_steps != 1:
                assert cfg.train_batch_size == 1, "haven't tested the other ones"
                train_step_with_accum(cfg, args, train_loader, model, optimizer, epoch, progress_bar)
            else:
                train_step(cfg, args, train_loader, model, optimizer, epoch, progress_bar)
            save_checkpoint(model, optimizer, epoch, args, cfg, is_best=False)
    except Exception as e:
        print(e)
        save_checkpoint(model, optimizer, epoch, args, cfg, is_best=False)
    
    val_loss = 0.
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            prepared_inputs = model.prepare_train_input(batch)
            loss = model(*prepared_inputs)
            if not torch.isnan(loss):
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch: {epoch}|{cfg.num_epochs}: Val Loss: {val_loss}")
        try:
            wandb.log({'Val Loss': val_loss})
        except:
            logger.warning("trackio failed")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, args, cfg, is_best=True)
        best_epoch = epoch

    logger.info(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

    if epoch - best_epoch >= cfg.patience:
        logger.warning(f'Early stop at epoch {epoch}')

    torch.cuda.empty_cache()
    progress_bar.close()
    wandb.finish()

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
