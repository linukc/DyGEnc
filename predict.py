import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore",
"\*`pad_token_id` to `eos_token_id`:128001 for open-end generation\*")
from tqdm import tqdm

import torch
import pandas as pd
from loguru import logger
from huggingface_hub import whoami
from hydra import initialize, compose
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore

from src.cfgs import set_config
from src.utils.seed import set_seed
from src.datasets import load_dataset
from src.utils.ckpt import reload_model
from src.model.graph_llm import DyGEnc
from src.utils.collate import collate_fn


def eval(args, cfg):
    set_seed(seed=cfg.seed)
    logger.info(cfg)

    val_dataset = load_dataset[args.dataset_name](split=cfg.val_split,
                                                  lm_model=cfg.val_lm_model,
                                                  seq_limit=cfg.val_seq_limit)
    val_loader = DataLoader(val_dataset,
                            num_workers=cfg.val_num_workers,
                            batch_size=cfg.val_batch_size,
                            drop_last=False, pin_memory=True, shuffle=False,
                            collate_fn=collate_fn)

    model = DyGEnc(cfg)
    model = reload_model(model, args)
    model.eval()

    save_path = f"eval/{args.dataset_name}"
    os.makedirs(save_path, exist_ok=True)

    progress_bar_test = tqdm(range(len(val_loader)))
    with open(os.path.join(save_path, f"{args.exp_name}.json"), "wt") as f:
        for _, batch in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)
    progress_bar_test.close()


if __name__ == "__main__":
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    assert whoami(), "Please, login in your HF account to access llama models!"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', help='one of star|agqa|custom', required=True, choices=["agqa", "star", "custom"])
    parser.add_argument('--exp_name', help='experiment_name', required=True)
    parser.add_argument('--ckpt_path', help='path to model state dict', required=False, default=None)
    args, remaining_argv = parser.parse_known_args()

    cs = ConfigStore.instance()
    cs.store(name="config", node=set_config[args.dataset_name])

    with initialize(config_path=None, version_base=None):
        cfg = compose(config_name="config", overrides=remaining_argv)

    eval(args, cfg)
