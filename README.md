# DyGEnc

A novel method for encoding a sequence of textual scene graphs for QA.

## Installation

```bash
# Core dependencies
conda create -y -n dygenc python=3.10
conda activate dygenc
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install transformers sentencepiece
pip install peft
# Other dependencies
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
# Optional dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install git+https://github.com/bfshi/scaling_on_scales.git
```

## Running from scratch

### Downloading

Follow instructions in the `datasets/` folder.

At the end call `source setup.bash` to set path to data or add argument for a custom path:

```bash
#!/bin/bash

PWD=${1:-`pwd`}

AGQA_ROOT=$PWD/agqa
STAR_ROOT=$PWD/star
```

### Data Preprocessing

```bash
# star
python -m src.datasets.preprocess.star
# agga
python -m src.datasets.preprocess.agga
```

### Training

Check `src/cfgs/<dataset_name>.py` before start:

```bash
python3 train.py --dataset_name=<(agqa|star)> --exp_name=<name> <optional args>
```

### Prediction

Check `src/cfgs/<dataset_name>.py` before start:

```bash
python3 predict.py --dataset_name=<(agqa|star)> --exp_name=<name> <optional args>
```

After this step, you can find results in the corresponding folder `eval/<dataset_name>/<exp_name>`.

### Evaluating

```bash
python3 eval.py --dataset_name=<(agqa|star)> --exp_name=<name>
```

### To reproduce results

```bash
./scripts/ablation_temp_encoding.sh # Table 1
./scripts/ablation_num_q_latent.sh # Table 2
./scripts/components.sh # Table 3
./scripts/star.sh # Table 4
./scripts/agqa.sh # Table 5
```

## Application

You can use model to answer questions about sequence of dynamic events based on textual scene graphs.
Before you start, check custom dataset structure in `datasets/drobot` - you should manually create json with questions at least.

Generate VSG sequence (3 types of sgg available: *nvila_factual_cfg*, *nvila_gpt_cfg*, *gpt*):

```bash
python3 -m src.vsg.generate --sgg_method=<one of 3> --image_folder=<your_path> --output_folder=<your_path>
```

Preprocess (embed and retrieve based on questions):
```bash
CUSTOM_ROOT=<your path> python3 -m src.datasets.preprocess.custom 
```

Call predict:
```bash
CUSTOM_ROOT=<your path> python3 predict.py --dataset_name=custom --exp_name=<output_folder> --ckpt_path=<path_to_weights>
```

You can find results in corresponding json file in `eval` folder.
