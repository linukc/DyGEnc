## Installation

```bash
# Core dependencies
conda create -y -n dygenc python=3.10
conda activate dygenc
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install transformers==4.50.1 sentencepiece
pip install peft
# Other dependencies
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
# Optional dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install git+https://github.com/bfshi/scaling_on_scales.git
pip install --upgrade tokenizers==0.21.4
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
