#!/bin/bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install -c conda-forge rdkit openbabel graph-tool -y
pip install torch_geometric

# python -c "import torch; print(torch.__version__)"
# python -c "import torch; print(torch.version.cuda)"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

pip install tensorboard
pip install wandb