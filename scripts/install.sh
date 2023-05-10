#!/bin/bash

pip install open_clip_torch pandas numpy tqdm toml openai python-dotenv
pip install -i https://download.pytorch.org/whl/cu117 torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0
pip install -f https://data.pyg.org/whl/torch-1.13.0+cu117.html pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric
