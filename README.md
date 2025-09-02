# AI-based D-amino acid substitution for optimizing antimicrobial peptides to treat multidrug-resistant bacterial infection
This repository contains the code for the paper "AI-based D-amino acid substitution for optimizing antimicrobial peptides to treat multidrug-resistant bacterial infection"

## Requirements
```
mamba_ssm==2.2.4
numpy==1.26.3
pandas==2.1.4
rdkit==2024.3.5
scikit_learn==1.4.1.post1
scipy==1.13.0
torch==2.2.0
torchmetrics==1.3.1
torchvision==0.17.0
```
You can install them with `pip install -r requirements.txt`


## Training

`main.py` can train model with Classification or Regression tasks. 

example: 
```
python main.py \
    --q-encoder rn18 \ # Encoder, can be rn18, lstm, gru, mamba, mha
    --channels 16 \ # Encoder channels. Does not work for ResNet18
    --side-enc mamba \ # Side sequence Encoder, only lstm and mamba implemented, only use with rn18 encoder
    --fusion diff \ # Fusion method, can be att, mlp or diff
    --task cls \ # Task, can be cls or reg
    --loss ce \ # Loss, can be ce or mse, some other losses can be found in code
    --batch-size 32 \ # Batch size
    --epochs 50 \ # Epochs
    --gpu 0 \ # GPU index to use, -1 for cpu
# ===CNN only options=== \
    --pcs \ # Enable protease cleavage site dyeing for input pictures
    --resize 768 \ # Resize input pictures, can be 1 or 2 numbers like 768 or 768 512
```
Corresponding model weight checkpoints will be saved in the subdirectory of `run-cls` or `run-reg`, e.g. `/run-cls/rn18-diff-16-mamba-pcs-768-ce-32-0.001-50/`.

For more arguments, please refer to the code of `main.py`.

## UDA
You can use `uda.py` to train a UDA model. Arguments are the same as `main.py`, except that you need to specify the case domain with `--case r2` or `--case {YOUR_PEPTIDE_SEQUENCE}`.

Model weights will be saved in the subdirectory of the normal training directory, e.g. `/run-cls/rn18-diff-16-mamba-pcs-768-ce-32-0.001-50/uda_r2/`

## Inference
You can simple replace `main.py` with `infer.py` in your training command to do inference on test set.

For case study scanning, please use `infer_case.py` with an additional argument `--case r2` or `--case {YOUR_PEPTIDE_SEQUENCE}`. If you wish to inference with UDA weights, please use `--uda` argument.

Inference results will be saved in the weights directory in `csv` format, e.g. `/run-cls/rn18-diff-16-mamba-pcs-768-ce-32-0.001-50/preds_test.csv`.