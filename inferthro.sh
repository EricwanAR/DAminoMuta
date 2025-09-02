# !/bin/bash
python infer.py --task cls --loss ce --q-encoder lstm --channels 256 --fusion diff
python infer.py --task cls --loss ce --q-encoder mamba --channels 256 --fusion diff
python infer.py --task cls --loss ce --q-encoder mha --channels 256 --fusion diff
python infer.py --task cls --loss ce --q-encoder gru --channels 256 --fusion diff
python infer.py --task cls --loss ce --q-encoder rn18 --channels 16 --fusion diff --pcs --side-enc mamba
python infer.py --task cls --loss ce --q-encoder rn18 --channels 16 --fusion diff --pcs --side-enc mamba --uda r2
python infer.py --task reg --loss mse --q-encoder lstm --channels 256 --fusion diff
python infer.py --task reg --loss mse --q-encoder mamba --channels 256 --fusion diff
python infer.py --task reg --loss mse --q-encoder mha --channels 256 --fusion diff
python infer.py --task reg --loss mse --q-encoder gru --channels 256 --fusion diff
python infer.py --task reg --loss mse --q-encoder rn18 --channels 16 --fusion diff --pcs --side-enc mamba
python infer.py --task reg --loss mse --q-encoder rn18 --channels 16 --fusion diff --pcs --side-enc mamba --uda r2
