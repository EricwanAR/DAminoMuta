#!/bin/bash
python main.py --task cls --loss ce --q-encoder lstm --channels 256 --fusion mlp
python main.py --task cls --loss ce --q-encoder lstm --channels 256 --fusion att
python main.py --task cls --loss ce --q-encoder lstm --channels 256 --fusion diff
python main.py --task cls --loss ce --q-encoder mamba --channels 256 --fusion mlp
python main.py --task cls --loss ce --q-encoder mamba --channels 256 --fusion att
python main.py --task cls --loss ce --q-encoder mamba --channels 256 --fusion diff
python main.py --task cls --loss ce --q-encoder mha --channels 256 --fusion mlp
python main.py --task cls --loss ce --q-encoder mha --channels 256 --fusion att
python main.py --task cls --loss ce --q-encoder mha --channels 256 --fusion diff
python main.py --task cls --loss ce --q-encoder gru --channels 256 --fusion mlp
python main.py --task cls --loss ce --q-encoder gru --channels 256 --fusion att
python main.py --task cls --loss ce --q-encoder gru --channels 256 --fusion diff
python main.py --task cls --loss ce --q-encoder rn18 --channels 16 --fusion mlp
python main.py --task cls --loss ce --q-encoder rn18 --channels 16 --fusion att
python main.py --task cls --loss ce --q-encoder rn18 --channels 16 --fusion diff
python main.py --task cls --loss ce --q-encoder rn18 --channels 16 --pcs --fusion diff
python main.py --task cls --loss ce --q-encoder rn18 --channels 16 --pcs --fusion diff --side-enc mamba
python main.py --task cls --loss ce --q-encoder rn18 --channels 16 --pcs --fusion diff --side-enc mamba --non-siamese
python uda.py --task cls --loss ce --q-encoder rn18 --channels 16 --pcs --fusion diff --side-enc mamba --case r2
