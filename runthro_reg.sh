#!/bin/bash
# python main.py --q-encoder lstm --channels 256 --fusion mlp
# python main.py --q-encoder lstm --channels 256 --fusion att
# python main.py --q-encoder lstm --channels 256 --fusion diff
# python main.py --q-encoder mamba --channels 256 --fusion mlp
# python main.py --q-encoder mamba --channels 256 --fusion att
# python main.py --q-encoder mamba --channels 256 --fusion diff
# python main.py --q-encoder mha --channels 256 --fusion mlp
# python main.py --q-encoder mha --channels 256 --fusion att
# python main.py --q-encoder mha --channels 256 --fusion diff
# python main.py --q-encoder gru --channels 256 --fusion mlp
# python main.py --q-encoder gru --channels 256 --fusion att
# python main.py --q-encoder gru --channels 256 --fusion diff
# python main.py --q-encoder cnn --channels 16 --fusion mlp
# python main.py --q-encoder cnn --channels 16 --fusion att
# python main.py --q-encoder cnn --channels 16 --fusion diff
# python main.py --q-encoder cnn --channels 16 --pcs --fusion mlp
# python main.py --q-encoder cnn --channels 16 --pcs --fusion att
# python main.py --q-encoder cnn --channels 16 --pcs --fusion diff
python main.py --q-encoder rn18 --channels 16 --fusion mlp
python main.py --q-encoder rn18 --channels 16 --fusion att
python main.py --q-encoder rn18 --channels 16 --fusion diff
python main.py --q-encoder rn18 --channels 16 --pcs --fusion mlp
python main.py --q-encoder rn18 --channels 16 --pcs --fusion att
python main.py --q-encoder rn18 --channels 16 --pcs --fusion diff
# python main.py --q-encoder cnn --channels 16 --side-enc lstm --pcs --fusion att