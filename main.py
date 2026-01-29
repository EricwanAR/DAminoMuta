import argparse
import json
import logging
import os
import time

from dataset import PeptidePairDataset, PeptidePairPicDataset
from network import DMutaPeptide, DMutaPeptideCNN
from sklearn.model_selection import KFold
from train import train, train_cls
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from loss import MLCE, SuperLoss, LogCoshLoss, BMCLoss
from utils import set_seed


parser = argparse.ArgumentParser()
# model setting
parser.add_argument('--q-encoder', dest='q_encoder', type=str, default='lstm',
                    help='lstm mamba mla')
parser.add_argument("--side-enc", dest='side_enc', type=str, default=None,
                    help="use side features")
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--fusion', type=str, default='att',
                    help='mlp att diff')
parser.add_argument('--non-siamese', dest='non_siamese', action='store_true', default=False,
                    help="use non-siamese architecture")

# task & dataset setting
parser.add_argument('--task', type=str, default='reg',
                    help='reg or cls')
parser.add_argument('--one-way', action='store_true', dest='one_way', default=False,
                    help='use one-way constructed dataset')
parser.add_argument('--max-length', dest='max_length', type=int, default=30,
                    help='Max length for sequence filtering')
parser.add_argument('--split', type=int, default=5,
                    help="Split k fold in cross validation (default: 5)")
parser.add_argument('--seed', type=int, default=42,
                    help="Seed (default: 1)")
parser.add_argument('--pcs', action='store_true', default=False,
                    help='Consider protease cleavage site')
parser.add_argument('--resize', type=int, default=[768], nargs='+',
                    help='resize the image')

# training setting
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU index to use, -1 for CPU (default: 0)')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=32,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=0.0005,
                    help='weight decay (default: 0.0005)')
parser.add_argument('--metric-avg', type=str, dest='metric_avg', default='macro',
                    help='metric average type')
parser.add_argument('--glob-feat', action='store_true', default=False,
                    help='use global features')
parser.add_argument('--loss', type=str, default='mse',
                    help='loss function')
parser.add_argument('--dir', action='store_true', default=False,
                    help='use DIR')

args = parser.parse_args()


def main():
    set_seed(args.seed)
    if args.task == 'reg':
        args.classes = 1
        trainer = train
        if args.loss == "mse" or args.loss in ['ce']:
            args.loss = 'mse'
            criterion = nn.MSELoss()
        elif args.loss == "smoothl1":
            criterion = nn.SmoothL1Loss()
        elif args.loss == "super":
            criterion = SuperLoss()
        elif args.loss in ["bmc", "bmc_ln"]:
            criterion = BMCLoss()
        else:
            raise NotImplementedError("unimplemented regression task loss function")
    elif args.task == 'cls':
        trainer = train_cls
        args.classes = 2
        if args.loss == 'ce' or args.loss in ['mse', 'smoothl1', 'super']:
            args.loss = 'ce'
            criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("unimplemented classification task loss function")
    else:
        raise NotImplementedError("unimplemented task")
    
    if args.q_encoder in ['cnn', 'rn18']:
        weight_dir = f'./run-{args.task}/{"non-siamese-" if args.non_siamese else ""}{args.q_encoder}-{args.fusion}-{args.channels}{f"-{args.side_enc}" if args.side_enc else ""}{"-mixpcs" if args.mix_pcs else ""}{"-pcs" if args.pcs==True else ""}{"-" + "x".join(str(n) for n in args.resize) if args.resize else ""}{"-oneway" if args.one_way else ""}-{args.loss + "-dir" if args.dir else args.loss}-{str(args.batch_size)}-{str(args.lr)}-{str(args.epochs)}'
    else:
        weight_dir = f'./run-{args.task}/{"non-siamese-" if args.non_siamese else ""}{args.q_encoder}-{args.fusion}-{args.channels}{"-oneway" if args.one_way else ""}-{args.loss + "-dir" if args.dir else args.loss}-{str(args.batch_size)}-{str(args.lr)}-{str(args.epochs)}'

    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    
    logging.basicConfig(handlers=[
        logging.FileHandler(filename=os.path.join(weight_dir, "training.log"), encoding='utf-8', mode='w+'),
        logging.StreamHandler()],
        format="%(asctime)s: %(message)s", datefmt="%F %T", level=logging.INFO)

    logging.info(f'saving_dir: {weight_dir}')
    
    with open(os.path.join(weight_dir, "config.json"), "w") as f:
        f.write(json.dumps(vars(args)))

    device = torch.device("cpu" if args.gpu == -1 or not torch.cuda.is_available() else f"cuda:{args.gpu}")

    if args.q_encoder in ['cnn', 'rn18']:
        logging.info('Loading Training Dataset')
        all_set = PeptidePairPicDataset(mode='train', pad_length=args.max_length, task=args.task, one_way=args.one_way, gf=args.glob_feat, side_enc=args.side_enc, pcs=args.pcs, resize=args.resize)
        logging.info('Loading Test Dataset')
        test_set = PeptidePairPicDataset(mode='test', pad_length=args.max_length, task=args.task, gf=args.glob_feat, side_enc=args.side_enc, pcs=args.pcs, resize=args.resize)
    else:
        logging.info('Loading Train Dataset')
        all_set = PeptidePairDataset(mode='train', pad_length=args.max_length, task=args.task, one_way=args.one_way, gf=args.glob_feat)
        logging.info('Loading Test Dataset')
        test_set = PeptidePairDataset(mode='test', pad_length=args.max_length, task=args.task, gf=args.glob_feat)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    best_perform_list = [[] for i in range(5)]
    test_perform_list = [[] for i in range(5)]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_set)):
        train_set= Subset(all_set, train_idx)
        valid_set = Subset(all_set, val_idx)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        if args.q_encoder in ['cnn', 'rn18']:
            model = DMutaPeptideCNN(q_encoder=args.q_encoder, classes=args.classes, channels=args.channels, dir=args.dir, gf=args.glob_feat, side_enc=args.side_enc, fusion=args.fusion, non_siamese=args.non_siamese)
        else:
            model = DMutaPeptide(q_encoder=args.q_encoder, classes=args.classes, channels=args.channels, dir=args.dir, gf=args.glob_feat, fusion=args.fusion, non_siamese=args.non_siamese)

        model.to(device)
        # model.compile()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        if args.loss == 'bmc_ln':
            optimizer.add_param_group({'params': criterion.noise_sigma, 'lr': args.lr, 'name': 'noise_sigma'})
        weights_path = f"{weight_dir}/model_{fold}.pth"

        logging.info(f'Running Cross Validation {fold}')
        logging.info(f'Fold {fold}  Train set:{len(train_set)}, Valid set:{len(valid_set)}, Test set: {len(test_set)}')
        best_metric = -float('inf')
        best_test = -float('inf')
        start_time = time.time()
        if args.task == 'reg':
            for epoch in range(1, args.epochs + 1):
                train_loss, mae, rse, pcc, kcc = trainer(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer)
                logging.info(f'Epoch: {epoch:03d} Train Loss: {train_loss:.3f}, mae: {mae:.3f}, rse: {rse:.3f}, pcc: {pcc:.3f}, kcc: {kcc:.3f}')
                scheduler.step()
                avg_metric = (pcc + kcc) - (mae + rse)
                if avg_metric > best_metric:
                    logging.info(f'Epoch: {epoch:03d} New best VALIDATION metrics')
                    torch.save(model.state_dict(), weights_path)
                    best_metric = avg_metric
                    best_perform_list[fold] = np.asarray([mae, rse, pcc, kcc])
                
                _, test_mae, test_rse, test_pcc, test_kcc = trainer(args, epoch, model, None, test_loader, device, None, None)
                logging.info(f'Epoch: {epoch:03d} Test results, ap: mae: {test_mae:.3f}, rse: {test_rse:.3f}, pcc: {test_pcc:.3f}, kcc: {test_kcc:.3f}')
                test_metric = (test_pcc + test_kcc) - (test_mae + test_rse)
                if test_metric > best_test and epoch > 10:
                    logging.info(f'Epoch: {epoch:03d} New best TEST metrics')
                    best_test = test_metric
                    test_perform_list[fold] = np.asarray([test_mae, test_rse, test_pcc, test_kcc])
                    torch.save(model.state_dict(), weights_path.replace('.pth', '_test.pth'))
        
        elif args.task == 'cls':
            for epoch in range(1, args.epochs + 1):
                train_loss, ap, auc, f1, acc = trainer(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer)
                logging.info(f'Epoch: {epoch:03d} Train Loss: {train_loss:.3f}, ap: {ap:.3f}, auc: {auc:.3f}, f1: {f1:.3f}, acc: {acc:.3f}')
                scheduler.step()
                avg_metric = ap + auc #+ f1 + acc
                if avg_metric > best_metric:
                    logging.info(f'Epoch: {epoch:03d} New best VALIDATION metrics')
                    torch.save(model.state_dict(), weights_path)
                    best_metric = avg_metric
                    best_perform_list[fold] = np.asarray([ap, auc, f1, acc])
                
                _, test_ap, test_auc, test_f1, test_acc = trainer(args, epoch, model, None, test_loader, device, None, None)
                logging.info(f'Epoch: {epoch:03d} Test results, ap: {test_ap:.3f}, auc: {test_auc:.3f}, f1: {test_f1:.3f}, acc: {test_acc:.3f}')
                test_metric = test_ap + test_auc #+ test_f1 + test_acc
                if test_metric > best_test and epoch > 10:
                    logging.info(f'Epoch: {epoch:03d} New best TEST metrics')
                    best_test = test_metric
                    test_perform_list[fold] = np.asarray([test_ap, test_auc, test_f1, test_acc])
                    torch.save(model.state_dict(), weights_path.replace('.pth', '_test.pth'))
        
        torch.save(model.state_dict(), weights_path.replace('.pth', '_last.pth'))
        logging.info(f'used time {(time.time()-start_time)/3600:.2f}h')

    logging.info(f'Cross Validation Finished!')
    best_perform_list = np.asarray(best_perform_list)
    test_perform_list = np.asarray(test_perform_list)
    logging.info('Best validation perform list\n%s', best_perform_list)
    logging.info('mean: %s', np.round(np.mean(best_perform_list, 0), 3))
    logging.info('std: %s', np.round(np.std(best_perform_list, 0), 3))
    logging.info('Best test perform list\n%s', test_perform_list)
    logging.info('mean: %s', np.round(np.mean(test_perform_list, 0), 3))
    logging.info('std: %s', np.round(np.std(test_perform_list, 0), 3))
    perform = open(weight_dir+'/result.txt', 'w')
    perform.write('Valid\n')
    perform.write(','.join([str(i) for i in np.mean(best_perform_list, 0)])+'\n')
    perform.write(','.join([str(i) for i in np.std(best_perform_list, 0)])+'\n')
    perform.write('Test\n')
    perform.write(','.join([str(i) for i in np.mean(test_perform_list, 0)])+'\n')
    perform.write(','.join([str(i) for i in np.std(test_perform_list, 0)])+'\n')


if __name__ == "__main__":
    main()
