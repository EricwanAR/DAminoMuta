import argparse
import json
import logging
import os
import math
from copy import deepcopy

from dataset import PeptidePairPicCaseDataset, PeptidePairPicDataset
from network import DMutaPeptide, DMutaPeptideCNN
from train import move_to_device, update_ce_loss_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from utils import set_seed, zip_restart_dataloader as zrd
from torchmetrics import MeanAbsoluteError, RelativeSquaredError, PearsonCorrCoef, KendallRankCorrCoef, Accuracy, F1Score, AveragePrecision, AUROC


parser = argparse.ArgumentParser(description='resnet26')
# model setting
parser.add_argument('--model', type=str, default='resnet34',
                    help='resnet34 resnet50 densenet')
parser.add_argument('--q-encoder', dest='q_encoder', type=str, default='lstm',
                    help='lstm mamba mla')
parser.add_argument("--side-enc", dest='side_enc', type=str, default=None,
                    help="use side features")
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--fusion', type=str, default='att',
                    help='mlp att diff')
parser.add_argument('--glob-feat', dest='glob_feat', action='store_true', default=False,
                    help="use global features")
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
parser.add_argument('--mix-pcs', dest='mix_pcs', action='store_true', default=False,
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
parser.add_argument('--pretrain', type=str, dest='pretrain', default='',
                    help='path of the pretrain model')
parser.add_argument('--metric-avg', type=str, dest='metric_avg', default='macro',
                    help='metric average type')

parser.add_argument('--loss', type=str, default='mse',
                    help='loss function')
parser.add_argument('--dir', action='store_true', default=False,
                    help='use DIR')

parser.add_argument('--case', type=str, default='r2')
parser.add_argument('--pt', action='store_true', default=False)

args = parser.parse_args()

class GaussianNoise(nn.Module):
    def __init__(self, mean=0., sigma=0.15):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.sigma = sigma

    def forward(self, x):
        return x + torch.randn_like(x) * self.sigma + self.mean

strong_transforms = T.Compose([
    T.RandomResizedCrop(args.resize, scale=(0.7, 1.0)),
    T.RandomGrayscale(0.2),
    GaussianNoise(0., 0.4),
])
weak_transforms = T.Compose([
    T.RandomResizedCrop(args.resize, scale=(0.9, 1.0)),
    GaussianNoise(0., 0.05),
])

def strong_aug(x, device=torch.device('cpu')):
    return aug_and_move(x, strong_transforms, 0.2, device, True)

def weak_aug(x, device=torch.device('cpu')):
    return aug_and_move(x, weak_transforms, 0.05, device, True)

def aug_and_move(x, transforms: T.Transform, seq_noise=0.05, device=torch.device('cpu'), non_blocking=False):
    if isinstance(x, (tuple, list)):
        return type(x)(aug_and_move(x_i, transforms, seq_noise, device, non_blocking) for x_i in x)
    if len(x.shape) == 3:
        return (x + torch.randn_like(x) * seq_noise).to(device, non_blocking=non_blocking)
    else:
        # return transforms(x.to(device, non_blocking=non_blocking))
        return torch.stack([transforms(s) for s in x.to(device, non_blocking=non_blocking)], dim=0)


def update_ema(student_model: nn.Module, teacher_model: nn.Module, alpha):
    for s_param, t_param in zip(student_model.parameters(), teacher_model.parameters()):
        t_param.data = t_param.data.mul_(alpha).add_(s_param.data, alpha=(1 - alpha))


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = max(0.0, min(1.0, current / rampup_length))
        return float(math.exp(-5.0 * (1.0 - current) * (1.0 - current)))
    

def consistency_loss_ce(s_pred, t_pred, threshold=None):
    probs = F.softmax(t_pred.detach(), dim=1)  # (B, C)
    max_probs, pseudo_labels = probs.max(dim=1)  # (B,), (B,)
    if threshold is None:
        mask = torch.ones_like(max_probs, dtype=torch.float)
    else:
        mask = max_probs.ge(threshold).float()  # (B,) 0/1
    loss = F.cross_entropy(s_pred, pseudo_labels, reduction='none')  # (B,)
    loss = (loss * mask).sum() / (mask.sum().clamp(min=1.0))
    return loss


def main():
    set_seed(args.seed)
    if args.task == 'reg':
        args.classes = 1
        if args.loss == "mse" or args.loss in ['ce']:
            args.loss = 'mse'
            criterion = nn.MSELoss()
            criterion_cons = criterion
        else:
            raise NotImplementedError("unimplemented regression task loss function")
    elif args.task == 'cls':
        args.classes = 2
        if args.loss == 'ce' or args.loss in ['mse', 'smoothl1', 'super']:
            args.loss = 'ce'
            criterion = nn.CrossEntropyLoss()
            criterion_cons = consistency_loss_ce
            # criterion_cons = nn.MSELoss()
        else:
            raise NotImplementedError("unimplemented classification task loss function")
    else:
        raise NotImplementedError("unimplemented task")

    if args.q_encoder in ['cnn', 'rn18']:
        weight_dir = f'./run-{args.task}/{args.q_encoder}{"-non-siamese" if args.non_siamese else ""}-{args.fusion}-{args.channels}{f"-{args.side_enc}" if args.side_enc else ""}{"-mixpcs" if args.mix_pcs else ""}{"-pcs" if args.pcs==True else ""}{"-" + "x".join(str(n) for n in args.resize) if args.resize else ""}{"-gf" if args.glob_feat else ""}{"-oneway" if args.one_way else ""}-{args.loss + "-dir" if args.dir else args.loss}-{str(args.batch_size)}-{str(args.lr)}-{str(args.epochs)}/uda_{args.case}'
    else:
        weight_dir = f'./run-{args.task}/{args.q_encoder}{f"-non-siamese" if args.non_siamese else ""}-{args.fusion}-{args.channels}{"-gf" if args.glob_feat else ""}{"-oneway" if args.one_way else ""}-{args.loss + "-dir" if args.dir else args.loss}-{str(args.batch_size)}-{str(args.lr)}-{str(args.epochs)}/uda_{args.case}'

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

    logging.info('Loading Training Dataset')
    unlabel_set = PeptidePairPicCaseDataset(case=args.case, pad_length=args.max_length, side_enc=args.side_enc, pcs=args.pcs, resize=args.resize, gf=args.glob_feat)
    unlabel_loader = DataLoader(unlabel_set, batch_size=args.batch_size // 2, shuffle=True, drop_last=True, num_workers=16,  pin_memory=True)

    label_set = PeptidePairPicDataset(mode='train', pad_length=args.max_length, task=args.task, side_enc=args.side_enc, pcs=args.pcs, resize=args.resize, one_way=args.one_way, gf=args.glob_feat)
    label_loader = DataLoader(label_set, batch_size=args.batch_size // 2, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)

    if args.case == 'r2':
        logging.info('Loading Validation Dataset')
        val_set = PeptidePairPicDataset(mode='r2_case', pad_length=args.max_length, task=args.task, side_enc=args.side_enc, pcs=args.pcs, resize=args.resize, gf=args.glob_feat)
        val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=16, pin_memory=True)
        metric_funcs = get_metric_funcs(args.task, device)
    else:
        val_loader = None

    best_val_metric = -float('inf')
    logging.info(f"Start UDA training")
    weights_path = f"{weight_dir}/model_uda_{'{role}'}.pth"

    student = DMutaPeptideCNN(q_encoder=args.q_encoder, classes=args.classes, channels=args.channels, dir=args.dir, gf=args.glob_feat, side_enc=args.side_enc, fusion=args.fusion, non_siamese=args.non_siamese).to(device).train()
    if args.pt:
        student.load_state_dict(torch.load(os.path.join(os.path.dirname(weight_dir), 'model_0_test.pth'), map_location=device))
    teacher = deepcopy(student).to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    
    global_step = 0
    rampup_length = 1500
    for epoch in range(1, args.epochs+1):
        train_loss = []
        for (x_l, y_l), (x_u, _) in zrd(label_loader, unlabel_loader):
            x_l = move_to_device(x_l, device, non_blocking=True)
            y_l = y_l.to(device, non_blocking=True)

            pred_l = student(x_l)
            if args.loss == 'ce':
                update_ce_loss_weight(criterion, y_l, num_classes=2, device=device)
            loss_sup = criterion(pred_l, y_l)

            with torch.no_grad():
                t_pred = teacher(weak_aug(x_u, device))

            s_pred = student(strong_aug(x_u, device))

            loss_cons = criterion_cons(s_pred, t_pred)

            λ = 1.0 * sigmoid_rampup(global_step, rampup_length)
            loss = loss_sup + λ * loss_cons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            alpha = 0.99
            update_ema(student, teacher, alpha)

            global_step += 1
            train_loss.append(loss.item())
        
        train_loss = sum(train_loss) / len(train_loss)

        if val_loader:
            with torch.no_grad():
                val_pred, val_gt = [], []
                for x, gt in val_loader:
                    x = move_to_device(x, device, non_blocking=True)
                    out = teacher(x)
                    # out = student(x)
                    val_pred.append(out)
                    val_gt.append(gt.to(device, non_blocking=True))
                val_pred = torch.cat(val_pred, dim=0)
                val_gt = torch.cat(val_gt, dim=0)

            if args.task == 'cls':
                val_pred = torch.softmax(val_pred, dim=1)[:, 1]
                val_ap = metric_funcs['ap'](val_pred, val_gt).item()
                val_auc = metric_funcs['auc'](val_pred, val_gt).item()
                val_f1 = metric_funcs['f1'](val_pred, val_gt).item()
                val_acc = metric_funcs['acc'](val_pred, val_gt).item()
                val_metric = val_ap + val_auc
                logging.info(f'Epoch {epoch} Train Loss: {train_loss:.4f} Val: ap: {val_ap:.4f} auc: {val_auc:.4f} f1: {val_f1:.4f} acc: {val_acc:.4f}')
            elif args.task == 'reg':
                val_mae = metric_funcs['mae'](val_pred, val_gt).item()
                val_rse = metric_funcs['rse'](val_pred, val_gt).item()
                val_pcc = metric_funcs['pcc'](val_pred, val_gt).item()
                val_kcc = metric_funcs['kcc'](val_pred, val_gt).item()
                val_metric = val_pcc + val_kcc - val_mae - val_rse
                logging.info(f'Epoch {epoch} Train Loss: {train_loss:.4f} Val: mae: {val_mae:.4f} rse: {val_rse:.4f} pcc: {val_pcc:.4f} kcc: {val_kcc:.4f}')
            else:
                raise NotImplementedError

            if val_metric > best_val_metric:
                best_val_metric = val_metric
                logging.info(f'Epoch: {epoch} New best VAL metrics')
                torch.save(student.state_dict(), weights_path.format(role='student'))
                torch.save(teacher.state_dict(), weights_path.format(role='teacher'))
        else:
            logging.info(f'Epoch {epoch} Train Loss: {train_loss:.4}')
            if (args.task == 'reg' and train_loss > 0.199) or (args.task == 'cls' and train_loss > 0.259):
                val_metric = -train_loss
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    torch.save(student.state_dict(), weights_path.format(role='student'))
                    torch.save(teacher.state_dict(), weights_path.format(role='teacher'))
            else:
                break
                
    logging.info('UDA training finished')
    torch.save(student.state_dict(), weights_path.format(role='student_last'))
    torch.save(teacher.state_dict(), weights_path.format(role='teacher_last'))


def get_metric_funcs(task, device):
    if task == 'reg':
        metric_funcs = {
            'mae': MeanAbsoluteError().to(device),
            'rse': RelativeSquaredError().to(device),
            'pcc': PearsonCorrCoef().to(device),
            'kcc': KendallRankCorrCoef().to(device)
        }
    elif task == 'cls':
        metric_funcs = {
            'ap': AveragePrecision(task='binary').to(device),
            'auc': AUROC(task='binary').to(device),
            'f1': F1Score(task='binary').to(device),
            'acc': Accuracy(task='binary').to(device)
        }
    else:
        raise NotImplementedError(f'Task {task} not supported')
    return metric_funcs

if __name__ == '__main__':
    main()