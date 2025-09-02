import argparse
from dataset import PeptidePairDataset, PeptidePairPicDataset
from network import DMutaPeptide, DMutaPeptideCNN
from train import move_to_device
import torch
from torch.utils.data import DataLoader
from utils import set_seed
import pandas as pd
from torchmetrics import MeanAbsoluteError, RelativeSquaredError, PearsonCorrCoef, KendallRankCorrCoef, F1Score, Accuracy, AveragePrecision, AUROC

parser = argparse.ArgumentParser()
# model setting
parser.add_argument('--q-encoder', dest='q_encoder', type=str, default='cnn',
                    help='lstm mamba mla')
parser.add_argument('--channels', type=int, default=16)
parser.add_argument("--side-enc", dest='side_enc', type=str, default=None,
                    help="use side features")
parser.add_argument('--fusion', type=str, default='mlp',
                    help='mlp att')
parser.add_argument('--non-siamese', dest='non_siamese', action='store_true', default=False,
                    help="use non-siamese architecture")

# task & dataset setting
parser.add_argument('--one-way', action='store_true', dest='one_way', default=False,
                    help='use one-way constructed dataset')
parser.add_argument('--max-length', dest='max_length', type=int, default=30,
                    help='Max length for sequence filtering')
parser.add_argument('--resize', type=int, default=[768], nargs='+',
                    help='resize the image')
parser.add_argument('--split', type=int, default=5,
                    help="Split k fold in cross validation (default: 5)")
parser.add_argument('--seed', type=int, default=1,
                    help="Seed (default: 1)")
parser.add_argument('--pcs', action='store_true', default=False,
                    help='Consider protease cut site')

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

parser.add_argument('--loss', type=str, default='ce',
                    help='loss function')
parser.add_argument('--dir', action='store_true', default=False,
                    help='use DIR')

parser.add_argument('--uda', type=str, default=None)

args = parser.parse_args()


if args.q_encoder in ['cnn', 'rn18']:
    weight_dir = f'./run-{args.task}/{f"non-siamese-" if args.non_siamese else ""}{args.q_encoder}-{args.fusion}-{args.channels}{f"-{args.side_enc}" if args.side_enc else ""}{"-pcs" if args.pcs==True else ""}{"-" + "x".join(str(n) for n in args.resize) if args.resize else ""}{"-gf" if args.glob_feat else ""}{"-oneway" if args.one_way else ""}-{args.loss + "-dir" if args.dir else args.loss}-{str(args.batch_size)}-{str(args.lr)}-{str(args.epochs)}'
else:
    weight_dir = f'./run-{args.task}/{f"non-siamese-" if args.non_siamese else ""}{args.q_encoder}-{args.fusion}-{args.channels}{"-gf" if args.glob_feat else ""}{"-oneway" if args.one_way else ""}-{args.loss + "-dir" if args.dir else args.loss}-{str(args.batch_size)}-{str(args.lr)}-{str(args.epochs)}'

if args.uda:
    weight_dir += f'/uda_{args.uda}'

print(weight_dir)

def metrics(preds, gt, task):
    avg = 'marco'
    device = preds.device
    if task == 'cls':
        metric_1 = AveragePrecision(average=avg, task='binary').to(device)
        metric_2 = AUROC(average=avg, task='binary').to(device)
        metric_3 = F1Score(average=avg, task='binary').to(device)
        metric_4 = Accuracy(average=avg, task='binary').to(device)
        all_metrics = [metric_1(preds, gt).item(), 
                metric_2(preds, gt).item(), 
                metric_3(preds, gt).item(), 
                metric_4(preds, gt).item()]
    
    elif task == 'reg':
        metric_1 = MeanAbsoluteError().to(device)
        metric_2 = RelativeSquaredError(num_outputs=1).to(device)
        metric_3 = PearsonCorrCoef(num_outputs=1).to(device)
        metric_4 = KendallRankCorrCoef(num_outputs=1).to(device)
        all_metrics = [metric_1(preds, gt).item(), 
                metric_2(preds, gt).item(), 
                metric_3(preds.squeeze(), gt.squeeze()).mean().item(), 
                metric_4(preds.squeeze(), gt.squeeze()).mean().item()]

    return [f'{i * 100:.2f}' for i in all_metrics]
    

def main(dataset):
    set_seed(args.seed)
    if args.task == 'reg':
        args.classes = 1
    elif args.task == 'cls':
        args.classes = 2
    else:
        raise NotImplementedError("unimplemented task")

    device = torch.device("cpu" if args.gpu == -1 or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    
    if args.q_encoder in ['cnn', 'rn18']:
        model = DMutaPeptideCNN(q_encoder=args.q_encoder, classes=args.classes, channels=args.channels, dir=args.dir, gf=args.glob_feat, side_enc=args.side_enc, fusion=args.fusion, non_siamese=args.non_siamese).to(device).eval()
        test_set = PeptidePairPicDataset(mode=dataset, pad_length=args.max_length, task=args.task, gf=args.glob_feat, side_enc=args.side_enc, pcs=args.pcs, resize=args.resize)

    else:
        model = DMutaPeptide(q_encoder=args.q_encoder, classes=args.classes, channels=args.channels, dir=args.dir, gf=args.glob_feat, fusion=args.fusion, non_siamese=args.non_siamese).to(device).eval()
        test_set = PeptidePairDataset(mode=dataset, pad_length=args.max_length, task=args.task, gf=args.glob_feat)
    
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    df = pd.DataFrame()
    raw_preds = []
    ckpt_names = ['model_uda_teacher'] if args.uda else [f'model_{i}_test' for i in range(5)]
    for i in ckpt_names:
        model.load_state_dict(torch.load(f'{weight_dir}/{i}.pth', map_location=device))
        preds = []
        gt_list_valid = []
        with torch.no_grad():
            for data in test_loader:
                x, gt = data
                gt_list_valid.append(gt.to(device))
                out = model(move_to_device(x, device))
                if args.dir:
                    out, _ = out
                preds.append(out)
        r_pred = torch.cat(preds, dim=0)
        if args.task == 'reg':
            preds = r_pred.cpu().numpy()
        elif args.task == 'cls':
            preds = torch.softmax(r_pred, dim=-1)[:, 1].cpu().numpy()
        gt_tensor = torch.cat(gt_list_valid, dim=0)
        gt_list_valid = gt_tensor.cpu().numpy()
        df[f'{i}'] = preds
        raw_preds.append(r_pred)
    if args.task == 'cls':
        preds_tensor = torch.softmax(torch.stack(raw_preds, 0).mean(0), dim=-1)[:, 1]
    elif args.task == 'reg':
        preds_tensor = torch.stack(raw_preds, 0).mean(0)
    df['fusion'] = preds_tensor.cpu().numpy()
    df['gt'] = gt_list_valid
    df.to_csv(f'{weight_dir}/preds_{dataset}.csv', index=False)
    return metrics(preds_tensor, gt_tensor, args.task)


if __name__ == '__main__':
    if args.task == 'cls':
        df = pd.DataFrame(columns=['dataset', 'AUPRC', 'AUROC', 'F1', 'ACC'])
    elif args.task == 'reg':
        df = pd.DataFrame(columns=['dataset', 'MAE', 'RSE', 'PCC', 'KCC'])

    datasets = [
        'test',
        'r2_case',
        ]

    for dataset in datasets:
        results = main(dataset)
        df.loc[len(df) + 1] = [dataset] + results
    df.to_csv(f'{weight_dir}/inference_results.csv', index=False)
    print(df)
