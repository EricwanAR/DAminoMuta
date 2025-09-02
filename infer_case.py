import argparse
import time
from dataset import PeptidePairPicCaseDataset, encode_sequence
from network import DMutaPeptideCNN
from train import move_to_device
import torch
from torch.utils.data import DataLoader
from utils import set_seed
import pandas as pd

parser = argparse.ArgumentParser()
# model setting
parser.add_argument('--q-encoder', dest='q_encoder', type=str, default='cnn',
                    help='lstm mamba mla')
parser.add_argument('--channels', type=int, default=16)
parser.add_argument("--side-enc", dest='side_enc', type=str, default='lstm',
                    help="use side features")
parser.add_argument('--fusion', type=str, default='att',
                    help='mlp att')
parser.add_argument('--non-siamese', dest='non_siamese', action='store_true', default=False,
                    help="use non-siamese architecture")

# task & dataset setting
parser.add_argument('--task', type=str, default='cls',
                    help='reg or cls')
parser.add_argument('--one-way', action='store_true', dest='one_way', default=False,
                    help='use one-way constructed dataset')
parser.add_argument('--max-length', dest='max_length', type=int, default=30,
                    help='Max length for sequence filtering')
parser.add_argument('--resize', type=int, default=[768], nargs='+',
                    help='resize the image')
parser.add_argument('--split', type=int, default=5,
                    help="Split k fold in cross validation (default: 5)")
parser.add_argument('--seed', type=int, default=1,
                    help="Seed for model initialization (default: 1)")
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

# Case Study Specific
parser.add_argument('--case', type=str, default='r2',
                    help='case to infer')
parser.add_argument('--uda', type='store_true', default=False)

args = parser.parse_args()


if args.gpu != -1:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')


class FasterModelForCase(DMutaPeptideCNN):
    def cache_temp_vector(self, seq):
        if self.side_enc:
            seq_seq = seq[1]
            seq = seq[0]
            if self.side_encoder.__class__.__name__ == 'MambaModel':
                self.temp_seq_vector = self.norm(self.side_encoder(seq_seq))
            else:
                self.temp_seq_vector = self.norm(self.side_encoder(seq_seq)[0][:, -1, :])
        self.temp_vector = self.norm(self.q_encoder(seq))

    def forward(self, x, labels=None, epoch=0):
        seq2 = x

        if self.side_enc:
            seq2_seq = seq2[1]
            seq2 = seq2[0]

        batch_size = seq2.shape[0]

        fusion = []

        fusion.append(self.temp_vector.expand(batch_size, -1))
        fusion.append(self.norm(self.q_encoder_2(seq2)))
        if self.side_enc:
            fusion.append(self.temp_seq_vector.expand(batch_size, -1))
            if self.side_encoder.__class__.__name__ == 'MambaModel':
                fusion.append(self.norm(self.side_encoder_2(seq2_seq)))
            else:
                fusion.append(self.norm(self.side_encoder_2(seq2_seq)[0][:, -1, :]))

        if self.fusion_method == 'mlp':
            fusion = torch.cat(fusion, dim=-1)
        elif self.fusion_method == 'diff':
            if not self.side_enc:
                fusion = torch.cat([fusion[1] - fusion[0]] + fusion[2:], dim=-1)
            else:
                fusion = torch.cat([fusion[1] - fusion[0], fusion[3] - fusion[2]] + fusion[4:], dim=-1)
        elif self.fusion_method == 'att':
            tokens = torch.stack(fusion, dim=1)
            attn_output, _ = self.attn(tokens, tokens, tokens)
            fusion = attn_output.reshape(attn_output.size(0), -1)
        else:
            raise ValueError("Invalid fusion method: choose either 'mse' or 'att'.")

        if self.DIR:
            features = fusion
            fusion = self.FDS.smooth(fusion, labels, epoch)
        
        pred = self.fc(fusion)

        if self.DIR:
            return pred, features
        else:
            return pred

class CustomDataset(PeptidePairPicCaseDataset):
    def __getitem__(self, idx):
        variant  = self.variants[idx]
        seq2, label = variant, variant
        img2 = self.read_img(variant)

        if self.side_enc:
            img2 = (img2, encode_sequence(seq2, self.pad_length))

        return img2, label

def load_model(args, weight_path, device, temp_batch):
    model = FasterModelForCase(q_encoder=args.q_encoder, classes=args.classes, channels=args.channels, dir=args.dir, gf=args.glob_feat, side_enc=args.side_enc, fusion=args.fusion, non_siamese=args.non_siamese).to(device).eval()
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    model.cache_temp_vector(move_to_device(temp_batch, device))
    model.compile()
    return model


def main():
    set_seed(args.seed)
    if args.task == 'reg':
        args.classes = 1
    elif args.task == 'cls':
        args.classes = 2
    else:
        raise NotImplementedError("unimplemented task")
    weight_dir = f'./run-{args.task}/{args.q_encoder}{f"-non-siamese" if args.non_siamese else ""}-{args.fusion}-{args.channels}{f"-{args.side_enc}" if args.side_enc else ""}{"-pcs" if args.pcs==True else ""}{"-" + "x".join(str(n) for n in args.resize) if args.resize else ""}{"-gf" if args.glob_feat else ""}{"-oneway" if args.one_way else ""}-{args.loss + "-dir" if args.dir else args.loss}-{str(args.batch_size)}-{str(args.lr)}-{str(args.epochs)}'

    if args.uda:
        weight_dir += f'/uda_{args.case}'
    
    device = torch.device("cpu" if args.gpu == -1 or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    
    print(weight_dir)
    print(device)
    
    test_set = CustomDataset(case=args.case, pad_length=args.max_length, side_enc=args.side_enc, pcs=True, resize=args.resize, gf=args.glob_feat)
    test_loader = DataLoader(test_set, batch_size=192, shuffle=False, num_workers=16, pin_memory=True)

    temp_batch = test_set.template_pic.unsqueeze(0)
    if args.side_enc:
        temp_batch = (temp_batch, test_set.template_seq.unsqueeze(0))
    
    ckpt_names = ['model_uda_teacher'] if args.uda else [f'model_{i}_test' for i in range(5)]
    models = [load_model(args, f'{weight_dir}/{i}.pth', device, temp_batch) for i in ckpt_names]

    all_seqs = []
    logits_batches = []
    start_time = time.time()

    with torch.no_grad():
        for x, gt in test_loader:
            x = move_to_device(x, device, non_blocking=True)

            logits = torch.zeros(len(models), len(gt), args.classes, device=device)
            for i, m in enumerate(models):
                logits[i] = m(x)
            
            logits_batches.append(logits.cpu())
            all_seqs.extend(gt)

    all_logits = torch.cat(logits_batches, dim=1)
    
    if args.task == 'reg':
        preds = all_logits.mean(0).squeeze().tolist()
    elif args.task == 'cls':
        preds = torch.softmax(all_logits, dim=-1)[:, :, 1].mean(0).tolist()

    consumed_time = time.time() - start_time
    print(f'total consumed time: {consumed_time} s')
    print(f'time per sample: {consumed_time / len(test_set)} s')

    df = pd.DataFrame({
        "seq":  all_seqs,
        "pred": preds,
    })

    df.to_csv(f'{weight_dir}/preds_case_{args.case}.csv', index=False)


if __name__ == '__main__':
    main()