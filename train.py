import torch
from torchmetrics import MeanAbsoluteError, RelativeSquaredError, PearsonCorrCoef, KendallRankCorrCoef, F1Score, Accuracy, AveragePrecision, AUROC


def move_to_device(batch, device, non_blocking=False):
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(item, device, non_blocking) for item in batch)
    return batch.to(device, non_blocking=non_blocking)


def train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    train_loss = 0
    num_labels = model.classes
    metric_mae = MeanAbsoluteError().to(device)
    metric_rse = RelativeSquaredError(num_outputs=num_labels).to(device)
    metric_pcc = PearsonCorrCoef(num_outputs=num_labels).to(device)
    metric_kcc = KendallRankCorrCoef(num_outputs=num_labels).to(device)

    if args.dir:
        encodings, labels = [], []

    if train_loader is not None:
        model.train()
        for data in train_loader:
            x, gt = data
            x = move_to_device(x, device)
            if args.dir:
                out, features = model(x,
                                      gt.to(device),
                                      epoch)
                encodings.append(features.detach().cpu())
                labels.append(gt.cpu())
            else:
                out = model(x)
            loss = criterion(out, gt.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        if args.dir:
            encodings, labels = torch.cat(encodings), torch.cat(labels)
            model.FDS.update_last_epoch_stats(epoch)
            model.FDS.update_running_stats(encodings, labels, epoch)
            encodings, labels = [], []
            

    model.eval()
    preds = []
    gt_list_valid = []
    with torch.no_grad():
        for data in valid_loader:
            x, gt = data
            x = move_to_device(x, device)
            gt_list_valid.append(gt.to(device))
            out = model(x)
            if args.dir:
                out, _ = out
            preds.append(out)

    # calculate metrics
    preds = torch.cat(preds, dim=0)
    gt_list_valid = torch.cat(gt_list_valid, dim=0)

    mae = metric_mae(preds, gt_list_valid).item()
    rse = metric_rse(preds, gt_list_valid).item()
    pcc = metric_pcc(preds.squeeze(), gt_list_valid.squeeze()).mean().item()
    kcc = metric_kcc(preds.squeeze(), gt_list_valid.squeeze()).mean().item()
    return train_loss, mae, rse, pcc, kcc


def update_ce_loss_weight(loss_fn: torch.nn.CrossEntropyLoss, gt: torch.Tensor, num_classes: int, device):
    """
    根据当前 batch 的 ground truth 标签更新 nn.CrossEntropyLoss 对象中的 weight 缓冲区，
    使用逆频率方法计算新权重，并通过 register_buffer 进行原地更新。
    
    参数:
      loss_fn (nn.CrossEntropyLoss): 已初始化的 nn.CrossEntropyLoss 对象，
                                      要求在初始化时已经注册了 weight 缓冲区。
      gt (torch.Tensor): 当前 batch 的 ground truth 标签，1D整数张量，标签取值范围 [0, num_classes-1]。
    """
    class_counts = torch.bincount(gt, minlength=num_classes).float()
    epsilon = 1e-6
    new_weights = 1.0 / (class_counts + epsilon)
    new_weights = new_weights / new_weights.sum() * num_classes
    # 使用 register_buffer 来更新 loss_fn 内部的 weight 缓冲区
    loss_fn.register_buffer('weight', new_weights.to(device))

def train_cls(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    train_loss = 0
    num_labels = model.classes
    avg = args.metric_avg
    if num_labels == 1 or num_labels == 2:
        task = 'binary'
    else:
        task = 'multiclass'
    metric_acc = Accuracy(average=avg, task=task, num_classes=num_labels).to(device)
    metric_f1 = F1Score(average=avg, task=task, num_classes=num_labels).to(device)
    metric_ap = AveragePrecision(average=avg, task=task, num_classes=num_labels).to(device)
    metric_auc = AUROC(average=avg, task=task, num_classes=num_labels).to(device)

    if train_loader is not None:
        model.train()
        for data in train_loader:
            x, gt = data
            x = move_to_device(x, device)
            out = model(x)
            update_ce_loss_weight(criterion, gt, num_classes=num_labels, device=device)
            loss = criterion(out, gt.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader)

    model.eval()
    preds = []
    gt_list_valid = []
    with torch.no_grad():
        for data in valid_loader:
            x, gt = data
            x = move_to_device(x, device)
            gt_list_valid.append(gt.to(device))
            out = model(x)
            preds.append(out)

    # calculate metrics
    preds = torch.softmax(torch.cat(preds, dim=0), dim=-1).squeeze()
    gt_list_valid = torch.cat(gt_list_valid, dim=0).int().squeeze()

    if num_labels == 2:
        preds = preds[:, 1]

    ap = metric_ap(preds, gt_list_valid).item()
    auc = metric_auc(preds, gt_list_valid).item()
    f1 = metric_f1(preds, gt_list_valid).item()
    acc = metric_acc(preds, gt_list_valid).item()
    return train_loss, ap, auc, f1, acc

def train_base(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    train_loss = 0
    num_labels = model.classes
    metric_mae = MeanAbsoluteError().to(device)
    metric_rse = RelativeSquaredError(num_outputs=num_labels).to(device)
    metric_pcc = PearsonCorrCoef(num_outputs=num_labels).to(device)
    metric_kcc = KendallRankCorrCoef(num_outputs=num_labels).to(device)

    if args.dir:
        encodings, labels = [], []

    if train_loader is not None:
        model.train()
        for data in train_loader:
            seq1, gt = data
            if args.dir:
                out, features = model(seq1.to(device),
                                      gt.to(device),
                                      epoch)
                encodings.append(features.detach().cpu())
                labels.append(gt.cpu())
            else:
                out = model(seq1.to(device))
            loss = criterion(out, gt.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        if args.dir:
            encodings, labels = torch.cat(encodings), torch.cat(labels)
            model.FDS.update_last_epoch_stats(epoch)
            model.FDS.update_running_stats(encodings, labels, epoch)
            encodings, labels = [], []
            

    model.eval()
    preds = []
    gt_list_valid = []
    with torch.no_grad():
        for data in valid_loader:
            seq1, gt = data
            gt_list_valid.append(gt.to(device))
            out = model(seq1.to(device))
            if args.dir:
                out, _ = out
            preds.append(out)

    # calculate metrics
    preds = torch.cat(preds, dim=0)
    gt_list_valid = torch.cat(gt_list_valid, dim=0)

    mae = metric_mae(preds, gt_list_valid).item()
    rse = metric_rse(preds, gt_list_valid).item()
    pcc = metric_pcc(preds.squeeze(), gt_list_valid.squeeze()).mean().item()
    kcc = metric_kcc(preds.squeeze(), gt_list_valid.squeeze()).mean().item()
    return train_loss, mae, rse, pcc, kcc