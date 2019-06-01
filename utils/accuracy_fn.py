import numpy as np
import torch


def multi_label_accuracy_fn(label_transformer, top_k, preds, targets):
    parsed_preds = np.array(label_transformer.parse_prediction(preds, top_k=max(top_k), to_string=False))

    res = []
    num_correct = {}
    for k in top_k:
        num_correct[k] = 0

    for index in range(0, max(top_k)):
        for pred, target in zip(parsed_preds[:, index, :], targets):
            pred = list(filter(lambda x: x != 0, pred))
            if not (len(pred) == len(target) and (np.array(pred) == target.cpu().numpy()).all()):
                continue
            for key in num_correct:
                if key <= index:
                    continue
                num_correct[key] += 1

    for key in num_correct:
        res.append(num_correct[key] / targets.size(0))
    return res


def default_accuracy_fn(label_transformer, top_k, output, target):
    """Computes the precision@k for the specified values of k"""
    target = torch.cat((target.cpu(), torch.ones((target.shape[0], 1)).long()), dim=1)
    target = target.t().contiguous().view(-1, 1)
    output = label_transformer.parse_prediction(output, top_k=max(top_k), to_string=False)
    maxk = max(top_k)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().cpu()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res
