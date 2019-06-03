import torch
import torch.nn as nn


class AttentionLoss(nn.Module):
    def __init__(self):
        super(AttentionLoss, self).__init__()
        self.internal_loss = nn.NLLLoss()
        self.device = torch.device("cpu")

    def forward(self, pred, target):
        addition = torch.ones((target.shape[0], 1)).long().to(self.device)
        target = torch.cat((target, addition), dim=1)
        target = target.t().contiguous().view(-1)
        return self.internal_loss(pred, target)

    def to(self, *args, **kwargs):
        self.device = args[0]
        if not isinstance(self.device, torch.device):
            raise KeyError('args #0 must be device')
        return super(AttentionLoss, self).to(*args, **kwargs)
