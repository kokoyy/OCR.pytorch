import torch
import torch.nn as nn
from torch_baidu_ctc import CTCLoss as internalCTC


class CTCLoss(nn.Module):

    def __init__(self, blank=0, reduction='mean', use_baidu=False):
        super(CTCLoss, self).__init__()
        self.use_baidu_implement = use_baidu
        if self.use_baidu_implement:
            self.internal_loss = internalCTC(reduction=reduction, blank=blank)
        else:
            self.internal_loss = nn.CTCLoss(blank=blank, reduction=reduction)

    def forward(self, log_probs, targets):
        input_lengths = ()
        preds = log_probs.permute(1, 0, 2)
        for pred in preds:
            input_lengths = input_lengths + (len(pred),)

        target_lengths = ()
        for target in targets:
            target_lengths = target_lengths + (len(target),)

        if self.use_baidu_implement:
            # warp_ctc
            targets = targets.view((-1)).cpu()
            input_lengths = torch.IntTensor(input_lengths)
            target_lengths = torch.IntTensor(target_lengths)

        return self.internal_loss(log_probs, targets, input_lengths, target_lengths)

