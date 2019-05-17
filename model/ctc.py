import torch.nn as nn


class CTCLoss(nn.Module):

    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        super(CTCLoss, self).__init__()
        self.internal_loss = nn.CTCLoss(blank, reduction, zero_infinity)

    def forward(self, log_probs, targets):
        input_lengths = ()
        preds = log_probs.permute(1, 0, 2)
        for pred in preds:
            input_lengths = input_lengths + (len(pred),)

        target_lengths = ()
        for target in targets:
            target_lengths = target_lengths + (len(target),)

        return self.internal_loss(log_probs, targets, input_lengths, target_lengths)

