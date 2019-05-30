import math

import torch
import torch.nn as nn


class CTPNLoss(nn.Module):
    def __init__(self, cuda=False):
        super(CTPNLoss, self).__init__()
        self.scoreLoss = nn.CrossEntropyLoss()
        self.verticalLoss = nn.SmoothL1Loss()
        self.sideLoss = nn.SmoothL1Loss()
        self.use_cuda = cuda

    def forward(self, output, target):
        """
        :param target:
        :param output:
        :param score: shape=(1, 2k, h, w), 一个是文字得分，一个是非文字得分, 等同于训练一个二分类.
        :param vertical_pred: shape=(1, 2k, h, w) anchor的Y轴坐标和anchor的高度
        :param side_refinement_pred: shape=(1, k, h, w) anchor的X轴偏移量
        :param positive: shape=(j, i, k, center) 分类为文字的anchor sample; 用于和score做计算，判断该anchor所属分类
                         在conv5 feature-map上第j行的第i列的proposal area上的第k个anchor（在conv5上每个位置都有10个anchor）
                         center: anchor在Y轴上的中心点
        :param negative: 分类为非文字的anchor sample
        :param vertical_reg: shape=(j, i, k, vc, vh)
                             vc: 提议的anchor box和真实文字anchor box的 Y轴距离度量
                                 vc = (yc(groundTruth) - yc(proposalAnchorBox)) / h(proposalAnchorBox)
                             vh: 提议的anchor box和真实文字anchor box的 高度差度量
                                 log(h(groundTruth) / h(proposalAnchorBox))
        :param side_refinement_reg: shape = (j, i, k, vw) vw: 提议的anchor box和真实文字anchor box的 宽度差度量
        :return:

        """
        score, vertical_pred, side_refinement_pred = output[:]
        positive, negative, vertical_reg, side_refinement_reg = target[:]
        cls_loss = 0.
        for sample in positive:
            sample = self.unpack(sample)
            target = torch.LongTensor([1])
            pred_score = score[0, sample[2] * 2:sample[2] * 2 + 2, sample[0], sample[1]].unsqueeze(0)
            cls_loss += self.scoreLoss(pred_score, target.cuda() if self.use_cuda else target)
        for sample in negative:
            sample = self.unpack(sample)
            target = torch.LongTensor([0])
            pred_score = score[0, sample[2] * 2:sample[2] * 2 + 2, sample[0], sample[1]].unsqueeze(0)
            cls_loss += self.scoreLoss(pred_score, target.cuda() if self.use_cuda else target)
        cls_loss = 0. if (len(positive) + len(negative)) == 0 else cls_loss / (len(positive) + len(negative))

        vertical_loss = 0.
        for sample in vertical_reg:
            sample = self.unpack(sample)
            target = torch.FloatTensor([sample[3:]])
            pred_vertical = vertical_pred[0, sample[2] * 2:sample[2] * 2 + 2, sample[0], sample[1]].unsqueeze(0)
            vertical_loss += self.verticalLoss(pred_vertical, target.cuda() if self.use_cuda else target)
        vertical_loss = 0. if len(vertical_reg) == 0 else vertical_loss / len(vertical_reg)

        side_refinement_loss = 0.
        for sample in side_refinement_reg:
            sample = self.unpack(sample)
            target = torch.FloatTensor([sample[3]])
            pred_side_refinement = side_refinement_pred[0, sample[2], sample[0], sample[1]].unsqueeze(0)
            side_refinement_loss += self.sideLoss(pred_side_refinement, target.cuda() if self.use_cuda else target)
        side_refinement_loss = 0. if len(side_refinement_reg) == 0 else side_refinement_loss / len(side_refinement_reg)

        total_loss = cls_loss + vertical_loss + side_refinement_loss
        total_loss = torch.FloatTensor([math.inf]) if total_loss == 0 else total_loss
        print('total_loss', total_loss, 'cls_loss:', cls_loss, 'vertical_loss:',
              vertical_loss, 'side_refinement_loss:', side_refinement_loss)
        return total_loss

    @staticmethod
    def unpack(data):
        if not isinstance(data[0], torch.Tensor):
            return data
        else:
            output = []
            for x in data:
                output.append(x.item())
            return output
