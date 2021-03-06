import torch.nn as nn
import torch.nn.functional as fn
import torchvision.models as models


class DenseNetLinear(nn.Module):

    def __init__(self, num_classes, conv0=None, pool0=None, avg_pool=True):
        super(DenseNetLinear, self).__init__()
        densenet_features = models.densenet121().features
        if conv0 is not None:
            assert conv0.in_channels == 3
            assert conv0.out_channels == 64
            densenet_features[0] = conv0
        if pool0 is not None:
            densenet_features[3] = pool0
        self.features = densenet_features
        if avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.classifier = nn.Linear(1024, num_classes)
        #  init weights and bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = fn.relu(self.features(x), inplace=True)
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = x.permute(3, 0, 1, 2)
        x = x.contiguous().view(x.shape[0], x.shape[1], -1)
        x = self.classifier(x)
        return fn.log_softmax(x, dim=2)
