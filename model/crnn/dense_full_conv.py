import torch.nn as nn
import torch.nn.functional as fn
import torchvision.models as models


class DenseNetCTC(nn.Module):

    def __init__(self, num_classes):
        super(DenseNetCTC, self).__init__()
        densenet_features = models.densenet121().features
        densenet_features[0] = nn.Conv2d(3, 64, 3, 1, 1)
        self.features = densenet_features
        self.classifier = nn.Conv2d(1024, num_classes, (2, 1), (1, 1), 0)
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
        x = fn.leaky_relu(self.features(x), inplace=True)
        x = self.classifier(x)
        x = x.permute(3, 0, 1, 2)
        x = x.contiguous().view(x.shape[0], x.shape[1], -1)
        return fn.log_softmax(x, dim=2)
