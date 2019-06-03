import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as models


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x):
        return func.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.conv_layer = models.vgg16_bn().features
        self.classifier = nn.Conv2d(512, 2, 3, 1, 1)
        self.deconv_layer = nn.Sequential(
            Upsample(),
            nn.Conv2d(2, 2, 3, 1, 0),
            Upsample(),
            nn.Conv2d(2, 2, 3, 1, 0),
            Upsample(),
            nn.Conv2d(2, 2, 3, 1, 0),
            Upsample(),
            nn.Conv2d(2, 2, 3, 1, 0),
            Upsample(),
            nn.Conv2d(2, 2, 3, 1, 0),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.classifier(x)
        return self.deconv_layer(x)


if __name__ == '__main__':
    model = FCN()
    samples = torch.randn(8, 3, 224, 224)
    output = model(samples)
