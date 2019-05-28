import torch.nn as nn
import torch.nn.functional as fn


class DenseBLSTMCTC(nn.Module):

    def __init__(self, num_classes):
        super(DenseBLSTMCTC, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )

        self.lstm1 = nn.LSTM(512 * 4, 256, bidirectional=True)
        self.linear1 = nn.Linear(512, 512)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True)
        self.linear2 = nn.Linear(512, num_classes)

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
        x = self.cnn(x)
        x = x.permute(3, 0, 1, 2)
        x = x.contiguous().view(x.shape[0], x.shape[1], -1)

        x, _ = self.lstm1(x)
        x = self.linear1(x)

        x, _ = self.lstm2(x)
        x = self.linear2(x)
        return fn.log_softmax(x, dim=2)
