import torch.nn as nn
import torch.nn.functional as F

VGG_FEATURES_MAPPING = {
    'features.0.weight': 'conv1_1.weight',
    'features.0.bias': 'conv1_1.bias',
    'features.2.weight': 'conv1_2.weight',
    'features.2.bias': 'conv1_2.bias',
    'features.5.weight': 'conv2_1.weight',
    'features.5.bias': 'conv2_1.bias',
    'features.7.weight': 'conv2_2.weight',
    'features.7.bias': 'conv2_2.bias',
    'features.10.weight': 'conv3_1.weight',
    'features.10.bias': 'onv3_1.bias',
    'features.12.weight': 'conv3_2.weight',
    'features.12.bias': 'conv3_2.bias',
    'features.14.weight': 'conv3_3.weight',
    'features.14.bias': 'conv3_3.bias',
    'features.17.weight': 'conv4_1.weight',
    'features.17.bias': 'conv4_1.bias',
    'features.19.weight': 'conv4_2.weight',
    'features.19.bias': 'conv4_2.bias',
    'features.21.weight': 'conv4_3.weight',
    'features.21.bias': 'conv4_3.bias',
    'features.24.weight': 'conv5_1.weight',
    'features.24.bias': 'conv5_1.bias',
    'features.26.weight': 'conv5_2.weight',
    'features.26.bias': 'conv5_2.bias',
    'features.28.weight': 'conv5_3.weight',
    'features.28.bias': 'conv5_3.bias',
}


class VGG_16(nn.Module):
    def __init__(self):
        super(VGG_16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.activeF = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.activeF(self.conv1_1(input))
        x = self.activeF(self.conv1_2(x))
        x = self.pool1(x)

        x = self.activeF(self.conv2_1(x))
        x = self.activeF(self.conv2_2(x))
        x = self.pool2(x)

        x = self.activeF(self.conv3_1(x))
        x = self.activeF(self.conv3_2(x))
        x = self.activeF(self.conv3_3(x))
        x = self.pool3(x)

        x = self.activeF(self.conv4_1(x))
        x = self.activeF(self.conv4_2(x))
        x = self.activeF(self.conv4_3(x))
        x = self.pool4(x)

        x = self.activeF(self.conv5_1(x))
        x = self.activeF(self.conv5_2(x))
        x = self.activeF(self.conv5_3(x))
        return x


class BLSTM(nn.Module):
    def __init__(self, channel, hidden_unit, bidirectional=True):
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(channel, hidden_unit, bidirectional=bidirectional)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x, _ = self.lstm(x[0])
        x = x.unsqueeze(0)
        x = x.permute(0, 3, 2, 1)
        return x


# 作用不是很清楚 (N, C, H, W) -> (N, C*pad^2, H, W)
class conv2lstm(nn.Module):
    def __init__(self, kernel_size, padding, stride):
        super(conv2lstm, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        height = x.shape[2]

        x = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        x = x.reshape((x.shape[0], x.shape[1], height, -1))
        return x


class CTPN(nn.Module):
    def __init__(self, vgg_model):
        super(CTPN, self).__init__()
        self.features = nn.Sequential()
        vgg = VGG_16()
        vgg_state = {}
        for name, value in vgg_model:
            if name not in VGG_FEATURES_MAPPING:
                continue
            vgg_state[VGG_FEATURES_MAPPING[name]] = value
        vgg.state_dict().update(vgg_state)
        self.cnn = vgg
        self.rnn = BLSTM(512, 128)
        self.fc = nn.Conv2d(256, 512, 1)

        self.vertical_coordinate = nn.Conv2d(512, 2 * 10, 1)  # 最终输出2K个参数（k=10），10表示anchor的尺寸个数，2个参数分别表示anchor的h和dy
        self.score = nn.Conv2d(512, 2 * 10, 1)  # 最终输出是2K个分数（k=10），2表示有无字符，10表示anchor的尺寸个数
        self.side_refinement = nn.Conv2d(512, 10, 1)  # 最终输出1K个参数（k=10），该参数表示该anchor的水平偏移，用于精修文本框水平边缘精度，，10表示anchor的尺寸个数

    def forward(self, x, val=False):
        x = self.cnn(x)
        x = self.rnn(x)
        x = self.fc(x)
        x = F.relu(x, inplace=True)
        vertical_pred = self.vertical_coordinate(x)
        score = self.score(x)
        side_refinement = self.side_refinement(x)

        if val:
            score = score.reshape((score.shape[0], 10, 2, score.shape[2], score.shape[3]))
            score = score.squeeze(0)
            score = score.transpose(1, 2)
            score = score.transpose(2, 3)
            score = score.reshape((10, vertical_pred.shape[2], -1, 2))
            vertical_pred = vertical_pred.reshape(
                (vertical_pred.shape[0], 10, 2, vertical_pred.shape[2], vertical_pred.shape[3]))

        return score, vertical_pred, side_refinement


if __name__ == '__main__':
    import cv2
    import torchvision.models as models
    import torchvision.transforms as transfroms
    transform = transfroms.ToTensor()
    model = CTPN(models.vgg16_bn(True).named_parameters())
    pic = cv2.imread('/home/yuanyi/Pictures/test.png', cv2.IMREAD_COLOR)
    for name, val in model.named_parameters():
        print(name)
