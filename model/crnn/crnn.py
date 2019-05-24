import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, in_channel, hidden, out_channel):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(in_channel, hidden, bidirectional=True)
        self.embedding = nn.Linear(hidden * 2, out_channel)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        t, b, h = recurrent.size()
        t_rec = recurrent.view(t * b, h)
        output = self.embedding(t_rec)
        output = output.view(t, b, -1)
        return output


class CRNN(nn.Module):

    def __init__(self, image_height, input_channel, classes, output_channel, leaky_relu=False, lstm=True):
        """
        是否加入lstm特征层
        """
        super(CRNN, self).__init__()
        assert image_height % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        self.lstm = lstm

        cnn = nn.Sequential()

        def convRelu(i, batch_normalization=False):
            nIn = input_channel if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batch_normalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leaky_relu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 2), (1, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        if self.lstm:
            self.rnn = nn.Sequential(
                BidirectionalLSTM(512, output_channel, output_channel),
                BidirectionalLSTM(output_channel, output_channel, classes))
        else:
            self.linear = nn.Linear(output_channel * 2, classes)

    def forward(self, x):
        conv = self.cnn(x)

        b, c, h, w = conv.size()

        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        if self.lstm:
            output = self.rnn(conv)
        else:
            t, b, h = conv.size()
            t_rec = conv.contiguous().view(t * b, h)
            output = self.linear(t_rec)
            output = output.view(t, b, -1)

        return output
