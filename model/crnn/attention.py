import torch
import torch.nn as nn
import torch.nn.functional as func

import torchvision.models as models


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.linear(x)


class Encoder(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super(Encoder, self).__init__()
        cnn_origin = models.densenet121()
        self.cnn = cnn_origin.features
        self.rnn = BidirectionalLSTM(cnn_origin.classifier.in_features, hidden_size, hidden_size)

    def forward(self, x: torch.Tensor):
        x = func.leaky_relu(self.cnn(x), inplace=True)
        (N, C, H, T) = x.shape
        x = x.permute((3, 0, 1, 2)).contiguous().view(T, N, -1)
        x = self.rnn(x)
        return x


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, classes):
        super(AttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(classes, hidden_size)
        self.attention_linear = nn.Linear(hidden_size, hidden_size)
        self.attention_weights_linear = nn.Linear(hidden_size, 1)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.atten_combine = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, classes)

    def forward(self, encoder_output: torch.Tensor, hidden, last_predict):
        atte_x = self.attention_linear(hidden[0])
        attention_weights = torch.tanh((atte_x + self.embedding(last_predict)))
        attention_weights = attention_weights.unsqueeze(1) + encoder_output
        attention_weights = func.softmax(self.attention_weights_linear(attention_weights), dim=1).squeeze(2)
        x = torch.matmul(attention_weights.unsqueeze(1), encoder_output)
        x = func.leaky_relu(self.atten_combine(x), inplace=True)
        x, hidden = self.gru(x.permute(1, 0, 2), hidden)
        x = func.log_softmax(self.fc(x.squeeze(0)), dim=1)
        return x, hidden, attention_weights


class Attention(nn.Module):
    def __init__(self, hidden_size, classes, max_length):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(hidden_size)
        self.decoder = AttentionDecoder(hidden_size, classes)
        self.max_length = max_length
        self.device = None

    def forward(self, x, *args, **kwargs):
        target = kwargs['target']
        addition = torch.ones((target.shape[0], 1)).long().to(self.device)
        target = torch.cat((target, addition), dim=1)

        encoder_output = self.encoder(x).permute(1, 0, 2)
        hidden = torch.zeros(1, encoder_output.shape[0], self.hidden_size).to(self.device)
        output_all = None
        for idx in range(target.shape[1]):
            output, hidden, _ = self.decoder(encoder_output, hidden, target[:, idx])
            _, topi = output.data.topk(1)
            if output_all is None:
                output_all = output
            else:
                output_all = torch.cat((output_all, output), dim=0)
                # (batch*max_length, hidden_size)
        return output_all

    def to(self, *args, **kwargs):
        self.device = args[0]
        if not isinstance(self.device, torch.device):
            raise KeyError('args #0 must be device')
        return super(Attention, self).to(*args, **kwargs)

