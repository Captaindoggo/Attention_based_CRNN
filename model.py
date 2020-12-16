import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, inn, linn, n_classes):
        super(Attention, self).__init__()
        self.Wxb = nn.Linear(inn, inn)
        self.Vt = nn.Linear(inn, 1, bias=False)
        self.t = nn.Tanh()

        self.out = nn.Linear(linn, n_classes, bias=False)
        self.sftmx = nn.Softmax(dim=-1)
        self.lsftmx = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        apl_a = []
        for x_ in x:
          x_ = self.Wxb(x_)
          x_ = self.t(x_)
          x_ = self.Vt(x_)
          apl_a.append(x_)
        apl_a = torch.cat(apl_a, dim=1)
        x = x.transpose(0, 1)
        apl_a = self.sftmx(apl_a).unsqueeze(1)
        res = torch.bmm(apl_a, x).squeeze()
        res = self.lsftmx(self.out(res))
        return res


class AtnCRNN(nn.Module):
    def __init__(self, inn, hidden, num_layers, device, n_classes=2, nd=2):
        super(AtnCRNN, self).__init__()
        self.conv5 = nn.Conv1d(inn, inn, kernel_size=5, stride=2, dilation=1, groups=inn)
        self.conv20 = nn.Conv1d(inn, hidden, kernel_size=1, stride=8, groups=int(inn / 20))

        self.rnn = nn.GRU(hidden, hidden, num_layers=num_layers, dropout=0.1, bidirectional=True)
        self.num_layers = num_layers
        self.hidden = hidden
        self.device = device
        self.attention = Attention(hidden * 2, hidden * nd, n_classes)

    def forward(self, x):
        h = torch.zeros(self.num_layers * 2, x.shape[0], self.hidden).to(self.device)
        x = self.conv5(x)
        x = self.conv20(x)
        x = x.permute(2, 0, 1)
        # print(x.shape)

        x, hidn = self.rnn(x, h)
        x = self.attention(x)
        return x