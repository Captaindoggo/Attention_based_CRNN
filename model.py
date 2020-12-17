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

    def inference_layer(self, x):
        self.eval()
        return self.Vt(self.t(self.Wxb(x)))

    def inference_attn(self, apl_a, x):
        self.eval()
        x = x.transpose(0, 1)
        apl_a = self.sftmx(apl_a).unsqueeze(1)
        res = torch.bmm(apl_a, x).squeeze()
        res = self.lsftmx(self.out(res))
        return res

class AtnCRNN(nn.Module):
    def __init__(self, inn, hidden, num_layers, device, ks=5, n_classes=2, nd=2):
        super(AtnCRNN, self).__init__()
        self.ks=ks
        self.conv5 = nn.Conv1d(inn, inn, kernel_size=ks, stride=2, dilation=1, groups=inn)
        self.conv20 = nn.Conv1d(inn, hidden, kernel_size=1, stride=8, groups=int(inn/20))

        self.rnn = nn.GRU(hidden, hidden, num_layers=num_layers, dropout=0.1, bidirectional=True)
        self.num_layers=num_layers
        self.hidden=hidden
        self.device=device
        self.attention = Attention(hidden*2, hidden*nd, n_classes)

    def forward(self, x):
        h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden).to(self.device)
        x = self.conv5(x)
        x = self.conv20(x)
        x = x.permute(2, 0, 1)
        #print(x.shape)
        
        x, hidn = self.rnn(x, h)
        x = self.attention(x)
        return x
    
    def inference_crnn(self, x, h):
        self.eval()
        x = self.conv5(x)
        x = self.conv20(x)
        x = x.permute(2, 0, 1)
        x, hidn = self.rnn(x, h)
        return x, hidn
    

    def predict(self, wav, melspec, confidence=0.5):
        flag_kw = False
        mel = melspec(wav).unsqueeze(0).to(self.device)
        kw_p = []
        with torch.no_grad():
          l = 100
          r = mel.shape[2] - l+1
          l = l - self.ks
          hidn = torch.zeros(self.num_layers*2, mel.shape[0], self.hidden).to(self.device)
          outs, hidn = self.inference_crnn(mel[:, :, 0:l], hidn)
          xs = []
          for x in outs:
            x_ = self.attention.inference_layer(x)
            xs.append(x_)
          res_x = torch.cat(xs, dim=1)
          p = self.attention.inference_attn(res_x, outs)
          kw_p.append(np.exp(p[1]))
          i = self.ks
          while i < r:
            xs = xs[1:]
            outs = outs[1:]
            frame = mel[:, :, i+l : i+l+self.ks]
            out, hidn = self.inference_crnn(frame, hidn)
            outs = torch.cat((outs, out))

            x_ = self.attention.inference_layer(out.squeeze(0))
            xs.append(x_)
            res_x = torch.cat(xs, dim=1)

            p = self.attention.inference_attn(res_x, outs)
            kw_p.append(np.exp(p[1]))
            if np.exp(p[1]) > confidence:
              if not flag_kw:
                print('Audio includes keyword')
                flag_kw = True
            i += self.ks
        
        plt.plot(np.zeros(len(kw_p))+confidence, label='minimum confidence')
        plt.plot(kw_p, label = 'probabilities of keyword')
        plt.legend()
        plt.show()
