import torch
import torch.nn as nn
import torch.nn.functional as F


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()
        self.lr1 = nn.Linear(2, 10)
        self.lr2 = nn.Linear(10, 10)
        self.lr3 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = F.relu(F.dropout(self.lr1(x), p=0.5, training=True))
        # x = F.relu(F.dropout(self.lr2(x), p=0.5, training=True))
        return self.lr3(x)


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 8),
                                     nn.ReLU(True),
                                     nn.Linear(8, 8),
                                     nn.ReLU(True),
                                     nn.Linear(8, 8),
                                     nn.ReLU(True),
                                     nn.Linear(8, 1))
        self.decoder = nn.Sequential(nn.Linear(1, 8),
                                     nn.ReLU(True),
                                     nn.Linear(8, 8),
                                     nn.ReLU(True),
                                     nn.Linear(8, 8),
                                     nn.ReLU(True),
                                     nn.Linear(8, 2))

    def forward(self, x):
        return self.decoder(self.encoder(x))

class RaPP(nn.Module):
    def __init__(self):
        super(RaPP, self).__init__()
        self.enc_layer_list = [nn.Linear(2, 8),
                               nn.ReLU(True),
                                nn.Linear(8, 8),
                                nn.ReLU(True),
                                nn.Linear(8, 8),
                                nn.ReLU(True),
                                nn.Linear(8, 1)
                               ]
        self.encoder = nn.Sequential(*self.enc_layer_list)
        self.decoder = nn.Sequential(nn.Linear(1, 8),
                                     nn.ReLU(True),
                                     nn.Linear(8, 8),
                                     nn.ReLU(True),
                                     nn.Linear(8, 8),
                                     nn.ReLU(True),
                                     nn.Linear(8, 2))

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == "__main__":
    toy = torch.zeros(100, 2).cuda()
    model = Dummy().cuda()
    print("str : ", model)

    mod = RaPP()