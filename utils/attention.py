import torch
import torch.nn as nn
import torch.nn.functional as F


class attention(nn.Module):
    def __init__(self, embedding_dim, droprate, cuda="cpu"):
        super(attention, self).__init__()
        self.embed_dim = embedding_dim
        self.droprate = droprate
        self.device = cuda
        self.att1 = nn.Linear(self.embed_dim * 4, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, feature1, feature2, n_neighs):
        feature2_reps = feature2.repeat(1, n_neighs, 1)
        x = torch.cat((feature1, feature2_reps), 2)
        x = self.relu(self.att1(x).to(self.device))
        x = F.dropout(x, training=self.training, p=self.droprate)
        x = self.att2(x).to(self.device)
        att = F.softmax(x, dim=0)
        return att
