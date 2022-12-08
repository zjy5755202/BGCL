import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class combiner(nn.Module):
    def __init__(self, embedding1, embedding2, embedding_dim, cuda='cpu'):
        super(combiner, self).__init__()

        self.embedding1 = embedding1
        self.embedding2 = embedding2

        self.embed_dim = embedding_dim
        self.device = cuda
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, 2)
        self.relu = nn.LeakyReLU(0.2, inplace=False)

    # 返回用户嵌入矩阵以及服务嵌入矩阵
    def forward(self, nodes_u, nodes_i):
        # 用户嵌入矩阵
        embedding1 = self.embedding1(nodes_u, nodes_i)

        embedding2 = self.embedding2(nodes_u, nodes_i)

        # 聚合的地方(注意力聚合)
        x = torch.cat((embedding1, embedding2), dim=1)
        x = self.relu(self.att1(x).to(self.device))
        x = F.dropout(x, training=self.training)
        x = self.att2(x).to(self.device)

        #
        att_w = F.softmax(x, dim=1)
        att_w1, att_w2 = att_w.chunk(2, dim=1)
        att_w1.repeat(self.embed_dim, 1)
        att_w2.repeat(self.embed_dim, 1)
        #
        # 最终的嵌入矩阵
        final_embed_matrix = torch.mul(embedding1, att_w1) + torch.mul(embedding2, att_w2)

        return final_embed_matrix, embedding1, embedding2
        # return embedding1
