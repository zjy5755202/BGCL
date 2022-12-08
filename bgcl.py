import torch.nn as nn
import torch.utils.data
from copy import deepcopy
import torch.nn.functional as F


class BGCL(nn.Module):
    def __init__(self, u_embedding, s_embedding, embed_dim, temp=0.5, droprate=0.5, beta_ema=0.999):
        super(BGCL, self).__init__()
        self.u_embed = u_embedding
        self.s_embed = s_embedding
        self.embed_dim = embed_dim
        self.droprate = droprate
        self.beta_ema = beta_ema


        self.u_layer1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.u_layer2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.s_layer1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.s_layer2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.us_layer1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.us_layer2 = nn.Linear(self.embed_dim, 1)

        self.u_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.s_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.us_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.layers = []

        if beta_ema > 0.:
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        self.ssl_temp = temp
        self.ssl_reg = 0.5
        self.ssl_ratio = 0.5
        self.delta = 0.5

        self.criterion = nn.L1Loss()
        # self.criterion = nn.MSELoss()
        # self.criterion = nn.SmoothL1Loss()
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)
        self.relu3 = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, nodes_u, nodes_s):
        nodes_u_embed, nodes_u_sge_1, nodes_u_sge_2 = self.u_embed(nodes_u, nodes_s)
        nodes_s_embed, nodes_s_sge_1, nodes_s_sge_2 = self.s_embed(nodes_u, nodes_s)

        x_u = self.relu1(self.u_bn(self.u_layer1(nodes_u_embed)))
        x_u = F.dropout(x_u, training=self.training, p=self.droprate)
        x_u = self.u_layer2(x_u)

        x_i = self.relu2(self.s_bn(self.s_layer1(nodes_s_embed)))
        x_i = F.dropout(x_i, training=self.training, p=self.droprate)
        x_i = self.s_layer2(x_i)

        x_ui = torch.cat((x_u, x_i), dim=1)
        x = self.relu3(self.us_bn(self.us_layer1(x_ui)))
        x = F.dropout(x, training=self.training, p=self.droprate)

        scores = self.us_layer2(x)
        return scores.squeeze(), self.ssl_loss(nodes_u_sge_1, nodes_u_sge_2, nodes_s_sge_1, nodes_s_sge_2)

    def regularization(self):
        regularization = 0
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        return regularization

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def loss(self, nodes_u, nodes_i, ratings):
        scores, ssl_loss = self.forward(nodes_u, nodes_i)
        loss = self.criterion(scores, ratings)
        total_loss = loss + self.ssl_ratio * ssl_loss + self.regularization()
        return total_loss

    def ssl_loss(self, nodes_u_sge_1, nodes_u_sge_2,
                 nodes_s_sge_1, nodes_s_sge_2):

        # cosine similarity
        normalize_user_emb_merge1 = torch.nn.functional.normalize(nodes_u_sge_1, p=2, dim=1)
        normalize_user_emb_merge2 = torch.nn.functional.normalize(nodes_u_sge_2, p=2, dim=1)
        user_pos_score = torch.sum(torch.multiply(normalize_user_emb_merge1, normalize_user_emb_merge2), axis=1)
        user_ttl_score = torch.matmul(normalize_user_emb_merge1, normalize_user_emb_merge2.t())
        user_pos_score = torch.exp(user_pos_score / self.ssl_temp)
        user_ttl_score = torch.sum(torch.exp(user_ttl_score / self.ssl_temp), axis=1)
        user_ssl_loss = -torch.sum(torch.log(user_pos_score / user_ttl_score))

        # cosine similarity
        normalize_item_emb_merge1 = torch.nn.functional.normalize(nodes_s_sge_1, p=2, dim=1)
        normalize_item_emb_merge2 = torch.nn.functional.normalize(nodes_s_sge_2, p=2, dim=1)
        item_pos_score = torch.sum(torch.multiply(normalize_item_emb_merge1, normalize_item_emb_merge2), axis=1)
        item_ttl_score = torch.matmul(normalize_item_emb_merge1, normalize_item_emb_merge2.t())
        item_pos_score = torch.exp(item_pos_score / self.ssl_temp)
        item_ttl_score = torch.sum(torch.exp(item_ttl_score / self.ssl_temp), axis=1)
        item_ssl_loss = -torch.sum(torch.log(item_pos_score / item_ttl_score))

        ssl_loss = user_ssl_loss + item_ssl_loss
        ssl_loss = self.ssl_reg * ssl_loss
        return ssl_loss
