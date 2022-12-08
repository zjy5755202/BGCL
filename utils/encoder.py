import torch
import torch.nn as nn

from utils.attention import attention
from utils.batchutils import get_neighbor_location


class encoder(nn.Module):
    def __init__(self, embedding_dim, un, sn, u2l, s2l, droprate, num_neighbor, num_user=339, num_service=5825,
                 num_user_location=137,
                 num_service_location=991, cuda="cpu", is_user_part=True):
        super(encoder, self).__init__()

        self.embed_dim = embedding_dim
        self.device = cuda
        self.is_user = is_user_part
        self.un = un
        self.sn = sn
        self.u2l = u2l
        self.s2l = s2l
        self.num_neighbor = num_neighbor
        self.droprate = droprate
        self.layer1 = nn.Linear(self.embed_dim * 4, self.embed_dim)
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)
        self.layer2 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)

        self.u_embedding_layer = nn.Embedding(num_user, self.embed_dim)
        self.s_embedding_layer = nn.Embedding(num_service, self.embed_dim)
        self.u_location_embedding_layer = nn.Embedding(num_user_location, self.embed_dim)
        self.s_location_embedding_layer = nn.Embedding(num_service_location, self.embed_dim)
        self.att = attention(self.embed_dim, self.droprate, cuda=self.device)
        self.u_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.un_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.s_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.sn_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)


    def forward(self, nodes_u, nodes_s):

        if self.is_user == True:
            user_id_embedding = self.u_embedding_layer(nodes_u.t()[0]).to(self.device)
            user_location_embedding = self.u_location_embedding_layer(nodes_u.t()[1]).to(self.device)
            user_embedding = torch.cat([user_id_embedding, user_location_embedding], dim=1)
            interactions = [self.un[i] for i in nodes_u.t()[0]]
            interactions = torch.LongTensor(interactions).to(self.device)
            neighs_id_feature = self.u_embedding_layer(interactions).to(self.device)
            neighs_location_feature = self.u_location_embedding_layer(
                torch.LongTensor(get_neighbor_location(interactions, self.u2l)).to(self.device))
            neighs_feature = torch.cat([neighs_id_feature, neighs_location_feature], dim=2)
            att_w = self.att(neighs_feature, user_embedding.unsqueeze(1), self.num_neighbor)
            neighbor_feature = (neighs_feature * att_w).sum(1)
            combined = torch.cat((user_embedding, neighbor_feature), dim=1)
            cmp_embed_matrix = self.relu1(self.un_bn(self.layer1(combined).to(self.device)))
            user_embedding = self.relu2(self.u_bn(self.layer2(user_embedding).to(self.device)))
            final = user_embedding + cmp_embed_matrix
        else:
            service_id_embedding = self.s_embedding_layer(nodes_s.t()[0]).to(self.device)
            service_location_embedding = self.s_location_embedding_layer(nodes_s.t()[1]).to(self.device)
            service_embedding = torch.cat([service_id_embedding, service_location_embedding], dim=1)
            interactions = [self.sn[i] for i in nodes_u.t()[0]]
            interactions = torch.LongTensor(interactions).to(self.device)
            neighs_id_feature = self.s_embedding_layer(interactions).to(self.device)
            neighs_location_feature = self.s_location_embedding_layer(
                torch.LongTensor(get_neighbor_location(interactions, self.s2l)).to(self.device))
            neighs_feature = torch.cat([neighs_id_feature, neighs_location_feature], dim=2)
            att_w = self.att(neighs_feature, service_embedding.unsqueeze(1), self.num_neighbor)
            neighbor_feature = (neighs_feature * att_w).sum(1)
            combined = torch.cat((service_embedding, neighbor_feature), dim=1)
            cmp_embed_matrix = self.relu1(self.sn_bn(self.layer1(combined).to(self.device)))
            service_embedding = self.relu2(self.s_bn(self.layer2(service_embedding).to(self.device)))
            final = service_embedding + cmp_embed_matrix

        return final
