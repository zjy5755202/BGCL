# -*- coding: utf-8 -*-
import os
import torch
import pickle
import argparse
import numpy as np
import torch.utils.data
import thop

from bgcl import BGCL
from test import test
from train import train
from utils.encoder import encoder
from utils.combiner import combiner


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='BGCL')
    parser.add_argument('--epochs', type=int, default=200,
                        metavar='N', help='number of epochs to train')
    parser.add_argument('--test_epochs', type=int, default=20,
                        metavar='N', help='number of epochs to test')
    parser.add_argument('--lr', type=float, default=0.001,
                        metavar='FLOAT', help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        metavar='STRING', help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='FLOAT', help='momentum')
    parser.add_argument('--embed_dim', type=int, default=128,
                        metavar='N', help='embedding dimension')
    parser.add_argument('--droprate', type=float, default=0.5,
                        metavar='FLOAT', help='dropout rate')
    parser.add_argument('--temp', type=float, default=0.1,
                        metavar='FLOAT', help='temp')
    parser.add_argument('--batch_size', type=int, default=64,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        metavar='N', help='input batch size for testing')
    parser.add_argument('--n_size', type=int, default=6,
                        metavar='N', help='neighbor_size')
    parser.add_argument('--dataset', type=str, default='wsdream',
                        metavar='STRING', help='dataset')
    parser.add_argument('--density', type=str, default='0.005',
                        metavar='STRING', help='density')
    parser.add_argument('--cold_start_density', type=str, default='0.500',
                        metavar='STRING', help='cold_start_density')
    parser.add_argument('--p', type=str, default='0.8',
                        metavar='STRING', help='p')
    parser.add_argument('--a', type=str, default='0.7',
                        metavar='STRING', help='a')
    parser.add_argument('--mode', type=str, default='sparsity',
                        metavar='STRING', help='mode')
    parser.add_argument('--datatype', type=str, default='tp',
                        metavar='STRING', help='datatype')
    parser.add_argument('--coldStartType', type=str, default='user',
                        metavar='STRING', help='coldStartType')

    args = parser.parse_args()

    print('Dataset: ' + args.dataset)
    print('Density: ' + args.density)
    print('-------------------- Hyperparams --------------------')
    print('dropout rate: ' + str(args.droprate))
    print('dimension of embedding: ' + str(args.embed_dim))
    print('type of optimizer: ' + str(args.optimizer))
    print('learning rate: ' + str(args.lr))
    print('p: ' + str(args.p))
    print('a: ' + str(args.a))
    print('datatype: ' + str(args.datatype))
    print('mode: ' + str(args.mode))
    print('coldStartType: ' + str(args.coldStartType))
    print('cold_start_density: ' + str(args.cold_start_density))
    if args.optimizer == 'SGD':
        print('momentum: ' + str(args.momentum))

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    data_path = './datasets/%s/%s/%s' % (args.dataset, args.mode, args.datatype)
    if args.mode == 'sparsity':
        all_data_path = data_path + '/' + args.datatype + '_allData_' + args.p + '_' + args.a + '_' + args.density + '.p'
    else:
        all_data_path = data_path + '/' + args.datatype + '_' + args.coldStartType + '_allData_' + args.p + '_' + args.a + '_' + args.cold_start_density + '_' + args.density + '.p'

    with open(all_data_path, 'rb') as meta:
        u2le, i2le, u2c, u2l, i2c, i2l, un1, sn1, un2, sn2, u_train, i_train, r_train, u_test, i_test, r_test, u_adj_1, i_adj_1, u_adj_2, i_adj_2 = pickle.load(
            meta)

    un1 = un1[:, :args.n_size]
    un2 = un2[:, :args.n_size]
    sn1 = sn1[:, :args.n_size]
    sn2 = sn2[:, :args.n_size]

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(u_train), torch.LongTensor(i_train),
                                              torch.FloatTensor(r_train))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(u_test), torch.LongTensor(i_test),
                                             torch.FloatTensor(r_test))

    _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=8, pin_memory=True)
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True,
                                        num_workers=8, pin_memory=True)

    u_embed_cmp1 = encoder(embed_dim, un1, sn1, u2l, i2l, args.droprate, args.n_size, cuda=device)

    u_embed_cmp2 = encoder(embed_dim, un2, sn2, u2l, i2l, args.droprate, args.n_size, cuda=device)

    u_embed = combiner(u_embed_cmp1, u_embed_cmp2, embed_dim, cuda=device)

    s_embed_cmp1 = encoder(embed_dim, un1, sn1, u2l, i2l, args.droprate, args.n_size, cuda=device, is_user_part=False)

    s_embed_cmp2 = encoder(embed_dim, un2, sn2, u2l, i2l, args.droprate, args.n_size, cuda=device, is_user_part=False)

    s_embed = combiner(s_embed_cmp1, s_embed_cmp2, embed_dim, cuda=device)

    # model
    model = BGCL(u_embed, s_embed, embed_dim, temp=args.temp, droprate=args.droprate).to(device)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rmse_mn = np.inf
    mae_mn = np.inf
    endure_count = 0

    #
    # nodes_i = torch.zeros(1, 2).int()
    # nodes_u = torch.zeros(1, 2).int()
    # flops, params = thop.profile(model, inputs=(nodes_u,nodes_i))  # 计算
    # print(flops)
    # print(params)

    for epoch in range(1, args.epochs + 1):
        # ====================   training    ====================
        train(model, _train, optimizer, epoch, rmse_mn, mae_mn, device)
        # ====================     test       ====================
        if epoch % args.test_epochs == 0:
            rmse, mae = test(model, _test, device)

            if rmse_mn > rmse:
                rmse_mn = rmse
                mae_mn = mae
                endure_count = 0
            else:
                endure_count += 1

            print("<Test> RMSE: %.5f, MAE: %.5f " % (rmse, mae))

        if endure_count > 30:
            break

    print('The best RMSE/MAE: %.5f / %.5f' % (rmse_mn, mae_mn))


if __name__ == "__main__":
    main()
