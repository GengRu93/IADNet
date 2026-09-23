#python3.8
import torch
import numpy as np
import os, sys
import matplotlib.pyplot as plt
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
import scipy.sparse as sp
import networkx as nx
import scipy.io as scio
import matplotlib.animation as animation
import random
from utils import L2_loss


import scipy.integrate
solver = scipy.integrate.solve_ivp




seed=32
random.seed(seed)
np.random.seed(seed)

from parameter_parser_get import get_args

from MSPnet_Network import GATnet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'
print(device)




if __name__ == "__main__":
    args = get_args()

    model_step1 = GATnet(args,device).to(device)
    # arrange data
    data_benzene = np.load('../data/data_benzene1/md17_benzene2017.npz')
    print(data_benzene.files)
    train_num = 50000

    position = data_benzene['R']  # ((49863, 12, 3))
    position = position[0:train_num]
    ixs = torch.randperm(position.shape[0])
    position = position[ixs]

    # print(k.shape)
    label = data_benzene['E']
    label = label[0:train_num]
    label = label[ixs]
    charges = data_benzene['z']

    charges = torch.Tensor(charges)
    charges = (charges - torch.min(charges)) / (torch.max(charges) - torch.min(charges))
    charge_scale = len(charges)
    charge_power = 2


    m = np.array([12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 12, 12, 16, 1, 1, 1, 1, 1, 1, 1, 1])
    m = torch.Tensor(m)
    charges = (charges.unsqueeze(-1))
    m = (m.unsqueeze(-1))


    charges = charges.repeat(position.shape[0], 1, 1)  # [49863, 12, 3]
    m = m.repeat(position.shape[0], 1, 1)



    N_atom = position.shape[1]
    W = np.ones([N_atom, N_atom])
    for i in range(N_atom):
        W[i, i] = 0

    adj = sp.coo_matrix(W)
    values = adj.data
    edge_index = np.vstack((adj.row, adj.col))
    edge_index = torch.Tensor(edge_index).long()

    F = data_benzene['F']
    F = F[0:train_num]

    F = F[ixs]
    F1 = np.mean(F, axis=0, keepdims=True)  # F (1, 12, 3)


    F = torch.Tensor(F)

    #

    label_E = torch.Tensor(label)

    position = position[0:12000]

    position = (position - np.min(position)) / (np.max(position) - np.min(position))



    position = torch.tensor(position, requires_grad=True, dtype=torch.float32)


    position_train = position[0:10000]

    F_train = F[0:10000]
    F_train = (F_train - torch.mean(F_train)) / torch.std(F_train)

    m_train = m[0:10000]

    label_E_train = label_E[0:10000]
    charges_train = charges[0:10000]

    position_test = position[10000:12000]

    F_test = F[10000:12000]
    F_test = (F_test - torch.mean(F_train)) / torch.std(F_train)

    m_test = m[10000:12000]
    charges_test = charges[10000:12000]

    label_E_test = label_E[10000:12000]

    testflag = True
    best = np.array([183, 133, 139, 208, 190, 138, 155, 117, 135, 144])
    for i in range(10):
        label1 = str(i) + '-gat.'
        best_step = best[i]
        path1 = '{}/{}{}{}.tar'.format(args.save_dir_gat, args.name, label1, best_step)
        model_step1.load_state_dict(torch.load(path1, map_location=torch.device(device)))

        pred_E_test, loss2_test, H = model_step1(position_test, edge_index,
                                                 charges_test, testflag)

    print('Test completed!, run ‘a0_plot_coef’ to draw the interaction matrix!')


