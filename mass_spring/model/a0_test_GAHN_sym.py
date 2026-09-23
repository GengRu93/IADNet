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

target = scio.loadmat('../data/targetm_test.mat')
dftarget_test = target['target']
input = scio.loadmat('../data/inputm_test.mat')
dfinput_test = input['input']

data_test = dfinput_test
label_test = dftarget_test



seed=32
random.seed(seed)
np.random.seed(seed)

from parameter_parser_get import get_args

from GAT_Network import GATnet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'
print(device)


if __name__ == "__main__":
    args = get_args()

    model_step1 = GATnet(args,device).to(device)
    # arrange data



    N = args.n_particle

    best =2000
    M = 6000

    tend = 30
    time = 30

    label1 = str(0) + '-gat.'
    best_step = best
    path1 = '{}/{}{}{}.tar'.format(args.save_dir_gat, args.name, label1, best_step)
    model_step1.load_state_dict(torch.load(path1, map_location=torch.device(device)))

    kwargs = {'rtol': 1e-12}
    t_eval = np.linspace(0, tend, M)


    data_testnp=data_test

    data_test = torch.tensor(data_test, requires_grad=True, dtype=torch.float32)
    label_test = torch.tensor(label_test, requires_grad=True, dtype=torch.float32)
    test_loss=[]
    pred_sol = np.zeros([data_test.shape[0],M,N*2])
    grouth_sol = np.zeros([data_test.shape[0],M,N*2])
    testflag=True
    for sim in range(data_test.shape[0]):
        print('sim', sim)

        dvt,loss2 = model_step1(data_test[sim].to(device),testflag)
        test_loss_ghnn=L2_loss(dvt,label_test[sim])

        test_loss.append(test_loss_ghnn.data.numpy())

        #print(data_testnp[sim].shape)
