#py3.8

import time
import torch
import numpy as np
import pandas as pd
import os, sys

from GAT_Network import GATnet
import networkx as nx
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
import scipy.integrate
import scipy.sparse as sp
solve_ivp = scipy.integrate.solve_ivp
from utils import L2_loss
import utils
from parameter_parser_get import get_args
import glob
import scipy.integrate
import scipy.io as scio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




def train(args):
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)




    dftarget = pd.read_csv("../data/target.csv", header=None, dtype=np.float32)
    dfinput = pd.read_csv("../data/input.csv", header=None, dtype=np.float32)

    dftarget_val = pd.read_csv("../data/target_val.csv", header=None, dtype=np.float32)
    dfinput_val = pd.read_csv("../data/input_val.csv", header=None, dtype=np.float32)

    data = dfinput.values  # torch.Size([750, 2])
    print(data.shape[0] / 2)
    label = dftarget.values
    dxdt = torch.Tensor(label)

    data_val = dfinput_val.values  # torch.Size([750, 2])
    label_val = dftarget_val.values



    x = torch.tensor(data, requires_grad=True, dtype=torch.float32)


    no_batches = int(x.shape[0] / args.batch_size)


    starttime = time.time()
###############################################################################################################
    for sim in range(1):#x.shape[0]
        stats = {'train_loss': [], 'test_loss': []}
        loss_values = []

        best = args.total_steps1 + 1
        model_step1 = GATnet(args, device).to(device)


        print("Num. of params: {:d}".format(utils.get_parameters_count(model_step1)))

        parmas = list(model_step1.parameters())
        optim1 = torch.optim.Adam(parmas, 0.001)
        optim2 = torch.optim.Adam(parmas, 0.0001)
        optim3 = torch.optim.Adam(parmas, 0.00001)

        starttime = time.time()
        testflag=False

        for step in range(args.total_steps1 + 1):
            tt0 = time.time()
            train_loss_epoch_even = 0.0
            train_loss_ham = 0.0
            train_loss_graph1 = 0.0

            for batch in range(no_batches):
                ixs = torch.randperm(x.shape[0])[:args.batch_size]

                dvt_hat,loss2 = model_step1(x[ixs].to(device),testflag)




                loss = L2_loss(dxdt[ixs].to(device),dvt_hat)+loss2#+L2_loss(dxdt_hat[:,31],fix.to(device))+L2_loss(dxdt_hat[:,0],fix.to(device))+L2_loss(dxdt_hat[:,32],fix.to(device))+L2_loss(dxdt_hat[:,63],fix.to(device))

                loss_ham=L2_loss(dxdt[ixs].to(device),dvt_hat)
                loss_graph1=loss2



                loss.backward()
                if step < 1000:
                    optim1.step()
                    optim1.zero_grad()
                elif step < 1500:
                    optim2.step()
                    optim2.zero_grad()

                else:
                    optim3.step()
                    optim3.zero_grad()
                train_loss_epoch_even += loss.item()

                train_loss_ham += loss_ham.item()

                train_loss_graph1 += loss_graph1.item()



            # logging
            stats['train_loss'].append(train_loss_epoch_even / no_batches)
            loss_values.append(train_loss_epoch_even / no_batches)






            if args.verbose and step % args.print_every == 0:
                print("sim {},step {}, train_loss {:.4e}|hma_loss {:.4e}|graph1_loss {:.4e}| time: {:>7.12f}".format(sim, step,
                                                                                                  train_loss_epoch_even / no_batches, train_loss_ham/ no_batches, train_loss_graph1  / no_batches,
                                                                                                  (time.time() - tt0)))

        name_save_loss = './loss/GAT_' + str(sim) + '_loss'
        path_loss = '{}.mat'.format(name_save_loss)
        scipy.io.savemat(path_loss, mdict={'loass_values': loss_values})
        # save
        os.makedirs(args.save_dir_gat) if not os.path.exists(args.save_dir_gat) else None
        label1 = str(sim) + '-gat.'
        # label2 = '-hnn2.'
        path1 = '{}/{}{}{}.tar'.format(args.save_dir_gat, args.name, label1, step)
        # path2 = '{}/{}{}{}.tar'.format(args.save_dir, args.name1, label2, step)
        torch.save(model_step1.state_dict(), path1)

    print("Optimization Finished!")
    endtime = time.time()
    print("Optimization time：!")
    print(endtime - starttime)

    return model_step1


if __name__ == "__main__":
    args = get_args()
    model_step1 = train(args)

