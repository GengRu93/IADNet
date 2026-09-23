#py3.8

import time
import torch
import numpy as np
import pandas as pd
import os, sys

from MSPnet_Network import GATnet
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
from sklearn.preprocessing import robust_scale
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




def train(args):
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_benzene = np.load('../data/data_benzene1/md17_benzene2017.npz')
    print(data_benzene.files)
    train_num0 = 10000
    train_num = 60000



    position = data_benzene['R']
    print(position.shape)
    position = position[train_num0 :train_num]
    ixs = torch.randperm(position.shape[0])
    position=position[ixs]

    #print(k.shape)
    label =data_benzene['E']
    label = label[train_num0 :train_num]
    label = label[ixs]
    charges = data_benzene['z']

    charges = torch.Tensor(charges)
    charges = (charges - torch.min(charges)) / ( torch.max(charges) - torch.min(charges))


    charges= (charges.unsqueeze(-1) )


    charges = charges.repeat(position.shape[0], 1, 1)





    N_atom=position.shape[1]
    W=np.ones([N_atom,N_atom])
    for i in range(N_atom):
        W[i,i]=0



    adj = sp.coo_matrix(W)
    values = adj.data
    edge_index = np.vstack((adj.row, adj.col))
    edge_index = torch.Tensor(edge_index).long()



    F=data_benzene['F']
    F = F[train_num0 :train_num]




    F = F[ixs]






    F = torch.Tensor(F)

#



    label_E = torch.Tensor(label)
    #print(label)
    position = position[0:12000]

    position = (position - np.min(position )) /(np.max(position )-np.min(position ))




    position = torch.tensor(position, requires_grad=True, dtype=torch.float32)


    position_train=position[0:10000]
    #
    F_train=F[0:10000]
    F_train = (F_train - torch.mean(F_train)) / torch.std(F_train)




    label_E_train=label_E[0:10000]
    charges_train=charges[0:10000]


    position_test = position[10000:12000]

    F_test = F[10000:12000]
    F_test = (F_test - torch.mean(F_train)) / torch.std(F_train)


    charges_test = charges[10000:12000]


    label_E_test = label_E[10000:12000]


    no_batches = int(position_train.shape[0] / args.batch_size)
    L1_loss=torch.nn.L1Loss()


    starttime = time.time()
    testflag = False
###############################################################################################################
    for sim in range(10):#x.shape[0]
        stats = {'train_loss': [], 'test_loss': []}
        loss_values = []
        loss_values_val = []

        best = args.total_steps1 + 1
        model_step1 = GATnet(args, device).to(device)


        print("Num. of params: {:d}".format(utils.get_parameters_count(model_step1)))

        parmas = list(model_step1.parameters())
        optim1 = torch.optim.Adam(parmas, 0.001)
        optim2 = torch.optim.Adam(parmas, 0.0001)
        optim3 = torch.optim.Adam(parmas, 0.00001)

        starttime = time.time()

        for step in range(args.total_steps1 + 1):
            tt0 = time.time()
            train_loss_epoch_even = 0.0
            train_loss_ham = 0.0
            test_loss_ham = 0.0
            train_loss_graph1 = 0.0

            for batch in range(no_batches):
                ixs = torch.randperm(position_train.shape[0])[:args.batch_size]

                pred_E,loss2,H = model_step1(position_train[ixs].to(device),edge_index.to(device),charges_train[ixs].to(device),testflag)
                #print(H)

                label_E_train=(label_E_train /torch.max(label_E_train)) #/ torch.std(label_E_train)




                loss = L2_loss(F_train[ixs].to(device),pred_E)+loss2#+0.0005*L2_loss(label_E_train[ixs].to(device),H)

                loss_ham=L2_loss(F_train[ixs].to(device),pred_E)  #L2_loss
                loss_graph1=loss2



                loss.backward()
                if step < 2500:
                    optim1.step()
                    optim1.zero_grad()
                elif step < 5000:
                    optim2.step()
                    optim2.zero_grad()

                else:
                    optim3.step()
                    optim3.zero_grad()
                train_loss_epoch_even += loss.item()
                #train_loss_epoch_even += loss.item()

                train_loss_ham += loss_ham.item()

                train_loss_graph1 += loss_graph1.item()



            # logging
            stats['train_loss'].append(train_loss_epoch_even / no_batches)
            loss_values.append(train_loss_epoch_even / no_batches)

            model_step1.eval()
            pred_E_test, loss2_test,H = model_step1(position_test.to(device), edge_index.to(device),
                                        charges_test.to(device), testflag)
            #pred_E_test=pred_E_test*std_E+mean_E
            #label_E_test=(label_E_test - mean_E) / std_E

            acc_val = L2_loss(F_test.to(device),
                              pred_E_test)  # +L2_loss(dxdt_hat[:,31],fix.to(device))+L2_loss(dxdt_hat[:,0],fix.to(device))+L2_loss(dxdt_hat[:,32],fix.to(device))+L2_loss(dxdt_hat[:,63],fix.to(device))
            loss_values_val.append(acc_val.item())
            # save
            os.makedirs(args.save_dir_gat) if not os.path.exists(args.save_dir_gat) else None
            label1 = str(sim) + '-gat.'
            # label2 = '-hnn2.'
            path1 = '{}/{}{}{}.tar'.format(args.save_dir_gat, args.name, label1, step)
            # path2 = '{}/{}{}{}.tar'.format(args.save_dir, args.name1, label2, step)
            torch.save(model_step1.state_dict(), path1)

            if loss_values_val[-1] < best:
                best = loss_values_val[-1]
                best_step = step
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience1:
                break

            files = glob.glob(args.save_dir_gat + '/*.tar')
            # print(files)

            for file in files:

                epoch_nb = int(file.split('.')[4])

                labell = file.split('.')[3]
                # print(labell)
                labell = labell.split('/')[2]
                # print(labell)
                if labell == 'Train' + str(sim) + '-gat':
                    # print(labell)
                    if epoch_nb < best_step:
                        os.remove(file)

            if args.verbose and step % args.print_every == 0:
                print("sim {},step {}, train_loss {:.4e},Ham_loss {:.4e},graph_loss {:.4e},acc_loss {:.4e}| time: {:>7.12f}".format(sim, step,
                                                                                                  train_loss_epoch_even / no_batches, train_loss_ham / no_batches,loss_graph1/ no_batches,
                                                                                                  acc_val,
                                                                                                  (time.time() - tt0)))
        files = glob.glob(args.save_dir_gat + '/*.tar')
        # print(files)
        for file in files:
            epoch_nb = int(file.split('.')[4])
            labell = file.split('.')[3]
            labell = labell.split('/')[2]
            if labell == 'Train' + str(sim) + '-gat':

                if epoch_nb > best_step:
                    os.remove(file)
        name_save_loss = '../code/E2.IADNet_benene/loss/GAT_' + str(sim) + '_loss'
        path_loss = '{}.mat'.format(name_save_loss)
        scipy.io.savemat(path_loss, mdict={'loass_values': loss_values})

    return model_step1


if __name__ == "__main__":
    args = get_args()
    model_step1 = train(args)

