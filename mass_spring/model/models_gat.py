import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import scipy.io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models_mlp import MLP,msg, aggr

class GAT(nn.Module):
    def __init__(self,args):

        super(GAT, self).__init__()

        self.W = nn.Parameter(torch.empty(size=(args.n_particle, args.n_particle)))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.args=args


        self.mlp_edge = MLP(4, 100, 1, args.nonlinearity)


    def forward(self, q,p,testflag):
        q_diff=self.hij_diff(q)
        q_pro = self.hij_pro(q)

        p_diff = self.hij_diff(p)
        p_pro = self.hij_pro(p)



        e1 = torch.cat([q_diff,q_pro, p_diff,p_pro], dim=2)
        #print(e1.shape)




        hh =self.mlp_edge(e1)
        hh=hh.squeeze(2)
        hh=hh.view(hh.shape[0], self.args.n_particle,self.args.n_particle)

        A1 = self.W
        #A1 = F.softmax(attention, dim=1)

        scipy.io.savemat('attention_coefs.mat', mdict={'attention_coefs': A1.cpu().detach().numpy()})
        x1 = A1 * hh
        #print(x1.shape)


        x1 = torch.sum(x1, dim=2)
        #print(x1.shape) #torch.Size([150, 6])
        x1 = x1.unsqueeze(2)





        loss1=self.add_graph_loss(A1)

        return x1,loss1





    def hij_diff(self, Wh):
        N = Wh.size()[1]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)


        Wh_repeated_alternating = Wh.repeat(1,N, 1)

        all_combinations_matrix = Wh_repeated_in_chunks - Wh_repeated_alternating

        return all_combinations_matrix


    def hij_pro(self, Wh):
        N = Wh.size()[1]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)

        Wh_repeated_alternating = Wh.repeat(1,N, 1)


        all_combinations_matrix = Wh_repeated_in_chunks * Wh_repeated_alternating

        return all_combinations_matrix


    def add_graph_loss(self, out_adj):

        graph_loss = 0

        graph_loss += 0.05* torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        #print('graph_loss2',graph_loss )
        return graph_loss





