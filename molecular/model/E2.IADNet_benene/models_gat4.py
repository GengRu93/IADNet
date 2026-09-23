import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import scipy.io
from torch_scatter import scatter_add
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models_mlp import MLPset,msg, MLP1,MLP
import torch.nn.functional as F
import scipy.sparse as sp
class GAT(nn.Module):
    def __init__(self,args):

        super(GAT, self).__init__()

        self.W1 = nn.Parameter(torch.empty(size=(args.n_particle * (args.n_particle - 1), 1)))

        nn.init.xavier_uniform_(self.W1.data, gain=1.414)

        self.args=args



        self.mlp_edge = MLP(3+3, 256, 1, 'silu')

        #self.mlp_t = MLP(1, 128, 1, 'silu')

        self.args.n_particle=args.n_particle


    def forward(self, position,edge_index,charges,testflag):

        senders_idx, receivers_idx = edge_index
        dist_p= (position[:, senders_idx] - position[:, receivers_idx])


        dot_product= (position[:, senders_idx]*position[:, receivers_idx])

        norm1=torch.norm( position[:, receivers_idx],dim=2,keepdim=True)
        norm2 = torch.norm(position[:, senders_idx], dim=2, keepdim=True)
        cos_angle3=dot_product/(norm1*norm2)



        e1 = torch.cat([dist_p,cos_angle3], dim=2) #,pro_charges






        hh =self.mlp_edge( e1)


        A1 = self.W1

        x1 = hh * A1

        x1 = scatter_add(x1, senders_idx, dim=1, dim_size=self.args.n_particle)


        h = x1#
        if testflag:


            scipy.io.savemat('attention_coefs10.mat', mdict={'attention_coefs': (A1).cpu().detach().numpy()})








        loss1=self.add_graph_loss(A1)


        return h,loss1



    def add_graph_loss(self, out_adj):

        graph_loss = 0

        graph_loss +=0.005*torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))

        return graph_loss

