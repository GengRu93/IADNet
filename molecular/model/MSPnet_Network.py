import torch
import torch.nn as nn
from models_gat4 import GAT
from models_mlp import MLP,msg, aggr


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GATnet(nn.Module):
    def __init__(self,args,device):

        super(GATnet, self).__init__()
        self.args=args
        self.setup_layers()
        self.M = self.permutation_tensor(self.args.n_particle*2).to(device)


    def setup_layers(self):


        self.gat = GAT(args=self.args)

    def forward(self, position,edge_index,charges,testflag):
        position = position.requires_grad_(True)



        with torch.enable_grad():



            h_z,loss1 = self.gat(position,edge_index,charges,testflag)

            h11 = h_z.squeeze(2)

            F = torch.autograd.grad(h11.sum(), position, create_graph=True)[0]



        return -F,loss1,h11.sum()


    def permutation_tensor(self, n):

        M = None

        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[:n // 2]])

        #print(M)
        return -M
