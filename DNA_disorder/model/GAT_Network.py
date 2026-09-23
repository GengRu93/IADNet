import torch
import torch.nn as nn
from models_gat import GAT
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

    def forward(self, x,testflag):
        x = x.requires_grad_(True)
        p=x[:, self.args.n_particle:2 *self.args.n_particle].requires_grad_(True)
        q=x[:,0:self.args.n_particle].requires_grad_(True)

        with torch.enable_grad():

            u1 = q.unsqueeze(2)
            u2 = p.unsqueeze(2)

            h_z,loss1 = self.gat(u1,u2,testflag)


            h11=h_z.squeeze(2)

            dF = torch.autograd.grad(h11.sum(), x, create_graph=True)[0]

        dqp = dF @ self.M.t()

        return dqp,loss1


    def permutation_tensor(self, n):

        M = None

        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[:n // 2]])

        #print(M)
        return -M
