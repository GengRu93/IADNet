
import torch
import numpy as np
from utils import choose_nonlinearity
from parameter_parser_get import get_args
args = get_args()
class MLP1(torch.nn.Module):

  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super( MLP1, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, output_dim)


    for l in [self.linear1]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    x1 = self.nonlinearity( self.linear1(x) )

    return x1

class MLPset(torch.nn.Module):

  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super( MLPset, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    x1 = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(x1) )
    return self.linear3(h+x1)

class MLP(torch.nn.Module):

  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super( MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    return self.linear3(h)


# Create model
#msg_net = nn.Sequential(
#    nn.Linear(args.n_dim*2, args.hs), nn.Tanh(),   #(2,60)
#    nn.Linear(args.hs, args.hs), nn.Tanh(),  #(60,60)
#    nn.Linear(args.hs, args.hs), nn.Tanh(),   #(60,60)
#    nn.Linear(args.hs, args.d)    #(60,40)
#)
#aggr_net = nn.Sequential(
#    nn.Linear(args.d+args.n_dim, args.hs), nn.Tanh(), #(41,60)
#    nn.Linear(args.hs, args.hs), nn.Tanh(),  #(60,60)
#    nn.Linear(args.hs, args.hs), nn.Tanh(),  #(60,60)
#    nn.Linear(args.hs, 1)  #(60,1)
#)


class msg(torch.nn.Module):

  def __init__(self, n_dim, gnn_hs, d, nonlinearity='tanh'):
    super(msg, self).__init__()
    self.linear1 = torch.nn.Linear(n_dim, gnn_hs)
    self.linear2 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear3 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear4 = torch.nn.Linear(gnn_hs, d, bias=None)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    h = self.nonlinearity(self.linear3(h))
    return self.linear4(h)



class aggr(torch.nn.Module):

  def __init__(self, n_dim, gnn_hs, d, nonlinearity='tanh'):
    super(aggr, self).__init__()
    self.linear1 = torch.nn.Linear(n_dim, gnn_hs)
    self.linear2 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear3 = torch.nn.Linear(gnn_hs, d, bias=None)

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    return self.linear3(h)


class MLP6(torch.nn.Module):

  def __init__(self, input_dim, gnn_hs, output_dim, nonlinearity='tanh'):
    super(MLP6, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, gnn_hs)
    self.linear2 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear3 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear4 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear5 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear6 = torch.nn.Linear(gnn_hs, 1, bias=None)

    for l in [self.linear1, self.linear2, self.linear3,self.linear4, self.linear5, self.linear6]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    h = self.nonlinearity(self.linear3(h))
    h = self.nonlinearity(self.linear4(h))
    h = self.nonlinearity(self.linear5(h))

    return self.linear6(h)




class MLP11(torch.nn.Module):

  def __init__(self, input_dim, gnn_hs, output_dim, nonlinearity='tanh'):
    super(MLP11, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, gnn_hs)
    self.linear2 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear3 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear4 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear5 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear6 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear7 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear8 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear9 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear10 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear11 = torch.nn.Linear(gnn_hs, 1, bias=None)

    for l in [self.linear1, self.linear2, self.linear3,self.linear4, self.linear5, self.linear6,self.linear7, self.linear8, self.linear9, self.linear10, self.linear11]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    h = self.nonlinearity(self.linear3(h))
    h = self.nonlinearity(self.linear4(h))
    h = self.nonlinearity(self.linear5(h))
    h = self.nonlinearity(self.linear6(h))
    h = self.nonlinearity(self.linear7(h))
    h = self.nonlinearity(self.linear8(h))
    h = self.nonlinearity( self.linear9(h) )
    h = self.nonlinearity(self.linear10(h))


    return self.linear11(h)