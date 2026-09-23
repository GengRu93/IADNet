import matplotlib.pyplot as plt
import numpy as np
f, ax = plt.subplots(figsize=(10, 10))
import scipy.io as scio
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import scipy.io as scio
import scipy.io
import matplotlib.pyplot as plt
import scipy.sparse as sp


N=12
W = np.ones([N, N])
for i in range(N):
    W[i, i] = 0

adj = sp.coo_matrix(W)
values = adj.data
edge_index = np.vstack((adj.row, adj.col))
senders_idx, receivers_idx = edge_index

s=np.zeros([N, N])
for sim in range(10):
    name='../code/E2.IADNet_benene/attention_coefs'+str(sim +1)+'.mat'
    coefs = scio.loadmat(name)
    W = coefs['attention_coefs']
    adj = np.zeros([N, N])
    for i in range(N):  # 4
        for j in range(N - 1):  # 4-1

            adj[senders_idx[i * (N - 1) + j], receivers_idx[i * (N - 1) + j]] = W[
            i * (N - 1) + j]

    s=s+(np.abs(adj)+np.abs(adj).T)/2


adj=s/10


s=adj



x = [0, 1, 2, 3, 4, 5,6,7,8,9,10,11]
y = [0, 1, 2, 3, 4, 5,6,7,8,9,10,11]
plt.xticks(x, ['1 (C)', '2 (C)',  '3 (C)', '4 (C)', '5 (C)','6 (C)', '7 (H)', '8 (H)', '9 (H)', '10 (H)', '11 (H)', '12 (H)'])
plt.yticks(y,  ['1 (C)', '2 (C)',  '3 (C)', '4 (C)', '5 (C)','6 (C)', '7 (H)', '8 (H)', '9 (H)', '10 (H)', '11 (H)', '12 (H)'])

plt.xticks(rotation=90)
plt.imshow(s,'hot_r') #'BuPu'
#plt.imshow((np.abs(adj)),'hot_r')
plt.colorbar()

plt.show()

s[s<0.35*np.max(np.max(s))]=0
print(0.35*np.max(np.max(s)))
#s[s>0]=1

x = [0, 1, 2, 3, 4, 5,6,7,8,9,10,11]
y = [0, 1, 2, 3, 4, 5,6,7,8,9,10,11]
plt.xticks(x, ['1 (C)', '2 (C)',  '3 (C)', '4 (C)', '5 (C)','6 (C)', '7 (H)', '8 (H)', '9 (H)', '10 (H)', '11 (H)', '12 (H)'])
plt.yticks(y,  ['1 (C)', '2 (C)',  '3 (C)', '4 (C)', '5 (C)','6 (C)', '7 (H)', '8 (H)', '9 (H)', '10 (H)', '11 (H)', '12 (H)'])

plt.xticks(rotation=90)
plt.imshow(s,'hot_r')

plt.colorbar()


plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
data_benzene = np.load('../data/data_benzene1/md17_benzene2017.npz')
position = data_benzene['R']
charge = data_benzene['z']
position=position[13]

for i in range(position.shape[0]):#(12, 3)))
    if charge[i]==6:
        c='gray'
    elif charge[i]==8:
        c = '#E64825'
    elif charge[i]==1:
        c = '#00BFFF'
    elif charge[i]==7:
        c = '#F49E39'
    ax.scatter(position[i][0], position[i][1], position[i][2], s=charge[i]*25, c=c, marker='o')

for i in range(position.shape[0]):
    for j in range(position.shape[0]):
        if s[i,j]!=0:
            start_point=position[i]
            end_point=position[j]
            ax.plot([start_point[0], end_point[0]],
                    [start_point[1], end_point[1]],
                    [start_point[2], end_point[2]], color='gray',linewidth='3')




ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))



plt.show()

print('Drawing completed')
