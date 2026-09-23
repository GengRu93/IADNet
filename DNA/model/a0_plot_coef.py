import matplotlib.pyplot as plt
import numpy as np
f, ax = plt.subplots(figsize=(10, 10))
import scipy.io as scio
coefs = scio.loadmat('attention_coefs.mat')
coefs = coefs['attention_coefs']

N=12
x = range(0,N,1)
y = range(0,N,1)
x1 = range(1,N+1,1)
for i in range(N):
    coefs[i][i]=0
plt.imshow(np.abs(coefs), 'CMRmap_r')
plt.xticks(x, x1,fontsize=32)
plt.yticks(y,  x1,fontsize=32)

plt.colorbar()


plt.savefig("./IADNet_DNA_weight_matrix.png")
plt.show()
s=np.abs(coefs)
coefs[s<0.35*np.max(np.abs(coefs))]=0
print(0.35*np.max(np.abs(coefs)))

plt.imshow(np.abs(coefs), 'CMRmap_r')
plt.xticks(x, x1,fontsize=32)
plt.yticks(y,  x1,fontsize=32)

plt.colorbar()


plt.savefig("./IADNet_DNA_interaction_matrix.png")
plt.show()
print('The heat map of the interaction matrix has been completed and saved in PNG format in the folder!')


