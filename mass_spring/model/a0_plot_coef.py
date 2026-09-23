import matplotlib.pyplot as plt
import numpy as np
f, ax = plt.subplots(figsize=(10, 10))
import scipy.io as scio
coefs = scio.loadmat('attention_coefs.mat')
coefs = coefs['attention_coefs']


x = range(0,6,1)
y = range(0,6,1)
x1 = range(1,7,1)
for i in range(6):
    coefs[i][i]=0


plt.imshow(np.abs(coefs),'CMRmap_r')#'BuGn' #BuPu #GnBu  #Greys  Purples  YlGnBu  Blues CMRmap_r


plt.xticks(x, x1,fontsize=32)
plt.yticks(y,  x1,fontsize=32)

plt.colorbar()


plt.savefig("./IADNet_spring_weight_matrix.png")
plt.show()

s=np.abs(coefs)
coefs[s<0.35*np.max(np.abs(coefs))]=0


plt.imshow(np.abs(coefs),'CMRmap_r')#'BuGn' #BuPu #GnBu  #Greys  Purples  YlGnBu  Blues CMRmap_r


plt.xticks(x, x1,fontsize=32)
plt.yticks(y,  x1,fontsize=32)

plt.colorbar()


plt.savefig("./IADNet_spring_interaction_matrix.png")
plt.show()

print('The heat map of the interaction matrix has been completed and saved in PNG format in the folder!')