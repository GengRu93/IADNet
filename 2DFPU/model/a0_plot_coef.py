import matplotlib.pyplot as plt
import numpy as np
f, ax = plt.subplots(figsize=(10, 10))
import scipy.io as scio
coefs = scio.loadmat('attention_coefs.mat')
coefs = coefs['attention_coefs']


x = range(0,64,1)
y = range(0,64,1)
x1 = range(1,64+1,1)
for i in range(64):
    coefs[i][i]=0

plt.imshow((coefs), 'CMRmap_r')

plt.xticks(x, x1)
plt.yticks(y,  x1)
plt.colorbar()
plt.xticks(rotation=90)
plt.savefig("./IADNet_FPUT_weight_matrix.png")
plt.show()


s=np.abs(coefs)
coefs[s<0.35*np.max(np.abs(coefs))]=0

plt.imshow((coefs), 'CMRmap_r')

plt.xticks(x, x1)
plt.yticks(y,  x1)
plt.colorbar()
plt.xticks(rotation=90)
plt.savefig("./IADNet_FPUT_interaction_matrix.png")
plt.show()

print('The heat map of the interaction matrix has been completed and saved in PNG format in the folder!')