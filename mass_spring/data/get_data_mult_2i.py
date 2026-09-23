import torch
import numpy as np
import scipy.integrate
solver = scipy.integrate.solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from kg_equ import func,Hqp
from  symint import sym_euler_explicit
import scipy.io
torch.backends.cudnn.determinstic = True
import argparse
import math
# parameters
from parameter_parser_get import get_args
args = get_args()
seed=6
random.seed(seed)
np.random.seed(seed)
sample_train=50
samlpe_val=30
sample_test=20
sample=sample_train+samlpe_val+sample_test
N=args.n_particle

y0 = np.zeros([1, N * 2])
for k in range(N):
    y0[0, k] = (random.random()) * np.sin(2 * np.pi * k / (2 * (N - 1)))

state=y0
for i in range(sample-1):
    y0 = np.zeros([1, N * 2])
    for k in range(N):
        y0[0, k] = (random.random()) * np.sin(2 * np.pi * k / (2 * (N - 1)))
    state1=y0
    state = np.concatenate((state, state1), axis=0)



time=30
M=30000
index=np.arange(0,M,200)
print('a',index.shape[0])
t_eval = np.linspace(0, time, M)
#print(t_eval[index]) #0.2
flag = False
kwargs = {'rtol': 1e-12}
for i in range(sample_train):
    print('train', i)
    [q, p] = sym_euler_explicit(Hqp, t_eval, state[i])

    tval = t_eval[index]
    #print(tval)

    #q = sol['y'][0:N,index]
    #p = sol['y'][N:2*N,index]
    sol=qp=np.concatenate([q, p], 0 )
    #print(sol.shape) #(12, 30000)

    data1 = qp[:, 0:-1]
    data2 = qp[:,1:]
    sol=data2
    qp=sol[:,index]
    print('qp.shape',qp.shape)
    dxdt=(data2-data1)/0.001
    dxdt = dxdt[:, index]

    #fig = plt.figure(figsize=(10.0, 8.0), dpi=70)



#print(sol['y'].shape)#(64, 500)
    xval = qp #(64, 10000)

    if flag:
        x_input = np.concatenate([x_input, xval], 1)
        x_target = np.concatenate([x_target, dxdt], 1)
    else:
        x_input = xval
        x_target = dxdt
        flag = True



fig = plt.figure(figsize=(12, 4), facecolor='white')

print('Generating plots...')
pos = np.linspace(0, N - 1, num=N)
im_particle = []
    #
for i in range(50):
        # print('d',d)
    im_particle.append(plt.plot(pos, qp[0:N,i ], 'ro'))
print('done')
im_ani = animation.ArtistAnimation(fig, im_particle, interval=20, repeat_delay=3000, blit=True)
plt.show()

target_file = np.savetxt("target.csv", x_target.T, delimiter=',')
input_file = np.savetxt("input.csv", x_input.T, delimiter=',')

#A_file = np.savetxt("A.csv", A, delimiter=',')
#B_file = np.savetxt("B.csv", B, delimiter=',')
print('train.shape')
print(x_target.T.shape)




flag = False
for i in range(sample_train,sample_train+samlpe_val):
    print('val',i)
    [q, p] = sym_euler_explicit(Hqp, t_eval, state[i])

    tval = t_eval[index]
    #q = sol['y'][0:N,index]
    #p = sol['y'][N:2*N,index]
    sol=qp=np.concatenate([q, p], 0 )
    data1 = qp[:, 0:-1]
    data2 = qp[:, 1:]
    sol = data2
    qp = sol[:, index]
    print('qp.shape', qp.shape)
    dxdt = (data2 - data1) / 0.001
    dxdt=dxdt[:, index]
    if flag:
        x_input = np.concatenate([x_input, xval], 1)
        x_target = np.concatenate([x_target, dxdt], 1)
    else:
        x_input = xval
        x_target = dxdt
        flag = True






target_file = np.savetxt("target_val.csv", x_target.T, delimiter=',')
input_file = np.savetxt("input_val.csv", x_input.T, delimiter=',')
print('val.shape')
print(x_target.T.shape)



x_inputm =np.zeros([sample_test,index.shape[0],N*2])
x_targetm=np.zeros([sample_test,index.shape[0],N*2])

flag = False
k=0
for i in range(sample_train+samlpe_val,sample_train+samlpe_val+sample_test):
    print('test',i)
    [q, p] = sym_euler_explicit(Hqp, t_eval, state[i])

    tval = t_eval[index]
    #q = sol['y'][0:N,index]
    #p = sol['y'][N:2*N,index]
    sol=qp=np.concatenate([q, p], 0 )
    data1 = qp[:, 0:-1]
    data2 = qp[:, 1:]
    sol = data2
    qp = sol[:, index]
    print('qp.shape', qp.shape)
    dxdt = (data2 - data1) / 0.001
    dxdt = dxdt[:, index]

    x_inputm[k] = qp.T



#print(sol['y'].shape)#(64, 500)
    xval = qp #(64, 10000)

    x_targetm[k] = dxdt.T
    if flag:
        x_input = np.concatenate([x_input, xval], 1)
        x_target = np.concatenate([x_target, dxdt], 1)
    else:
        x_input = xval
        x_target = dxdt
        flag = True
    k=k+1



scipy.io.savemat('targetm_test.mat', mdict={'target': x_targetm})
scipy.io.savemat('inputm_test.mat', mdict={'input': x_inputm})




target_file = np.savetxt("target_test.csv", x_target.T, delimiter=',')
input_file = np.savetxt("input_test.csv", x_input.T, delimiter=',')
print('test.shape')
print(x_target.T.shape)