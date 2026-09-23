import numpy as np
import matplotlib.pyplot as plt
from parameter_parser_get import get_args
args = get_args()
def sym_euler_explicit(Hqp,tspan,state0):
    N=args.n_particle
    #print(state0.shape)
    q0=state0[0:N]
    p0 = state0[N:2*N]


    #print(q0.shape[1])
    Nt = len(tspan)-1

    #print(Nt)

    hs = np.diff(tspan)
    #print(hs) #0.005
    #
    q = np.zeros([N,Nt+1])

    p = np.zeros([N,Nt+1])
    q[:,0]=q0
    p[:,0]=p0
    bb = np.array(
        [.0617588581356263250, .3389780265536433551, .6147913071755775662, -.1405480146593733802, .1250198227945261338,
         0])
    BB = np.array([0, 0.20517766154229, 0.40302128160421, -0.12092087633891, 0.51272193319241, 0])


    for i in range(Nt):
        h = hs[i]
        for j in range(5):

        #print(Hqp(q[:,i],p[:, i+1]).shape)
            q0 = q0 + h * BB[j]* Hqp(q0,p0)[0:N]
            p0 = p0+ h * bb[j] * Hqp(q0, p0)[N:2*N]
        #q[:, i + 1] = q[:, i] + h * BB[j] * Hqp(q[:, i + 1], p[:, i])[0:4]
        #p[:, i + 1] = p[:, i] + h * bb[j] * Hqp(q[:, i + 1], p[:, i])[4:8]


        q[:, i+1] = q0
        p[:, i+1] =p0

    #print(p.shape)
    return q,p


def sym_euler_explicit2(Hqp,tspan,state0):
    N=args.n_particle
    #print(state0.shape)
    q0=state0[0:N]
    p0 = state0[N:2*N]


    #print(q0.shape[1])
    Nt = len(tspan)-1

    #print(Nt)

    hs = np.diff(tspan)
    #print(hs) #0.005
    #
    q = np.zeros([N,Nt+1])

    p = np.zeros([N,Nt+1])
    q[:,0]=q0
    p[:,0]=p0
    bb = np.array(
        [.0617588581356263250, .3389780265536433551, .6147913071755775662, -.1405480146593733802, .1250198227945261338,
         0])
    BB = np.array([0, 0.20517766154229, 0.40302128160421, -0.12092087633891, 0.51272193319241, 0])


    for i in range(Nt):
        h = hs[i]
        for j in range(5):

        #print(Hqp(q[:,i],p[:, i+1]).shape)
            q0 = q0 + h * BB[j]* Hqp(state0)[0:N]
            state0 = np.concatenate([q0, p0], 0)
            p0 = p0+ h * bb[j] * Hqp(state0)[N:2*N]
            state0 = np.concatenate([q0, p0], 0)
        #q[:, i + 1] = q[:, i] + h * BB[j] * Hqp(q[:, i + 1], p[:, i])[0:4]
        #p[:, i + 1] = p[:, i] + h * bb[j] * Hqp(q[:, i + 1], p[:, i])[4:8]


        q[:, i+1] = q0
        p[:, i+1] =p0

    #print(p.shape)
    return q,p