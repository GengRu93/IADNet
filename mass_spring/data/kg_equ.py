import torch
import numpy as np
from parameter_parser_get import get_args
args = get_args()
import torch
import numpy as np
from parameter_parser_get import get_args
args = get_args()

def func(t, y):
    dH = np.zeros_like(y)
    N=args.n_particle

    #y[32] =0
    #y[63]=0
    #y[0] =0
    #y[31]=0

    #dH[N]=y[1]-2*y[0]+y[N-1]+y[2]+y[N-2]-2*y[0]-y[0]- y[0]**3
    #dH[N+1] = y[2] - 2 * y[1] + y[0] + y[3] + y[N - 1] - 2 * y[1] - y[1] - y[1] ** 3

    #dH[2*N-1]=y[0]-2*y[N-1]+y[N-2]+y[1]+y[N-3]-2*y[N-1]-y[N-1]- y[N-1]**3
    #dH[2 * N -2] = y[N-1] - 2 * y[N - 2] + y[N - 3] + y[0] + y[N - 4] - 2 * y[N - 2] - y[N - 2] - y[N - 2] ** 3
    index = np.array(
        [ 5, 0, 1, 2, 3, 4, 5, 0])


    for i in range(1,N+1):

        dH[N+i-1]=0.5*(y[index[i+1]]+y[index[i-1]]-2*y[index[i]])

    dH[N +0]=dH[N +0]-0.3*(y[index[1]]-y[index[4]])-0.7*(y[index[1]]-y[index[3]])
    dH[N + 2]=dH[N + 2]+0.7*(y[index[1]]-y[index[3]])
    dH[N + 3]=dH[N + 3]+0.3*(y[index[1]]-y[index[4]])
    for i in range(0, N):
        dH[i]=y[N+i]


    return dH


def Hqp(valpair1,valpair2):
    (q1,q2,q3,q4,q5,q6) = tuple(valpair1)
    (p1,p2,p3,p4,p5,p6)  = tuple(valpair2)


    return np.array([p1,p2,p3,p4,p5,p6,
                     0.5*(q2+q6 - 2*q1) -0.3*(q1-q4)-0.7*(q1-q3),
                     0.5*(q3+q1 - 2*q2) ,
                     0.5*(q4+q2 - 2*q3) +0.7*(q1 -q3),
                     0.5*(q5+q3 - 2*q4) +0.3*(q1 - q4),
                     0.5*(q6+q4 - 2*q5) ,
                     0.5 * (q1 + q5 - 2 * q6),
                     ]).T



