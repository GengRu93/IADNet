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

    Dn = 0.075
    an = 6.9
    K = 0.025
    pho = 2
    b = 0.35
    m = 1


    #index1 = np.array(
    #    [15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0])
    index1 = np.array(
        [5, 0, 1, 2, 3, 4, 5, 0])
    #index2 = np.array(
    #    [27,28,29,30,31,16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    #     28,
    #     29, 30, 31,16, 17, 18, 19, 20])
    index2 = np.array(
        [9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8])

    #index11 = np.array(
    #    [11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4])
    index11 = np.array(
        [3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2])

    K=1
    Dn=np.array([0.2,0.3,0.2,0.3,0.3,0.2])
    a=1


    for i in range(1,6+1):

        dH[12+i-1]=K*(y[index1[i+1]]+y[index1[i-1]]-2*y[index1[i]])+2*Dn[index1[i]]*a*(np.exp(-a*(y[index1[i]]-y[index2[2+i]]))-1)*np.exp(-a*(y[index1[i]]-y[index2[2+i]]))-K*(2*y[index1[i]]-y[index2[2+i+3]]-y[index2[2+i-3]])#

    for i in range(1, 6 + 1):

        dH[18 + i - 1] =K * (y[index2[2+i + 1]] + y[index2[2+i - 1]] - 2 * y[index2[2+i]]) - 2 *Dn[index1[i]] * a*(
        np.exp(-a * (y[index1[i]] - y[index2[2 + i]])) - 1) * np.exp(-a * (y[index1[i]] - y[index2[2 + i]])) +K*(y[index11[2+i+3]]+y[index11[2+i-3]]-2*y[index2[2+i]]) #
    for i in range(0, 12):
        dH[i]=y[12+i]


    return dH




def Hqp(valpair1,valpair2):
    K = 1
    Dn = np.array([0.2, 0.3, 0.2, 0.3, 0.3, 0.2])
    a = 1
    (q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12) = tuple(valpair1)
    (p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12)  = tuple(valpair2)


    return np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,
                     K * (q2 + q6 -2 * q1) + 2 * Dn[0] * a * (np.exp(-a * (q1 - q7)) - 1) * np.exp(-a * (q1 - q7)) - K * (2 * q1 - q10 - q10),
                     K * (q3 + q1 - 2 * q2) + 2 * Dn[1] * a * (np.exp(-a * (q2 - q8)) - 1) * np.exp(-a * (q2 - q8)) - K * (2 * q2 - q11 - q11),
                     K * (q4 + q2 - 2 * q3) + 2 * Dn[2] * a * (np.exp(-a * (q3 - q9)) - 1) * np.exp(-a * (q3 - q9)) - K * (2 * q3 - q12 - q12),
                     K * (q5 + q3 - 2 * q4) + 2 * Dn[3] * a * (np.exp(-a * (q4 - q10)) - 1) * np.exp(-a * (q4 - q10)) - K * (2 * q4 - q7 - q7),
                     K * (q6 + q4 - 2 * q5) + 2 * Dn[4] * a * (np.exp(-a * (q5 - q11)) - 1) * np.exp(-a * (q5 - q11)) - K * (2 * q5 - q8 - q8),
                     K * (q1 + q5 - 2 * q6) + 2 * Dn[5] * a * (np.exp(-a * (q6 - q12)) - 1) * np.exp(-a * (q6 - q12)) - K * (2 * q6 - q9 - q9),

                     K * (q8 + q12 - 2 * q7) - 2 * Dn[0] * a * (np.exp(-a * (q1 - q7)) - 1) * np.exp( -a * (q1 - q7)) + K *  ( q4 + q4-2 * q7),
                     K * (q9 + q7 - 2 * q8) - 2 * Dn[1] * a * (np.exp(-a * (q2 - q8)) - 1) * np.exp( -a * (q2 - q8)) + K * (q5 + q5 - 2 * q8),
                     K * (q10 + q8 - 2 * q9) - 2 * Dn[2] * a * (np.exp(-a * (q3 - q9)) - 1) * np.exp( -a * (q3 - q9)) + K * (q6+ q6 - 2 * q9),
                     K * (q11 + q9 - 2 * q10) - 2 * Dn[3] * a * (np.exp(-a * (q4 - q10)) - 1) * np.exp( -a * (q4 - q10)) + K * (q1 + q1 - 2 * q10),
                     K * (q12 + q10 - 2 * q11) - 2 * Dn[4] * a * (np.exp(-a * (q5 - q11)) - 1) * np.exp( -a * (q5 - q11)) + K * (q2 + q2 - 2 * q11),
                     K * (q7 + q11 - 2 * q12) - 2 * Dn[5] * a * (np.exp(-a * (q6 - q12)) - 1) * np.exp( -a * (q6 - q12)) + K * (q3 + q3 - 2 * q12),
                          ]).T




