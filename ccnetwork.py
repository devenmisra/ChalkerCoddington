# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# A & B Matrices

# +
import numpy as np
import scipy as sp

def ATransferMatrixGenerator(S, T, m, diagValues):
    Phi = np.diag(diagValues)
    A = np.array([[S, -T], [-T, S]])
    diagBlocks = []
    for i in range(m): 
        diagBlocks.append(A)
    Turn = sp.linalg.block_diag(*diagBlocks)
    
    return Turn @ Phi

def BTransferMatrixGenerator(C_s, C_o, m, diagValues): 
    Phi = np.diag(diagValues)
    A = np.array([[C_s, C_o], [C_o, C_s]])
    diagBlocks = [np.array(A[1,1])]
    for i in range(m-1): 
        diagBlocks.append(A)
    diagBlocks.append(np.array(A[0,0]))
    Turn = sp.linalg.block_diag(*diagBlocks) + sp.sparse.coo_array(([A[1, 0], A[0,1]], [(0, 2*m-1), (2*m-1, 0)]))

    return Turn @ Phi


# -

# Transfer Matrix Generator

def TransfMatGenerator(Theta, m, nw, fixedseed): 
    S = 1/np.cos(Theta)
    T = np.tan(Theta)
    Cs = 1/np.sin(Theta)
    Co = np.cos(Theta)/np.sin(Theta)
    np.random.seed(seed=fixedseed)
    phases = np.exp(2*np.pi*1j*np.random.rand(nw, 2, 2*m))
    MatrixList = []
    
    for j in range(0, nw):
        AB = ATransferMatrixGenerator(S, T, m, phases[j,0]) @ BTransferMatrixGenerator(Cs, Co, m, phases[j,1])
        MatrixList.append(AB)
    
    return MatrixList


# Lyapunov Exponent Finder

def LyapFinder(w, MatrixList):
    m = len(MatrixList[0])
    n = len(MatrixList)//w
    L = np.eye(m)
    LyapList = np.zeros(m)
    
    for q in range(0, n-1):
        
        if len(MatrixList[q*w : (q+1)*w]) < 2: 
            H = MatrixList[q*w : (q+1)*w][0]
        else: 
            H = np.linalg.multi_dot(MatrixList[q*w : (q+1)*w])

        B = H @ L
        LU = sp.linalg.lu(B)[1:3]
        L = LU[0]
        LyapList = LyapList + np.log(np.abs(np.diagonal(LU[1])))

    if len(MatrixList[(n-1)*w : n*w]) < 2:
        H = MatrixList[(n-1)*w : n*w][0]
    else:
        H = np.linalg.multi_dot(MatrixList[(n-1)*w : n*w])

    B = H @ L
    LyapList = LyapList + np.log(np.abs(np.diagonal(sp.linalg.qr(B)[1])))

    return LyapList/len(MatrixList)


# Complete Function

def fullfunction(ngen, n, m, w, Theta, seed):
    MatrixList = TransfMatGenerator(Theta, m, ngen*w, seed)
    print("Matrices generated")
    nmax = min([ngen, n])
    
    return LyapFinder(w, MatrixList[0 : nmax])


# Testing

fullfunction(1000, 1000, 8, 1, np.pi/4 + 0.005, 1)

fullfunction(10000, 10000, 16, 1, np.pi/4 + 0.005, 42)


# Generate Lyapunov Exponents for List of Thetas

def ListLyap(ngen, n, m, w, ThetaList, seed):
    nmax = min([ngen, n])
    LyapList = []
    for j in range(0, len(ThetaList)):
        MatrixList = TransfMatGenerator(ThetaList[j], m, ngen*w, seed)
        WholeList = LyapFinder(w, MatrixList[0:nmax])
        LyapList.append([ThetaList[j], -1/max([x for x in WholeList if x<0])])
        
    return LyapList


# Testing

testlist = ListLyap(1000, 1000, 16, 1, np.arange(0.1, 1.7, 0.1), 1)

# +
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

GaussianFit = curve_fit(gauss, np.array(testlist)[:,0], np.array(testlist)[:,1])[0]

plt.scatter(np.array(testlist)[:,0], np.array(testlist)[:,1], s=15);
plt.plot(np.arange(0.1,1.7,0.01), gauss(np.arange(0.1,1.7, 0.01), *GaussianFit));
plt.vlines(GaussianFit[2],min(np.array(testlist)[:,1]), max(np.array(testlist)[:,1]), color='red')
