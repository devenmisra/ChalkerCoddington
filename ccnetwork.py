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


# Extract Blocks from Block Diagonal Matrix

def extract_block_diag(A,M,k=0):
    """Extracts blocks of size M from the kth diagonal
    of square matrix A, whose size must be a multiple of M."""

    # Check that the matrix can be block divided
    if A.shape[0] != A.shape[1] or A.shape[0] % M != 0:
        raise StandardError('Matrix must be square and a multiple of block size')

    # Assign indices for offset from main diagonal
    if abs(k) > M - 1:
        raise StandardError('kth diagonal does not exist in matrix')
    elif k > 0:
        ro = 0
        co = abs(k)*M 
    elif k < 0:
        ro = abs(k)*M
        co = 0
    else:
        ro = 0
        co = 0

    blocks = np.array([A[i+ro:i+ro+M,i+co:i+co+M] for i in range(0,len(A)-abs(k)*M,M)])
    
    return blocks


# Node Insertions

def TransfMatInsert(MatrixList, m, nw, InsertProbability, InsertMatrix): 

    for i in range(nw):
        diagBlocks = extract_block_diag(MatrixList[i] ,2)
    
        for j in range(m):
            if np.random.rand(m)[j] > InsertProbability:
                diagBlocks[j] = InsertMatrix

        A = sp.linalg.block_diag(diagBlocks)
        MatrixList[i] = A

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


# Generate Lyapunov Exponents for a Single Value of Theta

def SingleLyap(ngen, n, m, w, Theta, seed):
    MatrixList = TransfMatGenerator(Theta, m, ngen*w, seed)
    nmax = min([ngen, n])
    
    return LyapFinder(w, MatrixList[0 : nmax])


# Generate Lyapunov Exponents for a List of Theta Values

def ListLyap(ngen, n, m, w, ThetaList, seed):
    nmax = min([ngen, n])
    LyapList = []
    for j in range(0, len(ThetaList)):
        MatrixList = TransfMatGenerator(ThetaList[j], m, ngen*w, seed)
        WholeList = LyapFinder(w, MatrixList[0:nmax])
        LyapList.append([ThetaList[j], -1/max([x for x in WholeList if x<0])])

    return np.array(LyapList)


# Testing

testList = ListLyap(100000, 100000, 16, 1, np.arange(0.1, 1.7, 0.1), 1)

# +
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

GaussianFit = curve_fit(gauss, testList[:,0], testList[:,1])[0]

plt.scatter(testList[:,0], testList[:,1], s=15);
plt.plot(np.arange(0.1,1.7,0.01), gauss(np.arange(0.1,1.7, 0.01), *GaussianFit));
plt.vlines(GaussianFit[2],min(testList[:,1]), max(testList[:,1]), color='red');
