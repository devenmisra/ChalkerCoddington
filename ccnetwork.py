# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os

#OpenBLAS Optimization for TR3970X
os.environ["OPENBLAS_CORETYPE"] = "Zen"
os.environ["OPENBLAS_NUM_THREADS"] = '64'

#MKL Optimization for TR3970X
#os.environ["LD_PRELOAD"] = "~/libfakeintel.so"
#os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"
# -

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
    Turn = sp.linalg.block_diag(*diagBlocks) + sp.sparse.coo_array(([A[1,0], A[0,1]], [(0, 2*m-1), (2*m-1, 0)]))

    return Turn @ Phi


# -

# Transfer Matrix Generator

def TransfMatGenerator(Theta, m, nw, seed): 
    S = 1/np.cos(Theta)
    T = np.tan(Theta)
    Cs = 1/np.sin(Theta)
    Co = np.cos(Theta)/np.sin(Theta)
    np.random.seed(seed)
    phases = np.exp(2*np.pi*1j*np.random.rand(nw, 2, 2*m))
    MatrixList = []
    
    for j in range(0, nw):

        AB = ATransferMatrixGenerator(S, T, m, phases[j,0]) @ BTransferMatrixGenerator(Cs, Co, m, phases[j,1])
        MatrixList.append(AB)
    
    return MatrixList


# Transfer Matrix Generator (w/ One-Loop Insertions)

# +
def extract_block_diag_A(A,M):
    
    blocks = np.array([A[i:i+M,i:i+M] for i in range(0,len(A),M)])
    
    return blocks

def extract_block_diag_B(A,M,m):

    edge = np.array([[A[0,0], A[0,2*m-1]], [A[2*m-1, 0], A[2*m-1, 2*m-1]]])
    blocks = np.array([A[i:i+M,i:i+M] for i in range(1,len(A)-1,M)])

    return np.stack([edge, *blocks])

def RMatrixGenerator(MatrixType, ATransfMatList, BTransfMatList, m): 
    randIndex = np.random.randint(0, m, size=10)

    if MatrixType == 'A':
        M = sp.linalg.block_diag(ATransfMatList[randIndex[0]], ATransfMatList[randIndex[1]]) @ sp.linalg.block_diag(1, BTransfMatList[randIndex[2]], 1) @ sp.linalg.block_diag(ATransfMatList[randIndex[3]], ATransfMatList[randIndex[4]])

    if MatrixType == 'B': 
        M = sp.linalg.block_diag(BTransfMatList[randIndex[5]], BTransfMatList[randIndex[6]]) @ sp.linalg.block_diag(1, ATransfMatList[randIndex[7]], 1) @ sp.linalg.block_diag(BTransfMatList[randIndex[8]], BTransfMatList[randIndex[9]])
    
    R_00 = M[0,0] + (M[0,1] + M[0,2])*(M[2,0] - M[1,0]) / (M[1,1] + M[1,2] - M[2,1] - M[2,2])
    R_01 = M[0,3] + (M[0,1] + M[0,2])*(M[2,3] - M[1,3]) / (M[1,1] + M[1,2] - M[2,1] - M[2,2])
    R_10 = M[3,0] + (M[3,1] + M[3,2])*(M[2,0] - M[1,0]) / (M[1,1] + M[1,2] - M[2,1] - M[2,2])
    R_11 = M[3,3] + (M[3,1] + M[3,2])*(M[2,3] - M[1,3]) / (M[1,1] + M[1,2] - M[2,1] - M[2,2])

    RMatrix = np.array([[R_00, R_01], [R_10, R_11]])
    
    return RMatrix


# -

def TransfMatGenerator_withReplacement(Theta, m, nw, rngseed, insertProbability=0.0): 
    S = 1/np.cos(Theta)
    T = np.tan(Theta)
    Cs = 1/np.sin(Theta)
    Co = np.cos(Theta)/np.sin(Theta)
    rng = np.random.default_rng(seed = rngseed)
    phases = np.exp((np.pi/2) * 1j * rng.uniform(low=-1, high=1, size=(nw, 2, 2*m)))
    MatrixList = []
    
    for j in range(0, nw):
        A = ATransferMatrixGenerator(S, T, m, phases[j,0])
        diagBlocksA = extract_block_diag_A(A, 2)

        B = BTransferMatrixGenerator(Cs, Co, m, phases[j,1])
        diagBlocksB = extract_block_diag_B(B, 2, m)

        for i in range(m): 
            if rng.uniform(low=0, high=1, size=2*m)[i] < insertProbability:
                diagBlocksA[i] = RMatrixGenerator('A', diagBlocksA, diagBlocksB, m)

            if rng.uniform(low=0, high=1, size=2*m)[m+i] < insertProbability:
                diagBlocksB[i] = RMatrixGenerator('B', diagBlocksA, diagBlocksB, m)

        A_R = sp.linalg.block_diag(*diagBlocksA)
        B_R = sp.linalg.block_diag(*[diagBlocksB[0][0,0], *diagBlocksB[1:], diagBlocksB[0][1,1]]) +  sp.sparse.coo_array(([diagBlocksB[0][0,1], diagBlocksB[0][1,0]], [(0, 2*m-1), (2*m-1, 0)]))

        AB_R = A_R @ B_R

        MatrixList.append(AB_R)
    
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
        LU = sp.linalg.lu(B, permute_l=True)
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
    LyapRange = []
    for j in range(0, len(ThetaList)):
        MatrixList = TransfMatGenerator_withReplacement(ThetaList[j], m, ngen*w, seed)
        WholeList = LyapFinder(w, MatrixList[0:nmax])
        LyapRange.append([ThetaList[j], -1/max([x for x in WholeList if x<0])])

    return np.array(LyapRange)


# Extra Functions for Batch Processing

def BatchList(ngen, n, m, w, ThetaList, seed, insertProbability):
    nmax = min([ngen, n])
    WholeList = []
    for j in range(0, len(ThetaList)):
        MatrixList = TransfMatGenerator_withReplacement(ThetaList[j], m, ngen*w, seed, insertProbability)
        WholeList.append(LyapFinder(w, MatrixList[0:nmax]))

    return np.array(WholeList)


def LyapListPairs(WholeList, ThetaList):
    LyapRange = []
    for j in range(0, len(ThetaList)):
        LyapRange.append([ThetaList[j], -1/max([x for x in WholeList[j] if x<0])])

    return np.array(LyapRange)


# Testing

import pickle
import time

lengths = [1000, 1000, 1000, 1000, 1000, 1000]
widths = [20, 40, 60, 80, 100, 120]
critVal = np.pi/4
thetaRange = np.linspace(critVal-0.4, critVal+0.4, 17)

# +
iP = 0.0

for nbatch in range(0,10): 

    print(f"Processing Batch {nbatch}")

    testList = dict()

    for length, width in zip(lengths, widths): 

        start = time.process_time()
    
        testList[f'{width}'] = BatchList(length, length, width, 1, thetaRange, seed=nbatch, insertProbability=iP)

        with open(f'batchLyapDataP{iP}/batchLyapDict{nbatch}.pickle', 'wb') as handle:
            pickle.dump(testList, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'Completed M = {width} | CPU Time: {time.process_time() - start}')

# +
with open(f'batchLyapDataP{iP}/batchLyapDict0.pickle', 'rb') as handle:
    WholeList = pickle.load(handle)

for nbatch in range(1,10):

    with open(f'batchLyapDataP{iP}/batchLyapDict{nbatch}.pickle', 'rb') as handle:
        batchLyapDict = pickle.load(handle)
        
    for width in widths: 

        WholeList[f'{width}'] = WholeList[f'{width}'] + batchLyapDict[f'{width}']

for width in widths: 
    
    testList[f'{width}'] = LyapListPairs(WholeList[f'{width}']/10, thetaRange)

# +
completeList = dict()

with open(f'batchLyapDataP{iP}/batchLyapDict0.pickle', 'rb') as handle:
    aggList = pickle.load(handle)

for nbatch in range(1,10): 

    with open(f'batchLyapDataP{iP}/batchLyapDict{nbatch}.pickle', 'rb') as handle:
        batchLyapDict = pickle.load(handle)

    for width in widths: 
        
        aggList[f'{width}'] = aggList[f'{width}'] + batchLyapDict[f'{width}']

for width in widths: 
    
    completeList[f'{width}'] = LyapListPairs(aggList[f'{width}']/10, thetaRange)

with open(f'completeLyapDataP{iP}.pickle', 'wb') as handle: 
    pickle.dump(completeList, handle, protocol=pickle.HIGHEST_PROTOCOL)

# +
iP = 0.5

for nbatch in range(0,10): 

    print(f"Processing Batch {nbatch}")

    testList = dict()

    for length, width in zip(lengths, widths): 

        start = time.process_time()
    
        testList[f'{width}'] = BatchList(length, length, width, 1, thetaRange, seed=nbatch, insertProbability=iP)

        with open(f'batchLyapDataP{iP}/batchLyapDict{nbatch}.pickle', 'wb') as handle:
            pickle.dump(testList, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'Completed M = {width} | CPU Time: {time.process_time() - start}')

# +
with open(f'batchLyapDataP{iP}/batchLyapDict0.pickle', 'rb') as handle:
    WholeList = pickle.load(handle)

for nbatch in range(1,5):

    with open(f'batchLyapDataP{iP}/batchLyapDict{nbatch}.pickle', 'rb') as handle:
        batchLyapDict = pickle.load(handle)
        
    for width in widths: 

        WholeList[f'{width}'] = WholeList[f'{width}'] + batchLyapDict[f'{width}']

for width in widths: 
    
    testList[f'{width}'] = LyapListPairs(WholeList[f'{width}']/5, thetaRange)

# +
completeList = dict()

with open(f'batchLyapDataP{iP}/batchLyapDict0.pickle', 'rb') as handle:
    aggList = pickle.load(handle)

for nbatch in range(1,5): 

    with open(f'batchLyapDataP{iP}/batchLyapDict{nbatch}.pickle', 'rb') as handle:
        batchLyapDict = pickle.load(handle)

    for width in widths: 
        
        aggList[f'{width}'] = aggList[f'{width}'] + batchLyapDict[f'{width}']

for width in widths: 
    
    completeList[f'{width}'] = LyapListPairs(aggList[f'{width}']/5, thetaRange)

with open(f'completeLyapDataP{iP}.pickle', 'wb') as handle: 
    pickle.dump(completeList, handle, protocol=pickle.HIGHEST_PROTOCOL)

# +
import matplotlib.pyplot as plt

for width in [20, 40, 80, 100, 120]: 
    x = testList[f'{width}'][:,0]
    y = testList[f'{width}'][:,1]/(width/2)
    plt.scatter(x, y, s=3);
    print(x[9], y[9])

# +
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

width = 80
GRange = critVal * (1 + np.linspace(-0.16, 0.16, 65))

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

GaussianFit = curve_fit(gauss, testList[f'{width}'][:,0], testList[f'{width}'][:,1])[0]

plt.scatter(testList[f'{width}'][:,0], testList[f'{width}'][:,1], s=15);
plt.plot(thetaRange, gauss(thetaRange, *GaussianFit));
plt.vlines(GaussianFit[2],min(testList[f'{width}'][:,1]), max(testList[f'{width}'][:,1]), color='red');
print(GaussianFit[2])
# -


