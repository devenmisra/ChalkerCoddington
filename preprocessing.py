# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: np-test-openblas
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

def TransfMatGenerator(Theta, m, nw, seed, insertProbability=0.0): 
    S = 1/np.cos(Theta)
    T = np.tan(Theta)
    Cs = 1/np.sin(Theta)
    Co = np.cos(Theta)/np.sin(Theta)
    np.random.seed(seed)
    phases = np.exp(2*np.pi*1j*np.random.rand(nw, 2, 2*m))
    MatrixList = []
    
    for j in range(0, nw):
        A = ATransferMatrixGenerator(S, T, m, phases[j,0])
        diagBlocksA = extract_block_diag_A(A, 2)

        B = BTransferMatrixGenerator(Cs, Co, m, phases[j,1])
        diagBlocksB = extract_block_diag_B(B, 2, m)

        for i in range(m): 
            if np.random.rand(2*m)[i] < insertProbability:
                diagBlocksA[i] = RMatrixGenerator('A', diagBlocksA, diagBlocksB, m)

            if np.random.rand(2*m)[m+i] < insertProbability:
                diagBlocksB[i] = RMatrixGenerator('B', diagBlocksA, diagBlocksB, m)

        A_R = sp.linalg.block_diag(*diagBlocksA)
        B_R = sp.linalg.block_diag(*[diagBlocksB[0][0,0], *diagBlocksB[1:], diagBlocksB[0][1,1]]) +  sp.sparse.coo_array(([diagBlocksB[0][0,1], diagBlocksB[0][1,0]], [(0, 2*m-1), (2*m-1, 0)]))

        #AB_R = A_R @ B_R

        MatrixList.append(A_R)
    
    return MatrixList


ThetaList = [np.pi/4 + 0.1, np.pi/4 - 0.1]


# Original Matrices

RMatListUnder = TransfMatGenerator(ThetaList[0], 1, 100000, 42, insertProbability = 0.0)
RMatListOver = TransfMatGenerator(ThetaList[1], 1, 100000, 42, insertProbability = 0.0)

# Replacement Matrices

RMatListUnder_R = TransfMatGenerator(ThetaList[0], 1, 100000, 42, insertProbability = 1.0)
RMatListOver_R = TransfMatGenerator(ThetaList[1], 1, 100000, 42, insertProbability = 1.0)

# Serialize Outputs

import pickle

# +
with open(f'RMatListUnder.pickle', 'wb') as handle:
            pickle.dump(RMatListUnder, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'RMatListOver.pickle', 'wb') as handle:
            pickle.dump(RMatListOver, handle, protocol=pickle.HIGHEST_PROTOCOL)

# +
with open(f'RMatListUnder_R.pickle', 'wb') as handle:
            pickle.dump(RMatListUnder_R, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'RMatListOver_R.pickle', 'wb') as handle:
            pickle.dump(RMatListOver_R, handle, protocol=pickle.HIGHEST_PROTOCOL)
# -

# Load from Pickle

# +
with open(f'RMatListUnder.pickle', 'rb') as handle:
            RMatListUnder = pickle.load(handle)

with open(f'RMatListOver.pickle', 'rb') as handle:
            RMatListOver = pickle.load(handle)

# +
with open(f'RMatListUnder_R.pickle', 'rb') as handle:
            RMatListUnder_R = pickle.load(handle)

with open(f'RMatListOver_R.pickle', 'rb') as handle:
            RMatListOver_R = pickle.load(handle)
# -

T = np.array(RMatListOver)

detList = np.linalg.det(T)

sqrtDetList = np.sqrt(detList)

etaList = np.angle(detList)

TNorm = np.array([T[i]/sqrtDetList[i] for i in range(100000)])

aList = np.angle(np.array([TNorm[i, 0, 0] for i in range(100000)]))

bList = np.angle(np.array([TNorm[i, 0, 1] for i in range(100000)]))

turnList = np.angle(np.array([np.vdot(TNorm[i, 0, 0], TNorm[i, 1, 1]) for i in range(100000)]))

# +
import matplotlib.pyplot as plt

plt.hist(etaList, 100);
# -

plt.hist(aList, 100);
plt.hist(bList, 100);

plt.hist(turnList, 100);

np.cov([etaList, aList, bList, turnList])
