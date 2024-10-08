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

def TransfMatGenerator(Theta, m, nw, rngseed, insertProbability=0.0): 
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

        #AB_R = A_R @ B_R

        MatrixList.append(*diagBlocksA)
    
    return MatrixList

MatrixCount = 10000


# Parameter Extraction

# +
import dcor

ThetaList = np.linspace(np.pi/32, np.pi/2 - np.pi/32, 15)

covList = dict()
dcovList = dict()
dcorList = dict()

for Theta in enumerate(ThetaList): 

    MatList = TransfMatGenerator(Theta[1], 1, MatrixCount, 42, insertProbability = 0.0)

    T = np.array(MatList)

    detList = np.linalg.det(T)
    sqrtDetList = np.sqrt(detList)
    etaVecList = np.array([(np.real(d), np.imag(d)) for d in detList])
    etaScaList = np.angle(detList)

    TNorm = np.array([T[i]/sqrtDetList[i] for i in range(MatrixCount)])

    turnList = np.arccos(1/np.emath.sqrt(np.real(np.array([TNorm[i, 0, 0] * TNorm[i, 1, 1] for i in range(MatrixCount)]))))
    turnZetaList = np.array(np.arccosh(1/np.cos(turnList)))
    
    aList = np.array([-TNorm[i, 0, 0] for i in range(MatrixCount)]) * np.array([TNorm[i, 0, 1] for i in range(MatrixCount)]) / (np.sinh(turnZetaList) * np.cosh(turnZetaList))
    bList = (np.array([-TNorm[i, 0, 0] for i in range(MatrixCount)]) / (np.array([TNorm[i, 0, 1] for i in range(MatrixCount)]))) * np.tanh(turnZetaList)
    
    aVecList = np.array([(np.real(a), np.imag(a)) for a in aList])
    bVecList = np.array([(np.real(b), np.imag(b)) for b in bList])
    aScaList = np.angle(aList)
    bScaList = np.angle(bList)

    SplitQRep = np.array([(np.real(TNorm[i][0,0]),  np.imag(TNorm[i][0,0]), np.real(TNorm[i][0,1]), np.imag(TNorm[i][0,1])) for i in range(MatrixCount)])

    aaList = SplitQRep[:,0]
    bbList = SplitQRep[:,2]
    zzList = SplitQRep[:,0]**2 + SplitQRep[:,1]**2

    covList[f'{Theta[0]}'] = np.cov([aaList, bbList, zzList])
    dcovList[f'{Theta[0]}'] = [dcor.u_distance_covariance_sqr(aaList, zzList), dcor.u_distance_covariance_sqr(bbList, zzList), dcor.u_distance_covariance_sqr(aaList, zzList), dcor.u_distance_covariance_sqr(aaList, aaList), dcor.u_distance_covariance_sqr(bbList, bbList)]
    dcorList[f'{Theta[0]}'] = [dcor.u_distance_correlation_sqr(aaList, zzList), dcor.u_distance_correlation_sqr(bbList, zzList), dcor.u_distance_correlation_sqr(aaList, zzList), dcor.u_distance_correlation_sqr(aaList, aaList), dcor.u_distance_correlation_sqr(bbList, bbList)]
# -

import matplotlib.pyplot as plt

# +
fig, axs = plt.subplots(2, 2)

axs[0, 0].hist(aScaList, 100, histtype='step')
axs[0, 0].set_title('a')
axs[0, 1].hist(bScaList, 100, histtype='step')
axs[0, 1].set_title('b')
axs[1, 0].hist(etaScaList, 100, histtype='step')
axs[1, 0].set_title('\u03b7')
axs[1, 1].hist(turnList, 100, histtype='step')
axs[1, 1].set_title('\u03b8')

fig.set_size_inches(8,6)
fig.tight_layout()

# +
import matplotlib.pyplot as plt

plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(covList.values()))[:,0,0], s=5)

plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(covList.values()))[:,1,1], s=5)

#plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(covList.values()))[:,2,2], s=5)

plt.legend(['cov(2a,2a)', 'cov(2b,2b)', 'cov(2eta, 2eta)'])

# +
plt.scatter(ThetaList, np.array(list(covList.values()))[:,0,0], s=5)

plt.scatter(ThetaList, np.array(list(covList.values()))[:,1,1], s=5)

#plt.scatter(ThetaList, np.array(list(covList.values()))[:,2,2], s=5)

plt.legend(['cov(2a,2a)', 'cov(2b,2b)', 'cov(2eta, 2eta)'])

# +
plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(dcovList.values()))[:,3], s=5)

plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(dcovList.values()))[:,4], s=5)

#plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(dcovList.values()))[:,5], s=5)

plt.legend(['dcov(2a,2a)', 'dcov(2b,2b)', 'dcov(2eta,2eta)'])

# +
plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(dcovList.values()))[:,0], s=5)

plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(dcovList.values()))[:,1], s=5)

#plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(dcovList.values()))[:,2], s=5)

plt.legend(['dcov(2a,zeta)', 'dcov(2b,zeta)', 'dcov(2eta,zeta)'])

# +
plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(dcorList.values()))[:,0], s=5)

plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(dcorList.values()))[:,1], s=5)

#plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(dcorList.values()))[:,2], s=5)

plt.legend(['dcor(2a,zeta)', 'dcor(2b,zeta)', 'dcor(2eta,zeta)'])

# +
plt.scatter(ThetaList, np.array(list(dcorList.values()))[:,0], s=5)

plt.scatter(ThetaList, np.array(list(dcorList.values()))[:,1], s=5)

#plt.scatter(ThetaList, np.array(list(dcorList.values()))[:,2], s=5)

plt.legend(['dcor(2a,theta)', 'dcor(2b,theta)', 'dcor(2eta,theta)'])
