# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# +
import numpy as np
import scipy as sp

def ATransferMatrixGenerator(S, T):
    A = np.array([[S, -T], [-T, S]])

    return A

def BTransferMatrixGenerator(C_s, C_o): 
    B = np.array([[C_s, C_o], [C_o, C_s]])

    return B


# -

rng = np.random.default_rng(seed=42)


def MatrixGenerator(MatrixType, Theta): 

    S = 1/np.cos(Theta)
    T = np.tan(Theta)
    Cs = 1/np.sin(Theta)
    Co = np.cos(Theta)/np.sin(Theta)

    A = ATransferMatrixGenerator(S,T)
    B = BTransferMatrixGenerator(Cs, Co)

    phases = np.exp((np.pi/2)*1j*rng.uniform(low=-1, high=1, size=3))

    if MatrixType == 'A':
        M = np.diag([phases[0], phases[0]**(-1)]) @ (phases[1] * np.eye(2)) @ A @ np.diag([phases[2], phases[2]**(-1)])
    
    if MatrixType == 'B':
        M = np.diag([phases[0], phases[0]**(-1)]) @ (phases[1] * np.eye(2)) @ B @ np.diag([phases[2], phases[2]**(-1)])

    return M


def RMatrixGenerator(MatrixType, Theta): 

    S = 1/np.cos(Theta)
    T = np.tan(Theta)
    Cs = 1/np.sin(Theta)
    Co = np.cos(Theta)/np.sin(Theta)

    A = ATransferMatrixGenerator(S,T)
    B = BTransferMatrixGenerator(Cs, Co)

    phases = np.exp((np.pi/2)*1j*rng.uniform(low=-1, high=1, size=(4,4)))

    if MatrixType == 'A':
        M = np.diag(phases[0]) @ sp.linalg.block_diag(A, A) @ np.diag(phases[1]) @ sp.linalg.block_diag(1, B, 1) @ np.diag(phases[2]) @ sp.linalg.block_diag(A, A) @ np.diag(phases[3])
    
    if MatrixType == 'B':
        M = np.diag(phases[0]) @ sp.linalg.block_diag(B, B) @ np.diag(phases[1]) @ sp.linalg.block_diag(1, A, 1) @ np.diag(phases[2]) @ sp.linalg.block_diag(B, B) @ np.diag(phases[3])

    R_00 = M[0,0] + (M[0,1] + M[0,2])*(M[2,0] - M[1,0]) / (M[1,1] + M[1,2] - M[2,1] - M[2,2])
    R_01 = M[0,3] + (M[0,1] + M[0,2])*(M[2,3] - M[1,3]) / (M[1,1] + M[1,2] - M[2,1] - M[2,2])
    R_10 = M[3,0] + (M[3,1] + M[3,2])*(M[2,0] - M[1,0]) / (M[1,1] + M[1,2] - M[2,1] - M[2,2])
    R_11 = M[3,3] + (M[3,1] + M[3,2])*(M[2,3] - M[1,3]) / (M[1,1] + M[1,2] - M[2,1] - M[2,2])

    RMatrix = np.array([[R_00, R_01], [R_10, R_11]])
    
    return RMatrix


MatrixCount = 25000

# +
import matplotlib.pyplot as plt
import dcor

ThetaList = np.linspace(np.pi/32, np.pi/2 - np.pi/32, 31)

covList = dict()
dcovList = dict()
dcorList = dict()

for Theta in enumerate(ThetaList): 

    MatList = []

    for i in range(MatrixCount): 
        MatList.append(MatrixGenerator('B', Theta[1]))

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

    # fig, axs = plt.subplots(2, 2)

    # axs[0, 0].hist(aScaList, 100, histtype='step')
    # axs[0, 0].set_title('a')
    # axs[0, 1].hist(bScaList, 100, histtype='step')
    # axs[0, 1].set_title('b')
    # axs[1, 0].hist(etaScaList, 100, histtype='step')
    # axs[1, 0].set_title('\u03b7')
    # axs[1, 1].hist(turnList, 100, histtype='step')
    # axs[1, 1].set_title('\u03b8')

    # fig.set_size_inches(8,6)
    # fig.tight_layout()

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
plt.scatter(ThetaList, np.array(list(dcovList.values()))[:,3], s=5)

plt.scatter(ThetaList, np.array(list(dcovList.values()))[:,4], s=5)

#plt.scatter(ThetaList, np.array(list(dcovList.values()))[:,5], s=5)

plt.legend(['dcov(2a,2a)', 'dcov(2b,2b)', 'dcov(2eta,2eta)'])

# +
plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(dcovList.values()))[:,0], s=5)

plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(dcovList.values()))[:,1], s=5)

#plt.scatter(np.arccosh(1/np.cos(ThetaList)), np.array(list(dcovList.values()))[:,2], s=5)

plt.legend(['dcov(2a,zeta)', 'dcov(2b,zeta)', 'dcov(2eta,zeta)'])

# +
plt.scatter(ThetaList, np.array(list(dcovList.values()))[:,0], s=5)

plt.scatter(ThetaList, np.array(list(dcovList.values()))[:,1], s=5)

#plt.scatter(ThetaList, np.array(list(dcovList.values()))[:,2], s=5)

plt.legend(['dcov(2a,theta)', 'dcov(2b,theta)', 'dcov(2eta,theta)'])

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
