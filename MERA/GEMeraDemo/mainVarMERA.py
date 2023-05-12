# mainVarMERA.py
# ---------------------------------------------------------------------
# Script file for initializing the Hamiltonian and MERA tensors before
# passing to a variational energy minimization routine. Initiates the
# extraction of conformal data from MERA after the minimization is complete.
#
# by Glen Evenbly (c) for www.tensors.net, (v1.2) - last modified 6/2019


# Preamble
import numpy as np
import matplotlib.pyplot as plt

from doVarMERA import doVarMERA
from doConformalMERA import doConformalMERA

# Example 1: crit Ising model #####
#######################################

# Set bond dimensions and simulation options
chi = 6
chimid = 4

OPTS = {'numiter': 2000,  # number of variatonal iterations
        'refsym': True,  # impose reflection symmetry
        'numtrans': 1,  # number of transitional layers
        'dispon': True,  # display convergence data
        'E0': -4 / np.pi,  # specify exact ground energy (if known)
        'sciter': 4}  # iterations of power method to find density matrix

# Define Hamiltonian (quantum critical Ising), do preliminary 2->1 blocking
OPTS['numtrans'] = int(max(OPTS['numtrans'],
                           np.ceil(np.log(chi) / (2 * np.log(4)))))
sX = np.array([[0, 1], [1, 0]], dtype=float)
sZ = np.array([[1, 0], [0, -1]], dtype=float)
htemp = -np.kron(sX, sX) - 0.5 * (
    np.kron(sZ, np.eye(2)) + np.kron(np.eye(2), sZ))
hbig = (0.5 * np.kron(np.eye(4), htemp) +
        np.kron(np.eye(2), np.kron(htemp, np.eye(2))) +
        0.5 * np.kron(htemp, np.eye(4))).reshape(2, 2, 2, 2, 2, 2, 2, 2)
hamAB = [0] * (OPTS['numtrans'] + 2)
hamBA = [0] * (OPTS['numtrans'] + 2)
hamAB[0] = (hbig.transpose(0, 1, 3, 2, 4, 5, 7, 6)).reshape(4, 4, 4, 4)
hamBA[0] = (hbig.transpose(1, 0, 2, 3, 5, 4, 6, 7)).reshape(4, 4, 4, 4)

# Initialize tensors
totLv = OPTS['numtrans'] + 1
chiZ = np.zeros(totLv + 1, dtype=int)
chiZ[0] = hamAB[0].shape[0]
chimidZ = np.zeros(totLv + 1, dtype=int)
chimidZ[0] = hamAB[0].shape[0]
for k in range(totLv):
  chiZ[k + 1] = min(chi, chiZ[k] * chimidZ[k])
  chimidZ[k + 1] = min(chimid, chiZ[k + 1])

wC = [0] * (OPTS['numtrans'] + 1)
vC = [0] * (OPTS['numtrans'] + 1)
uC = [0] * (OPTS['numtrans'] + 1)
for k in range(totLv):
  wC[k] = np.random.rand(chiZ[k], chimidZ[k], chiZ[k + 1])
  vC[k] = np.random.rand(chiZ[k], chimidZ[k], chiZ[k + 1])
  uC[k] = (np.eye(chiZ[k]**2, chimidZ[k]**2)).reshape(chiZ[k], chiZ[k],
                                                      chimidZ[k], chimidZ[k])

rhoAB = [0] * (OPTS['numtrans'] + 2)
rhoBA = [0] * (OPTS['numtrans'] + 2)
rhoAB[0] = np.eye(chiZ[0]**2).reshape(chiZ[0], chiZ[0], chiZ[0], chiZ[0])
rhoBA[0] = np.eye(chiZ[0]**2).reshape(chiZ[0], chiZ[0], chiZ[0], chiZ[0])
for k in range(totLv):
  rhoAB[k + 1] = np.eye(chiZ[k + 1]**2).reshape(chiZ[k + 1], chiZ[k + 1],
                                                chiZ[k + 1], chiZ[k + 1])
  rhoBA[k + 1] = np.eye(chiZ[k + 1]**2).reshape(chiZ[k + 1], chiZ[k + 1],
                                                chiZ[k + 1], chiZ[k + 1])
  hamAB[k + 1] = np.zeros((chiZ[k + 1], chiZ[k + 1], chiZ[k + 1], chiZ[k + 1]))
  hamBA[k + 1] = np.zeros((chiZ[k + 1], chiZ[k + 1], chiZ[k + 1], chiZ[k + 1]))

def print_tensor_shapes():
  print('shapes of tensors:')
  for k in range(len(wC)):
    print('wC[' + str(k) + '].shape = ' + str(wC[k].shape))
    print('vC[' + str(k) + '].shape = ' + str(vC[k].shape))
    print('uC[' + str(k) + '].shape = ' + str(uC[k].shape))
    print('rhoAB[' + str(k) + '].shape = ' + str(rhoAB[k].shape))
    print('rhoBA[' + str(k) + '].shape = ' + str(rhoBA[k].shape))
    print('hamAB[' + str(k) + '].shape = ' + str(hamAB[k].shape))
    print('hamBA[' + str(k) + '].shape = ' + str(hamBA[k].shape))
def troubleshooting():
  # Compute the conformal data from the optimized MERA
  scnum = 10
  scDims, scOps, Cfusion = doConformalMERA(wC[-1], uC[-1], vC[-1],
                                          rhoBA[-1], scnum)

  # Compare with known results for Ising CFT
  scDimsExact = [0, 1 / 8, 1, 1 + 1 / 8, 1 + 1 / 8, 2, 2, 2, 2, 2 + 1 / 8]
  plt.figure(1)
  plt.plot(range(scnum), scDimsExact, 'b', label="exact")
  plt.plot(range(scnum), scDims, 'rx', label="MERA")
  plt.legend()
  plt.title('critical Ising model')
  plt.xlabel('k')
  plt.ylabel('Scaling Dims: Delta_k')
  plt.show()
  # also print scdims
  for k in range(scnum):
    print('scDimsExact[%d] = %f, scDims[%d] = %f' %
          (k, scDimsExact[k], k, scDims[k]))


  Cess = [Cfusion[1, 1, 2], Cfusion[1, 2, 1], Cfusion[2, 1, 1]]
  CessExact = [0.5, 0.5, 0.5]
  print('Fusion coefficients: Exact_C(ep,sg,sg) = %f, MERA_C(ep,sg,sg) = %f' %
        (CessExact[1], np.real(Cess[1])))
  
# print_tensor_shapes()
troubleshooting()


# Perform variational optimization
Energy, hamAB, hamBA, rhoAB, rhoBA, wC, vC, uC = doVarMERA(
    hamAB, hamBA, rhoAB, rhoBA, wC, vC, uC, chi, chimid, OPTS)
# print_tensor_shapes()
troubleshooting()

# Expand bond dimension and increase number of transitional layers,
# then continue variational optimization
chi = 8
chimid = 6
OPTS['numtrans'] = 2
OPTS['numiter'] = 1800
Energy, hamAB, hamBA, rhoAB, rhoBA, wC, vC, uC = doVarMERA(
    hamAB, hamBA, rhoAB, rhoBA, wC, vC, uC, chi, chimid, OPTS)
# print_tensor_shapes()
troubleshooting()

# Expand bond dimension and increase number of transitional layers,
# then continue variational optimization
chi = 12
chimid = 8
OPTS['numtrans'] = 3
OPTS['numiter'] = 1400
Energy, hamAB, hamBA, rhoAB, rhoBA, wC, vC, uC = doVarMERA(
    hamAB, hamBA, rhoAB, rhoBA, wC, vC, uC, chi, chimid, OPTS)
# print_tensor_shapes()
troubleshooting()


# Save data
np.save('IsingData.npy', (wC, uC, vC, rhoAB, rhoBA))
