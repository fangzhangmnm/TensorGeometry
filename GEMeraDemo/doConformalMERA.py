# doConformalMERA.py
# ---------------------------------------------------------------------
# Extraction of conformal data from an optimised MERA
#
# by Glen Evenbly (c) for www.tensors.net, (v1.2) - last modified 6/2019

import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import eigs
from ncon import ncon


def doConformalMERA(wS, uS, vS, rhoBAS, scnum):
  """
  Compute conformal data from an modified binary MERA optimized for a
  scale-invariant critical point. Input 'ws', 'vs' and 'uS' are the
  isometries and disentangler from the scale-invariant layers, while
  'rhoBAS' is the fixed point density matrix. 'scnum' sets the number of
  scaling dimensions to compute.

  Outputs 'scDims', 'scOps', and 'Cfusion' are the scaling dimensions,
  scaling operators and fusion coefficients respectively.
  """

  # Diagonalize 1-site scaling superoperator
  chi = wS.shape[2]
  tensors = [wS, np.conj(wS), vS, np.conj(vS)]
  connects = [[-4, 1, 3], [-3, 1, 4], [3, 2, -2], [4, 2, -1]]
  ScSuper1 = ncon(tensors, connects).reshape(chi**2, chi**2)

  dtemp, utemp = eigs(ScSuper1, k=scnum, which='LM')
  scDims = -np.log2(abs(dtemp)) / 2

  # Normalize scaling operators
  scOps = [0 for x in range(scnum)]
  for k in range(scnum):
    scAtemp = utemp[:, k].reshape(chi, chi)
    scAtemp = scAtemp / LA.norm(scAtemp)

    tensors = [scAtemp, scAtemp, wS, np.conj(wS), uS, np.conj(uS), vS,
               np.conj(vS), rhoBAS]
    connects = [[8, 7], [3, 1], [7, 9, 11], [8, 10, 13], [2, 1, 9, 5],
                [2, 3, 10, 6], [4, 5, 12], [4, 6, 14], [13, 14, 11, 12]]
    cweight = ncon(tensors, connects)
    scOps[k] = scAtemp / np.sqrt(cweight)

  # Compute fusion coefficients (OPE coefficients)
  Cfusion = np.zeros((scnum, scnum, scnum), dtype=complex)
  for k1 in range(scnum):
    for k2 in range(scnum):
      for k3 in range(scnum):
        Otemp = scDims[k1] - scDims[k2] + scDims[k3]
        tensors = [scOps[k1], scOps[k2], scOps[k3], wS, np.conj(wS), uS,
                   np.conj(uS), vS, np.conj(vS), uS, np.conj(uS), wS,
                   np.conj(wS), wS, np.conj(wS), vS, np.conj(vS), rhoBAS]
        connects = [[5, 4], [3, 1], [28, 27], [4, 6, 11], [5, 7, 13],
                    [2, 1, 6, 9], [2, 3, 7, 10], [8, 9, 12], [8, 10, 14],
                    [11, 12, 16, 21], [13, 14, 17, 23], [15, 16, 18],
                    [15, 17, 19], [27, 26, 24], [28, 26, 25], [24, 21, 20],
                    [25, 23, 22], [19, 22, 18, 20]]
        Cfusion[k1, k2, k3] = (2**Otemp) * ncon(tensors, connects)

  return scDims, scOps, Cfusion