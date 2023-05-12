# doVarMERA.py
# ---------------------------------------------------------------------
# Variational energy minimization of (scale-invariant) modified binary MERA
#
# by Glen Evenbly (c) for www.tensors.net, (v1.2) - last modified 6/2019

import numpy as np
from numpy import linalg as LA
from ncon import ncon


def doVarMERA(hamAB, hamBA, rhoAB, rhoBA, wC, vC, uC, chi, chimid, OPTS):
  """
  Variational energy minimization of (scale-invariant) modified binary MERA
  for nearest neighbour 1D Hamiltonian. Inputs 'hamAB, hamBA, rhoAB, rhoBA,
  wC, vC, uC' are lists whose lengths are equal to the number of MERA
  levels. Input Hamiltonian specified through 'hamAB[0]' and 'hamBA[0]'.
  Bond dimensions specified through 'chi' and 'chimid'.

  OPTS is a dict containing the optional arguments:
    numiter: int=1000, number of variatonal iterations
    refsym: bool=True, impose reflection symmetry
    numtrans: int=2, number of transitional layers
    dispon: bool=True, specify wether or not to display convergence data
    E0: float=0.0, specify exact ground energy (if known)
    sciter: int=4, iterations of power method to find rho
  """
  if 'numiter' not in OPTS:
    OPTS['numiter'] = 1000
  if 'numtrans' not in OPTS:
    OPTS['numtrans'] = 2
  if 'refsym' not in OPTS:
    OPTS['refsym'] = True
  if 'dispon' not in OPTS:
    OPTS['dispon'] = True
  if 'sciter' not in OPTS:
    OPTS['sciter'] = 4
  if 'E0' not in OPTS:
    OPTS['E0'] = 0

  # Add extra layers if required
  totLv = OPTS['numtrans'] + 1
  for k in range(totLv - len(wC)):
    wC.append(wC[-1])
    vC.append(vC[-1])
    uC.append(uC[-1])

  for k in range(1 + totLv - len(hamAB)):
    hamAB.append(hamAB[-1])
    hamBA.append(hamBA[-1])
    rhoAB.append(rhoAB[-1])
    rhoBA.append(rhoBA[-1])

  # Expand tensors to new dimensions if required
  chiZ = np.zeros(totLv + 1, dtype=int)
  chiZ[0] = hamAB[0].shape[0]
  chimidZ = np.zeros(totLv + 1, dtype=int)
  chimidZ[0] = hamAB[0].shape[0]
  for k in range(totLv):
    chiZ[k + 1] = min(chi, chiZ[k] * chimidZ[k])
    chimidZ[k + 1] = min(chimid, chiZ[k])
    wC[k] = TensorExpand(wC[k], [chiZ[k], chimidZ[k + 1], chiZ[k + 1]])
    vC[k] = TensorExpand(vC[k], [chiZ[k], chimidZ[k + 1], chiZ[k + 1]])
    uC[k] = TensorExpand(uC[k],
                         [chiZ[k], chiZ[k], chimidZ[k + 1], chimidZ[k + 1]])
    hamAB[k + 1] = TensorExpand(
        hamAB[k + 1], [chiZ[k + 1], chiZ[k + 1], chiZ[k + 1], chiZ[k + 1]])
    hamBA[k + 1] = TensorExpand(
        hamBA[k + 1], [chiZ[k + 1], chiZ[k + 1], chiZ[k + 1], chiZ[k + 1]])
    rhoAB[k + 1] = TensorExpand(
        rhoAB[k + 1], [chiZ[k + 1], chiZ[k + 1], chiZ[k + 1], chiZ[k + 1]])
    rhoBA[k + 1] = TensorExpand(
        rhoBA[k + 1], [chiZ[k + 1], chiZ[k + 1], chiZ[k + 1], chiZ[k + 1]])

  # Ensure Hamiltonian is negative defined
  hamABstart = hamAB[0]
  hamBAstart = hamBA[0]
  bias = max(LA.eigvalsh(hamAB[0].reshape(chiZ[0]**2, chiZ[0]**2)))
  hamAB[0] = hamAB[0] - bias * np.eye(chiZ[0]**2).reshape(
      chiZ[0], chiZ[0], chiZ[0], chiZ[0])
  hamBA[0] = hamBA[0] - bias * np.eye(chiZ[0]**2).reshape(
      chiZ[0], chiZ[0], chiZ[0], chiZ[0])

  Energy = 0
  for k in range(OPTS['numiter']):
    # Find scale-invariant density matrix (via power method)
    for g in range(OPTS['sciter']):
      rhoABtemp, rhoBAtemp = DescendSuper(rhoAB[totLv], rhoBA[totLv],
                                          wC[totLv - 1], vC[totLv - 1],
                                          uC[totLv - 1], OPTS['refsym'])
      rhoAB[totLv] = 0.5 * (rhoABtemp + np.conj(
          rhoABtemp.transpose(2, 3, 0, 1))) / ncon([rhoABtemp], [[1, 2, 1, 2]])
      rhoBA[totLv] = 0.5 * (rhoBAtemp + np.conj(
          rhoBAtemp.transpose(2, 3, 0, 1))) / ncon([rhoBAtemp], [[1, 2, 1, 2]])
      if OPTS['refsym']:
        rhoAB[totLv] = 0.5 * rhoAB[totLv] + 0.5 * rhoAB[totLv].transpose(
            1, 0, 3, 2)
        rhoBA[totLv] = 0.5 * rhoBA[totLv] + 0.5 * rhoBA[totLv].transpose(
            1, 0, 3, 2)

    # Descend density matrix through all layers
    for p in range(totLv - 1, -1, -1):
      rhoAB[p], rhoBA[p] = DescendSuper(rhoAB[p + 1], rhoBA[p + 1], wC[p],
                                        vC[p], uC[p], OPTS['refsym'])

    # Compute energy and display
    if OPTS['dispon']:
      if np.mod(k, 100) == 1:
        Energy = (ncon([rhoAB[0], hamAB[0]], [[1, 2, 3, 4], [1, 2, 3, 4]]) +
                  ncon([rhoBA[0], hamBA[0]],
                       [[1, 2, 3, 4], [1, 2, 3, 4]])) / 4 + bias / 2
        ExpectX = ncon([
            rhoAB[0].reshape(2, 2, 2, 2, 2, 2, 2, 2),
            np.array([[0, 1], [1, 0]])
        ], [[4, 1, 2, 3, 5, 1, 2, 3], [4, 5]])

        print('Iteration: %d of %d, Energy: %f, Err: %e, Mag: %e' %
              (k, OPTS['numiter'], Energy, Energy - OPTS['E0'], ExpectX))

    # Optimise over all layers
    for p in range(totLv):
      if k > 9:
        uEnv = DisEnv(hamAB[p], hamBA[p], rhoBA[p + 1], wC[p], vC[p], uC[p],
                      OPTS['refsym'])
        if OPTS['refsym']:
          uEnv = uEnv + uEnv.transpose(1, 0, 3, 2)

        uC[p] = TensorUpdateSVD(uEnv, 2)

      if k > 1:
        wEnv = IsoEnvW(hamAB[p], hamBA[p], rhoBA[p + 1], rhoAB[p + 1], wC[p],
                       vC[p], uC[p])
        wC[p] = TensorUpdateSVD(wEnv, 2)
        if OPTS['refsym']:
          vC[p] = wC[p]
        else:
          vEnv = IsoEnvV(hamAB[p], hamBA[p], rhoBA[p + 1], rhoAB[p + 1], wC[p],
                         vC[p], uC[p])
          vC[p] = TensorUpdateSVD(vEnv, 2)

      hamAB[p + 1], hamBA[p + 1] = AscendSuper(hamAB[p], hamBA[p], wC[p], vC[p],
                                               uC[p], OPTS['refsym'])

  hamAB[0] = hamABstart
  hamBA[0] = hamBAstart

  return Energy, hamAB, hamBA, rhoAB, rhoBA, wC, vC, uC


def AscendSuper(hamAB, hamBA, w, v, u, refsym):
  """ apply the average ascending superoperator to the Hamiltonian """

  indList1 = [[6, 4, 1, 2], [1, 3, -3], [6, 7, -1], [2, 5, 3, 9], [4, 5, 7, 10],
              [8, 9, -4], [8, 10, -2]]
  indList2 = [[3, 4, 1, 2], [5, 6, -3], [5, 7, -1], [1, 2, 6, 9], [3, 4, 7, 10],
              [8, 9, -4], [8, 10, -2]]
  indList3 = [[5, 7, 2, 1], [8, 9, -3], [8, 10, -1], [4, 2, 9, 3],
              [4, 5, 10, 6], [1, 3, -4], [7, 6, -2]]
  indList4 = [[3, 6, 2, 5], [2, 1, -3], [3, 1, -1], [5, 4, -4], [6, 4, -2]]

  hamBAout = ncon(
      [hamAB, w, np.conj(w), u,
       np.conj(u), v, np.conj(v)], indList1)
  if refsym:
    hamBAout = hamBAout + hamBAout.transpose(1, 0, 3, 2)
  else:
    hamBAout = hamBAout + ncon(
        [hamAB, w, np.conj(w), u,
         np.conj(u), v, np.conj(v)], indList3)

  hamBAout = hamBAout + ncon(
      [hamBA, w, np.conj(w), u,
       np.conj(u), v, np.conj(v)], indList2)
  hamABout = ncon([hamBA, v, np.conj(v), w, np.conj(w)], indList4)

  return hamABout, hamBAout


def DescendSuper(rhoAB, rhoBA, w, v, u, refsym):
  """ apply the average descending superoperator to the density matrix """

  indList1 = [[9, 3, 4, 2], [-3, 5, 4], [-1, 10, 9], [-4, 7, 5, 6],
              [-2, 7, 10, 8], [1, 6, 2], [1, 8, 3]]
  indList2 = [[3, 6, 2, 5], [1, 7, 2], [1, 9, 3], [-3, -4, 7, 8],
              [-1, -2, 9, 10], [4, 8, 5], [4, 10, 6]]
  indList3 = [[3, 9, 2, 4], [1, 5, 2], [1, 8, 3], [7, -3, 5, 6], [7, -1, 8, 10],
              [-4, 6, 4], [-2, 10, 9]]
  indList4 = [[3, 6, 2, 5], [-3, 1, 2], [-1, 1, 3], [-4, 4, 5], [-2, 4, 6]]

  rhoABout = 0.5 * ncon(
      [rhoBA, w, np.conj(w), u,
       np.conj(u), v, np.conj(v)], indList1)
  if refsym:
    rhoABout = rhoABout + rhoABout.transpose(1, 0, 3, 2)
  else:
    rhoABout = rhoABout + 0.5 * ncon(
        [rhoBA, w, np.conj(w), u,
         np.conj(u), v, np.conj(v)], indList3)

  rhoBAout = 0.5 * ncon(
      [rhoBA, w, np.conj(w), u,
       np.conj(u), v, np.conj(v)], indList2)
  rhoBAout = rhoBAout + 0.5 * ncon(
      [rhoAB, v, np.conj(v), w, np.conj(w)], indList4)

  return rhoABout, rhoBAout


def DisEnv(hamAB, hamBA, rhoBA, w, v, u, refsym):
  """ compute the environment of a disentangler """

  indList1 = [[7, 8, 10, -1], [4, 3, 9, 2], [10, -3, 9], [7, 5, 4],
              [8, -2, 5, 6], [1, -4, 2], [1, 6, 3]]
  indList2 = [[7, 8, -1, -2], [3, 6, 2, 5], [1, -3, 2], [1, 9, 3],
              [7, 8, 9, 10], [4, -4, 5], [4, 10, 6]]
  indList3 = [[7, 8, -2, 10], [3, 4, 2, 9], [1, -3, 2], [1, 5, 3],
              [-1, 7, 5, 6], [10, -4, 9], [8, 6, 4]]

  uEnv = ncon(
      [hamAB, rhoBA, w, np.conj(w),
       np.conj(u), v, np.conj(v)], indList1)
  if refsym:
    uEnv = uEnv + uEnv.transpose(1, 0, 3, 2)
  else:
    uEnv = uEnv + ncon(
        [hamAB, rhoBA, w,
         np.conj(w), np.conj(u), v,
         np.conj(v)], indList3)

  uEnv = uEnv + ncon(
      [hamBA, rhoBA, w, np.conj(w),
       np.conj(u), v, np.conj(v)], indList2)

  return uEnv


def IsoEnvW(hamAB, hamBA, rhoBA, rhoAB, w, v, u):
  """ compute the environment of a 'w'-isometry """

  indList1 = [[7, 8, -1, 9], [4, 3, -3, 2], [7, 5, 4], [9, 10, -2, 11],
              [8, 10, 5, 6], [1, 11, 2], [1, 6, 3]]
  indList2 = [[1, 2, 3, 4], [10, 7, -3, 6], [-1, 11, 10], [3, 4, -2, 8],
              [1, 2, 11, 9], [5, 8, 6], [5, 9, 7]]
  indList3 = [[5, 7, 3, 1], [10, 9, -3, 8], [-1, 11, 10], [4, 3, -2, 2],
              [4, 5, 11, 6], [1, 2, 8], [7, 6, 9]]
  indList4 = [[3, 7, 2, -1], [5, 6, 4, -3], [2, 1, 4], [3, 1, 5], [7, -2, 6]]

  wEnv = ncon(
      [hamAB, rhoBA, np.conj(w), u,
       np.conj(u), v, np.conj(v)], indList1)
  wEnv = wEnv + ncon(
      [hamBA, rhoBA, np.conj(w), u,
       np.conj(u), v, np.conj(v)], indList2)
  wEnv = wEnv + ncon(
      [hamAB, rhoBA, np.conj(w), u,
       np.conj(u), v, np.conj(v)], indList3)
  wEnv = wEnv + ncon([hamBA, rhoAB, v, np.conj(v), np.conj(w)], indList4)

  return wEnv


def IsoEnvV(hamAB, hamBA, rhoBA, rhoAB, w, v, u):
  """ compute the environment of a 'v'-isometry """

  indList1 = [[6, 4, 1, 3], [9, 11, 8, -3], [1, 2, 8], [6, 7, 9], [3, 5, 2, -2],
              [4, 5, 7, 10], [-1, 10, 11]]
  indList2 = [[3, 4, 1, 2], [8, 10, 9, -3], [5, 6, 9], [5, 7, 8], [1, 2, 6, -2],
              [3, 4, 7, 11], [-1, 11, 10]]
  indList3 = [[9, 10, 11, -1], [3, 4, 2, -3], [1, 8, 2], [1, 5, 3],
              [7, 11, 8, -2], [7, 9, 5, 6], [10, 6, 4]]
  indList4 = [[7, 5, -1, 4], [6, 3, -3, 2], [7, -2, 6], [4, 1, 2], [5, 1, 3]]

  vEnv = ncon(
      [hamAB, rhoBA, w, np.conj(w), u,
       np.conj(u), np.conj(v)], indList1)
  vEnv = vEnv + ncon(
      [hamBA, rhoBA, w, np.conj(w), u,
       np.conj(u), np.conj(v)], indList2)
  vEnv = vEnv + ncon(
      [hamAB, rhoBA, w, np.conj(w), u,
       np.conj(u), np.conj(v)], indList3)
  vEnv = vEnv + ncon([hamBA, rhoAB, np.conj(v), w, np.conj(w)], indList4)

  return vEnv


def TensorExpand(A, chivec):
  """ expand tensor dimension by padding with zeros """

  if [*A.shape] == chivec:
    return A
  else:
    for k in range(len(chivec)):
      if A.shape[k] != chivec[k]:
        indloc = list(range(-1, -len(chivec) - 1, -1))
        indloc[k] = 1
        A = ncon([A, np.eye(A.shape[k], chivec[k])], [indloc, [1, -k - 1]])

    return A


def TensorUpdateSVD(wIn, leftnum):
  """ update an isometry using its (linearized) environment """

  wSh = wIn.shape
  ut, st, vht = LA.svd(
      wIn.reshape(np.prod(wSh[0:leftnum:1]), np.prod(wSh[leftnum:len(wSh):1])),
      full_matrices=False)
  return -(ut @ vht).reshape(wSh)