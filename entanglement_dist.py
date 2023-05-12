import torch
from opt_einsum import contract
import numpy as np
import string
import itertools as itt

def shannon_entropy(rho):
    dim=int(np.sqrt(rho.nelement()))
    rho=rho.reshape(dim,dim)
    rho=rho/torch.trace(rho) # normalize before eigval for numerical safety
    s=torch.linalg.eigvalsh(rho)
    s=s.abs()/s.abs().sum()
    return torch.xlogy(s,s).sum().abs()/np.log(2)

def svd_tensor_to_isometry(M,idx1,idx2):
    shape1=tuple(M.shape[i] for i in idx1)
    shape2=tuple(M.shape[i] for i in idx2)
    M=M.permute(idx1+idx2).reshape(np.prod(shape1),np.prod(shape2))
    u,_,vh=torch.linalg.svd(M,full_matrices=False)
    uvh=(u@vh).reshape(shape1+shape2).permute(tuple(np.argsort(idx1+idx2)))
    return uvh

def entanglement_distance(SA,SB,SAB,dA,dB):
    max_IAB=2*np.log2(min(dA,dB))
    assert SA+SB-SAB<=max_IAB*1.0001
    return torch.log((max_IAB/(SA+SB-SAB).clamp(0)).clamp(0))

def get_density_matrix(phi,idx):
    assert len(set(idx))==len(idx) and set(idx).issubset(range(phi.ndim))
    idx1=set(range(phi.ndim))-set(idx)
    phi=phi.permute(tuple(idx)+tuple(idx1))
    assert len(idx)<=len(string.ascii_lowercase)
    eq1=string.ascii_lowercase[:len(idx)]
    eq2=string.ascii_uppercase[:len(idx)]
    return contract(f'{eq1}...,{eq2}...->{eq1+eq2}',phi,phi.conj())

def get_tensor_geometry(phi):
    S_single=[shannon_entropy(get_density_matrix(phi,[i])) 
              for i in range(phi.ndim)]
    S_double={(i,j):shannon_entropy(get_density_matrix(phi,[i,j])) 
              for i,j in itt.combinations(range(phi.ndim),2)}
    dist=np.zeros((phi.ndim,phi.ndim))
    for (i,j) in itt.combinations(range(phi.ndim),2):
        dist[i,j]=entanglement_distance(S_single[i],S_single[j],S_double[(i,j)],phi.shape[i],phi.shape[j]).cpu().item()
        dist[j,i]=dist[i,j]
    return dist