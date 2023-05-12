from opt_einsum import contract
import numpy as np
import torch
from dataclasses import asdict
from tqdm.auto import tqdm
from utils import show_tensor_ijkl
import matplotlib.pyplot as plt
from copy import deepcopy
import itertools as itt
import os,sys
from IPython.utils.io import Tee
from contextlib import redirect_stdout
device=torch.device('cuda:1')
torch.set_default_tensor_type(torch.cuda.DoubleTensor)
torch.cuda.set_device(device)

from MERA import *

def get_Ising1D_Hamiltonians(J,g):
    sZ=torch.tensor([[1,0],[0,-1]])
    sX=torch.tensor([[0,1],[1,0]])
    eye=torch.eye(2)
    ZZ=contract('iI,jJ->ijIJ',sZ,sZ)
    XI=contract('iI,jJ->ijIJ',sX,eye)
    IX=contract('iI,jJ->ijIJ',eye,sX)
    h=-J*(ZZ+g*(XI+IX)/2)
    return h,h.clone()

def get_Ising1D_Hamiltonians_2blocked(J,g):
    sZ=torch.tensor([[1,0],[0,-1]])
    sX=torch.tensor([[0,1],[1,0]])
    eye=torch.eye(2)
    ZZII=contract('iI,jJ,kK,lL->ijklIJKL',sZ,sZ,eye,eye)
    IZZI=contract('iI,jJ,kK,lL->ijklIJKL',eye,sZ,sZ,eye)
    IIZZ=contract('iI,jJ,kK,lL->ijklIJKL',eye,eye,sZ,sZ)
    XIII=contract('iI,jJ,kK,lL->ijklIJKL',sX,eye,eye,eye)
    IXII=contract('iI,jJ,kK,lL->ijklIJKL',eye,sX,eye,eye)
    IIXI=contract('iI,jJ,kK,lL->ijklIJKL',eye,eye,sX,eye)
    IIIX=contract('iI,jJ,kK,lL->ijklIJKL',eye,eye,eye,sX)
    h=((-J*((ZZII+IIZZ)/2+IZZI)-J*g*(XIII+IXII+IIXI+IIIX)/2)/2).reshape(2,2,2,2,2,2,2,2)
    # need to take care of the reflection symmetry!
    hAB=contract('abcdABCD->abdcABDC',h).reshape(4,4,4,4)
    hBA=contract('abcdABCD->bacdBACD',h).reshape(4,4,4,4)
    return hAB, hBA


energy_ref=-4/np.pi
scdim_ref=[0,.125,1]+[1.125]*2+[2]*4+[2.125]*3+[3]*5+[3.125]*6+[4]*9+[4.125]*9+[5]*13+[5.125]*14

def troubleshoot(layers:'list[MERALayer]',options:MERAOptions,verbose=0):
    global energy_ref, scdim_ref
    energy=get_energy(layers[0],options).detach().cpu().item()
    energyErr=abs(energy-energy_ref)
    print('Energy', '%.7f'%energy, 'error', '%.2e'%energyErr)
    if verbose>0:
        ops,scdims=get_conformal_ops(layers[-1],k=10)
        print('scdims', scdims.detach().cpu().numpy().round(3))
        if verbose>1:
            print('ref   ', np.array(scdim_ref)[:len(scdims)].round(3))

        opes={}
        for i,j,k in itt.product(range(3),repeat=3):
            opes[(i,j,k)]=get_ope_coeff(layers[-1],[ops[i],ops[j],ops[k]],
                                        [scdims[i],scdims[j],scdims[k]])

        if verbose>1:
            print('ope coeffs')
            for i,j,k in itt.combinations_with_replacement(range(3),3):
                for i1,j1,k1 in sorted(set(itt.permutations([i,j,k]))):
                    print('C_'+str(i1)+str(j1)+str(k1), '%.3f'%opes[(i1,j1,k1)].abs(), end='\t')
                print()
        else:
            print('C_112', '%.3f'%opes[(1,1,2)].abs())




J,g=1,1
hAB,hBA=get_Ising1D_Hamiltonians_2blocked(J,g)
print('J=%f, g=%f'%(J,g))
print('Exact energy=%f'%energy_ref)

filename='./data/mera10layersX12.pth'
logname='./data/mera10layersX12.log'

with redirect_stdout(Tee(logname)):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    optionss=[
        MERAOptions(nLayers=2,max_dim=6,max_dim_mid=4,nSweeps=2000),
        MERAOptions(nLayers=3,max_dim=8,max_dim_mid=6,nSweeps=1800),
        MERAOptions(nLayers=4,max_dim=12,max_dim_mid=8,nSweeps=1400),
        MERAOptions(nLayers=5,max_dim=12,max_dim_mid=8,nSweeps=1400),
        MERAOptions(nLayers=6,max_dim=12,max_dim_mid=8,nSweeps=1400),
        MERAOptions(nLayers=7,max_dim=12,max_dim_mid=8,nSweeps=1400),
        MERAOptions(nLayers=8,max_dim=12,max_dim_mid=8,nSweeps=1400),
        MERAOptions(nLayers=9,max_dim=12,max_dim_mid=8,nSweeps=1400),
        MERAOptions(nLayers=10,max_dim=12,max_dim_mid=8,nSweeps=1400),
    ]
    for ii,options in enumerate(optionss):
        print('sweep using the following options: ')
        print(asdict(options))
        if ii==0:
            layers=init_layers(hAB,hBA,options)
        else:
            pad_layers_(layers,options)
        for i in tqdm(range(1,1+options.nSweeps)):
            options.iSweep=i
            sweep_layers(layers,options)
            if i%(options.nSweeps//10)==0:
                print('Sweep',i)
                troubleshoot(layers,options,verbose=1)
                torch.save((layers,options), filename)
                print('saved to', filename)
                sys.stdout.flush()
        troubleshoot(layers,options,verbose=2)
        torch.save((layers,options), filename)
        print('saved to', filename)
        sys.stdout.flush()