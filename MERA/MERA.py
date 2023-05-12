from opt_einsum import contract
import numpy as np
import torch
from dataclasses import dataclass
from math import prod
import itertools as itt


# reference https://www.tensors.net/mera

@dataclass
class MERAOptions:
    reflection_symmetry:bool=True
    nLayers:int=2
    max_dim_mid:int=4
    max_dim:int=6
    nIterRhoTop:int=4
    nSweeps:int=100
    iSweep:int=0
    freeze_u_before:int=10 # do not update u before step freeze_u_before

@dataclass
class MERALayer:
    '''
    1 2 1     1 1 2 1 2
    uuu www vvv rho hhh
    3 4 2 3 2 3 3 4 3 4

    w     vvv     w
    uuuuuuu |     |
    B     A B     A
    www vvv www vvv
    | uuu | | uuu |
    A B A B A B A B
    
    [rhoBA]  [rhoBA]  [rhoBA]  [rhoAB]
    www vvv  www vvv  www vvv  vvv www
    | uuu |  | uuu |  | uuu |  | | | |
    hAB | |  | hBA |  | | hAB  | hBA |
    [conj ]  [conj ]  [conj ]  [conj ]
    '''
    u: torch.Tensor
    w: torch.Tensor
    v: torch.Tensor
    rhoAB: torch.Tensor=torch.tensor(0.)
    rhoBA: torch.Tensor=torch.tensor(0.)
    hAB: torch.Tensor=torch.tensor(0.)
    hBA: torch.Tensor=torch.tensor(0.)
    h_bias: torch.Tensor=torch.tensor(0.)
    @property
    def d(self)->int:# dimension of the fine-grained space
        return self.w.shape[1]
    @property
    def D_mid(self)->int:# dimension of the fine-grained space after u
        return self.w.shape[2]
    @property
    def D(self)->int:# dimension of the coarse-grained space
        return self.w.shape[0]
    def unpack(layer,dtype=None):
        u,w,v=layer.u,layer.w,layer.v
        rhoAB,rhoBA=layer.rhoAB,layer.rhoBA
        hAB,hBA=layer.hAB,layer.hBA
        uc,wc,vc=torch.conj(u).clone(),torch.conj(w).clone(),torch.conj(v).clone()
        eye=torch.eye(layer.d,device=u.device)
        eyem=torch.eye(layer.D_mid,device=u.device)
        rtval=u,w,v,rhoAB,rhoBA,hAB,hBA,uc,wc,vc,eye,eyem
        if dtype is not None:
            rtval=tuple(t.to(dtype) for t in rtval)
        return rtval

def dcontract(derivative,eq,*tensors,**kwargs):
    assert all(tensor is not None for tensor in tensors)
    assert len(list(tensor for tensor in tensors if id(tensor)==id(derivative)))==1, f'{id(derivative)%3533} {[id(t)%3533 for t in tensors]}'
    idx = next(i for i, tensor in enumerate(tensors) if id(tensor)==id(derivative))
    eq_terms=eq.split(',')
    eq=','.join(eq_terms[:idx]+eq_terms[idx+1:])+'->'+eq_terms[idx]
    tensors=tensors[:idx]+tensors[idx+1:]
    return contract(eq,*tensors,**kwargs)

def symmetrize(T,eqs,signs=None):
    eqs=eqs if isinstance(eqs,(list,tuple)) else [eqs]
    signs=signs if signs is not None else [1 for _ in eqs]
    for eq,sign in zip(eqs,signs):
        T=(T+sign*contract(eq,T))/2
    return T


def get_energy(layer:MERALayer,options:MERAOptions):
    u,w,v,rhoAB,rhoBA,hAB,hBA,uc,wc,vc,eye,eyem=layer.unpack()
    term1=contract('IJij,ikl,jmn,lmop,IKL,JMN,LMOP,koKO,pP,nN',
                    rhoBA,w,v,u,wc,vc,uc,hAB,eye,eye)
    term2=contract('IJij,ikl,jmn,lmop,IKL,JMN,LMOP,kK,opOP,nN',
                    rhoBA,w,v,u,wc,vc,uc,eye,hBA,eye)
    term3=contract('IJij,ikl,jmn,lmop,IKL,JMN,LMOP,kK,oO,pnPN',
                    rhoBA,w,v,u,wc,vc,uc,eye,eye,hAB) \
            if not options.reflection_symmetry else term1
    term4=contract('IJij,ikl,jmn,IKL,JMN,kK,lmLM,nN',
                    rhoAB,v,w,vc,wc,eyem,hBA,eyem)
    return (term1+term2+term3+term4)/4+layer.h_bias


def propogate_down(layer:MERALayer,options:MERAOptions):
    u,w,v,rhoAB,rhoBA,hAB,hBA,uc,wc,vc,eye,eyem=layer.unpack()
    # print('propogate_down')
    # print(rhoBA.shape,w.shape,v.shape,u.shape,hAB.shape)
    rhoAB1=dcontract(hAB,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,koKO,pP,nN',
                    rhoBA,w,v,u,wc,vc,uc,hAB,eye,eye)
    rhoBA1=dcontract(hBA,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,kK,opOP,nN',
                    rhoBA,w,v,u,wc,vc,uc,eye,hBA,eye)
    rhoAB2=dcontract(hAB,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,kK,oO,pnPN',
                    rhoBA,w,v,u,wc,vc,uc,eye,eye,hAB)\
            if not options.reflection_symmetry else contract('ijIJ->jiJI',rhoAB1)
    rhoBA2=dcontract(hBA,'IJij,ikl,jmn,IKL,JMN,kK,lmLM,nN',
                    rhoAB,v,w,vc,wc,eyem,hBA,eyem)
    symmetries=['ijIJ->IJij']+ (['ijIJ->jiJI'] if not options.reflection_symmetry else [])
    new_rhoAB=symmetrize(rhoAB1+rhoAB2,symmetries)
    new_rhoBA=symmetrize(rhoBA1+rhoBA2,symmetries)
    new_rhoAB=new_rhoAB/contract('ijij',new_rhoAB)
    new_rhoBA=new_rhoBA/contract('ijij',new_rhoBA)
    return new_rhoAB,new_rhoBA

def propogate_up(layer:MERALayer,options:MERAOptions):
    u,w,v,rhoAB,rhoBA,hAB,hBA,uc,wc,vc,eye,eyem=layer.unpack()
    # print('propogate_up')
    # print(rhoBA.shape,v.shape,w.shape,vc.shape,wc.shape,hAB.shape)
    hBA1=dcontract(rhoBA,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,koKO,pP,nN',
                    rhoBA,w,v,u,wc,vc,uc,hAB,eye,eye)
    hBA2=dcontract(rhoBA,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,kK,opOP,nN',
                    rhoBA,w,v,u,wc,vc,uc,eye,hBA,eye)
    hBA3=dcontract(rhoBA,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,kK,oO,pnPN',
                    rhoBA,w,v,u,wc,vc,uc,eye,eye,hAB)\
            if not options.reflection_symmetry else contract('ijIJ->jiJI',hBA1)
    hAB1=dcontract(rhoAB,'IJij,ikl,jmn,IKL,JMN,kK,lmLM,nN',
                    rhoAB,v,w,vc,wc,eyem,hBA,eyem)
    symmetries=['ijIJ->IJij']+ (['ijIJ->jiJI'] if not options.reflection_symmetry else [])
    new_hAB=symmetrize(hAB1/2,symmetries)
    new_hBA=symmetrize((hBA1+hBA2+hBA3)/2,symmetries)
    return new_hAB,new_hBA


def get_u_env(layer:MERALayer,options:MERAOptions):
    u,w,v,rhoAB,rhoBA,hAB,hBA,uc,wc,vc,eye,eyem=layer.unpack()
    term1=dcontract(u,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,koKO,pP,nN',
                    rhoBA,w,v,u,wc,vc,uc,hAB,eye,eye)
    term2=dcontract(u,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,kK,opOP,nN',
                    rhoBA,w,v,u,wc,vc,uc,eye,hBA,eye)
    term3=dcontract(u,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,kK,oO,pnPN',
                    rhoBA,w,v,u,wc,vc,uc,eye,eye,hAB)\
            if not options.reflection_symmetry else contract('ijIJ->jiJI',term1)
    return (term1+term2+term3)/4

def get_w_env(layer:MERALayer,options:MERAOptions):
    u,w,v,rhoAB,rhoBA,hAB,hBA,uc,wc,vc,eye,eyem=layer.unpack()
    term1=dcontract(w,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,koKO,pP,nN',
                    rhoBA,w,v,u,wc,vc,uc,hAB,eye,eye)
    term2=dcontract(w,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,kK,opOP,nN',
                    rhoBA,w,v,u,wc,vc,uc,eye,hBA,eye)
    term3=dcontract(w,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,kK,oO,pnPN',
                    rhoBA,w,v,u,wc,vc,uc,eye,eye,hAB)
    term4=dcontract(w,'IJij,ikl,jmn,IKL,JMN,kK,lmLM,nN',
                    rhoAB,v,w,vc,wc,eyem,hBA,eyem)
    return (term1+term2+term3+term4)/4

def get_v_env(layer:MERALayer,options:MERAOptions):
    u,w,v,rhoAB,rhoBA,hAB,hBA,uc,wc,vc,eye,eyem=layer.unpack()
    term1=dcontract(v,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,koKO,pP,nN',
                    rhoBA,w,v,u,wc,vc,uc,hAB,eye,eye)
    term2=dcontract(v,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,kK,opOP,nN',
                    rhoBA,w,v,u,wc,vc,uc,eye,hBA,eye)
    term3=dcontract(v,'IJij,ikl,jmn,lmop,IKL,JMN,LMOP,kK,oO,pnPN',
                    rhoBA,w,v,u,wc,vc,uc,eye,eye,hAB)
    term4=dcontract(v,'IJij,ikl,jmn,IKL,JMN,kK,lmLM,nN',
                    rhoAB,v,w,vc,wc,eyem,hBA,eyem)
    return (term1+term2+term3+term4)/4



def svd_tensor_to_isometry(M,idx1=(0,),idx2=(1,)):
    shape1=tuple(M.shape[i] for i in idx1)
    shape2=tuple(M.shape[i] for i in idx2)
    M=M.permute(idx1+idx2).reshape(prod(shape1),prod(shape2))
    u,_,vh=torch.linalg.svd(M,full_matrices=False)
    uvh=(u@vh).reshape(shape1+shape2).permute(tuple(np.argsort(idx1+idx2)))
    return uvh

def get_isometry_from_environment(M,idx1=(0,),idx2=(1,)):
    return svd_tensor_to_isometry(M,idx1,idx2).conj()

def optimize_layer_(layer:MERALayer,options:MERAOptions):
    if options.reflection_symmetry:
        if options.iSweep>=options.freeze_u_before:
            layer.u=symmetrize(get_isometry_from_environment(symmetrize(get_u_env(layer,options),'ijIJ->jiJI'),[0,1],[2,3]),'ijIJ->jiJI')
        layer.w=get_isometry_from_environment(get_w_env(layer,options),[0],[1,2])
        layer.v=contract('ijk->ikj',layer.w.clone())
    else:
        if options.iSweep>=options.freeze_u_before:
            layer.u=get_isometry_from_environment(get_u_env(layer,options),[0,1],[2,3])
        layer.w=get_isometry_from_environment(get_w_env(layer,options),[0],[1,2])
        layer.v=get_isometry_from_environment(get_v_env(layer,options),[0],[1,2])


def init_layers(hAB,hBA,options:MERAOptions):
    d=hAB.shape[0]
    layers=[]
    # make sure hAB and hBA are negative semidefinite
    h_bias=torch.maximum(
        torch.linalg.eigvalsh(hAB.reshape((d**2,d**2))).max(),
        torch.linalg.eigvalsh(hBA.reshape((d**2,d**2))).max()
    )
    hAB=hAB-h_bias*torch.eye(d**2).reshape((d,d,d,d))
    hBA=hBA-h_bias*torch.eye(d**2).reshape((d,d,d,d))

    # generate the layers
    for i in range(options.nLayers):
        Dm=min(d,options.max_dim_mid)
        D=min(d**2,options.max_dim)
        # here we don't need to generate unitary
        u=torch.eye(Dm**2,d**2).reshape((Dm,Dm,d,d))
        assert D<d*Dm, 'considering blocking sites if the first w, v are unitary'
        w=svd_tensor_to_isometry(torch.randn((D,d,Dm)),[0],[1,2])
        v=svd_tensor_to_isometry(torch.randn((D,Dm,d)),[0,1],[2])
        layers.append(MERALayer(u=u,w=w,v=v,h_bias=h_bias))
        d=D

    layers[0].hAB=hAB.clone()
    layers[0].hBA=hBA.clone()
    layers[-1].rhoAB=torch.eye(D**2).reshape((D,D,D,D))
    layers[-1].rhoBA=torch.eye(D**2).reshape((D,D,D,D))

    return layers

def pad_tensor(A:torch.tensor,shape):
    '''expand tensor dimension by padding with zeros'''
    for k in range(len(shape)):
        if A.shape[k] != shape[k]:
            A=torch.cat([A,torch.zeros(A.shape[:k]+(shape[k]-A.shape[k],)+A.shape[k+1:])],dim=k)
    return A

def pad_layers_(layers:'list[MERALayer]',options:MERAOptions):
    d=layers[0].d
    for i in range(options.nLayers):
        Dm=min(d,options.max_dim_mid)
        D=min(d**2,options.max_dim)
        if i>=len(layers):
            # generate the layers
            layers.append(MERALayer(
                u=layers[-1].u.clone(),
                w=layers[-1].w.clone(),
                v=layers[-1].v.clone(),
                h_bias=layers[-1].h_bias,
                rhoAB=layers[-1].rhoAB.clone(),
                rhoBA=layers[-1].rhoBA.clone(),
                hAB=layers[-1].hAB.clone(),
                hBA=layers[-1].hBA.clone()
            ))
                                    
        # pad the tensors
        layers[i].u=pad_tensor(layers[i].u,(Dm,Dm,d,d))
        layers[i].w=pad_tensor(layers[i].w,(D,d,Dm))
        layers[i].v=pad_tensor(layers[i].v,(D,Dm,d))
        layers[i].rhoAB=pad_tensor(layers[i].rhoAB,(D,D,D,D))
        layers[i].rhoBA=pad_tensor(layers[i].rhoBA,(D,D,D,D))
        layers[i].hAB=pad_tensor(layers[i].hAB,(d,d,d,d))
        layers[i].hBA=pad_tensor(layers[i].hBA,(d,d,d,d))
        d=D
    # reset the density operator at the top layer
    layers[-1].rhoAB=torch.eye(D**2).reshape((D,D,D,D))
    layers[-1].rhoBA=torch.eye(D**2).reshape((D,D,D,D))

def sweep_layers(layers:'list[MERALayer]',options:MERAOptions):
    # determine the scale-invariant density operator
    layer=layers[-1]
    assert layer.D==layer.d
    # use the previous density operator as the initial guess
    for i in range(options.nIterRhoTop):
        layer.rhoAB,layer.rhoBA=propogate_down(layer,options)
    # propogate the density operators down
    for i in range(options.nLayers-1,-1,-1):
        # need to sweep back and forth to ensure convergence
        # in GE's code, he only optimize at the up sweep
        # maybe it will help jumping out local minimum?
        # if len(layers[i].hAB.shape)>0:
        #     optimize_layer_(layers[i],options)
        if i>0:
            layers[i-1].rhoAB,layers[i-1].rhoBA=propogate_down(layers[i],options)
    # propogate the Hamiltonians up
    for i in range(options.nLayers):
        optimize_layer_(layers[i],options)
        if i<options.nLayers-1:
            layers[i+1].hAB,layers[i+1].hBA=propogate_up(layers[i],options)
    options.iSweep+=1


######################################################################
#  here are how to extract conformal operators from the MERA

def renormalize_op(layer,op,scdim):
    '''
    we set renormalized lattice constance = 0.5
    [rhoBA    ]
    [www] [vvv]
    |   [u]   |
    op Id op Id = C(op,op)/(1)^(2 Δ)
    [conj     ]
    '''
    u,w,v,rhoAB,rhoBA,hAB,hBA,uc,wc,vc,eye,eyem=layer.unpack(op.dtype)
    C00=contract('IJij,ikl,jmn,lmop,IKL,JMN,LMOP,kK,oO,pP,nN->',
               rhoBA,w,v,u,wc,vc,uc,op,eye,op,eye)
    return op*C00.abs()**-.5


def get_conformal_ops(layer,k=10):
    '''
      |
    vvv
    | www
    | O |
    [conj]
    '''
    assert layer.d==layer.D
    w,v,wc,vc=layer.w,layer.v,torch.conj(layer.w),torch.conj(layer.v)
    d=layer.d
    M=contract('ijk,klm,IjK,KLm->iIlL',v,w,vc,wc)
    s,u=torch.linalg.eig(M.reshape((d**2,d**2)))
    s,u=s[torch.argsort(s.abs(),descending=True)],u[:,torch.argsort(s.abs(),descending=True)]
    scdims=torch.log2(s[:k].abs()).abs()/2
    ops=[u[:,i].reshape((d,d)) for i in range(k)]
    ops=[renormalize_op(layer,op,scdim) for op,scdim in zip(ops,scdims)]
    return ops,scdims



def get_ope_coeff(layer,ops,scdims):
    '''
    [vvv] [www] [vvv] [www]
    Id Id o0 Id o1 Id o2 id = C(o0,o1,o2) / 2^(Δ0+Δ2-Δ1)
    '''
    u,w,v,rhoAB,rhoBA,hAB,hBA,uc,wc,vc,eye,eyem=layer.unpack(ops[0].dtype)
    op01_level1=contract('ikl,jmn,lmop,IKL,JMN,LMOP,kK,oO,pP,nN->ijIJ',
                    w,v,u,wc,vc,uc,ops[0],eye,ops[1],eye)
    op2_level1=contract('ijk,IJK,jJ,kK->iI',
                    w,wc,ops[2],eyem) # we also need to propogate op2 to the next layer
    expval=contract('IJij,ikl,jmn,lmop,IKL,JMN,LMOP,kK,opOP,nN->',
                    rhoBA,w,v,u,wc,vc,uc,eye,op01_level1,op2_level1)
    C012=expval*2**(scdims[0]-scdims[1]+scdims[2])
    return C012
    
    
