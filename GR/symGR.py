from sympy import *
from opt_einsum import contract as _contract
import numpy as np
from collections import namedtuple

def contract(expr,*tensors,**args):
    if contract.dest is not None:
        expr+='->'+contract.dest
    rtval= _contract(expr,*tensors, backend='object',**args)
    return rtval.item() if rtval.shape==() else Array(rtval)
class contract_dest:
    def __init__(self, dest):
        self.dest= dest
    def __enter__(self):
        contract.dest,self.dest= self.dest,contract.dest
    def __exit__(self, *args):
        contract.dest= self.dest
contract.dest=None



class CoordinateTransformation:
    def __init__(self,coords1:list,coords2:list,coords2_in_1:dict,coords1_in_2:dict=None):
        self.coords1=coords1
        self.coords2=coords2
        self.coords2_in_1=coords2_in_1
        if coords1_in_2 is None:
            subs1_in_2=solve([a-b for a,b in coords2_in_1.items()],coords1,dict=True)
            coords1_in_2={x:x for x in coords1}
            print('try to solve for coords1_in_2')
            print(subs1_in_2)
            if len(subs1_in_2)>1:
                print('Warning: more than one solution found, use the first one')
            coords1_in_2.update(subs1_in_2[0])
        self.coords1_in_2=coords1_in_2
        self.jacobi=Matrix(together(derive_by_array(Array(coords2).subs(coords2_in_1),coords1))).T
        self.jacobi_inv=Matrix(together(derive_by_array(Array(coords1).subs(coords1_in_2),coords2))).T
    def transform(self,tensor,indice_positions='',inverse=False):
        if not inverse:
            jacobi,jacobi_inv,subs=self.jacobi,self.jacobi_inv,self.coords1_in_2
        else:
            jacobi,jacobi_inv,subs=self.jacobi_inv,self.jacobi,self.coords2_in_1
        for i,ind in enumerate(indice_positions):
            if ind=='u':
                # V^M=J^M_u v^u
                tensor=np.swapaxes(contract('Mu,u...->M...',jacobi,np.swapaxes(tensor,0,i)),0,i)
            elif ind=='d':
                # V_M=v_u (J^-1)^u_M
                tensor=np.swapaxes(contract('uM,u...->M...',jacobi_inv,np.swapaxes(tensor,0,i)),0,i)
        if isinstance(tensor,np.ndarray):
            tensor=Array(tensor)
        tensor=tensor.subs(subs)
        tensor=together(tensor)
        return tensor
    def inverse_transform(self,*args,**kwargs):
        return self.transform(*args,inverse=True,**kwargs)
    
def get_lie_derivative(coords):
    def lie_derivative(f,X,indice_positions=''):
        dX=derive_by_array(X,coords)
        # L_X f = X^u \partial_u f
        rtval=contract('i,i...->...',X,derive_by_array(f,coords))
        for i,ind in enumerate(indice_positions):
            if ind=='u':
                # L_X f^u -= (∂_v X^u) f^v
                rtval-=np.swapaxes(contract('vu,v...->u...',dX,np.swapaxes(f,0,i)),0,i)
            elif ind=='d':
                # L_X f_u += (∂_u X^v) f_v
                rtval+=np.swapaxes(contract('uv,v...->u...',dX,np.swapaxes(f,0,i)),0,i)
            elif ind=='U' or ind=='D':
                raise NotImplementedError
        if isinstance(rtval,np.ndarray):
            rtval=Array(rtval)
        return together(rtval)
    def lie_bracket(X,Y):
        return lie_derivative(Y,X,'u')
    return lie_derivative,lie_bracket

def get_covariant_derivative(coords, Gm=None, w=None):
    def covariant_derivative(v,indice_positions=''):
        ''' a: tensor, indice_positions: list of index type
            index types:
                - 'i': internal, 
                - 'u': curved upper, 'd' curved lower,
                - 'U': flat upper, 'D' flat lower
        '''
        rtval=derive_by_array(v,coords)
        for i,ind in enumerate(indice_positions):
            if ind=='u': # D_u T^v=∂_u T^v+Γ^v_ur T^r
                rtval+=np.swapaxes(contract('vur,r...->uv...',Gm,np.swapaxes(v,0,i)),1,i+1)
            elif ind=='d': # D_u T_v=∂_u T_v-Γ^r_uv T^r
                rtval-=np.swapaxes(contract('ruv,r...->uv...',Gm,np.swapaxes(v,0,i)),1,i+1)
            elif ind=='U': # D_u T^m=∂_u T^m+w_u^m_n T^n
                assert False, 'not tested!'
                rtval+=np.swapaxes(contract('umn,n...->um...',w,np.swapaxes(v,0,i)),1,i+1)
            elif ind=='D': # D_u T_m=∂_u T_m-w_u^n_m T_n
                assert False, 'not tested!'
                rtval-=np.swapaxes(contract('unm,n...->um...',w,np.swapaxes(v,0,i)),1,i+1)
        return Array(simplify(rtval))
    return covariant_derivative


    
def Christoffel_Symbols(g,coords):
    '''Γ^k_ij=1/2 g^kl (∂_i g_jl+∂_j g_il-∂_l g_ij)'''
    ginv=g.inv()
    dg=derive_by_array(g,coords)
    with contract_dest('kij'):
        Gm=(contract('kl,ijl',ginv,dg)+contract('kl,jil',ginv,dg)-contract('kl,lij',ginv,dg))/2
    return together(Gm)

def Riemann_Tensor(Gm,coords):
    '''R^r_smn=∂m Γ^r_ns - ∂n Γ^r_ms +Γ^r_ml Γ^l_ns - Γ^r_nl Γ^l_ms'''
    dGm=derive_by_array(Gm,coords)
    with contract_dest('rsmn'):
        Rie=contract('mrns',dGm)-contract('nrms',dGm)+contract('rml,lns',Gm,Gm)-contract('rnl,lms',Gm,Gm)
    return together(Rie)

def Ricci_Tensor(Rie):
    '''R_ij=R^r_irj'''
    with contract_dest('ij'):
        Ric=contract('rirj',Rie)
    return together(Ric)

def Ricci_Scalar(Ric,g):
    '''R=R_ij g^ij'''
    RicS=contract('ij,ij',Ric,g.inv())
    return expand(RicS)

def Einstein_Tensor(Ric,g):
    '''G_ij=R_ij-1/2 g_ij R'''
    RicS=Ricci_Scalar(Ric,g)
    Ein=Ric+Array(g)*(-RicS/2)
    return Ein.applyfunc(expand)