#!/usr/bin/python
""" Classes and functions for generating noddi kernels """
from __future__ import division, print_function, absolute_import

import warnings

from scipy.special import erf, gammainc
import numpy as np

from ..core.geometry import vector_norm

"""import scipy.optimize as opt

from dipy.utils.six.moves import range
from dipy.data import get_sphere
from ..core.gradients import gradient_table
from ..core.sphere import Sphere
from .vec_val_sum import vec_val_vect
from ..core.onetime import auto_attr
from .base import ReconstModel, ReconstFit"""

import scipy
import numpy.matlib as matlib

def GetParameterStrings(modelname):
    if modelname == 'StickIsoV_B0':
        strings = ['ficvf', 'di', 'dh', 'fiso', 'diso', 'b0', 'theta', 'phi']
    elif modelname ==  'StickTortIsoV_B0':
        strings = ['ficvf', 'di', 'fiso', 'diso', 'b0', 'theta', 'phi']
    elif modelname ==  'WatsonSHStick':
        strings = ['ficvf', 'di', 'dh', 'kappa', 'theta', 'phi']
    elif modelname ==  'WatsonSHStickIsoV_B0':
        strings = ['ficvf', 'di', 'dh', 'kappa', 'fiso', 'diso', 'b0', 'theta', 'phi']
    elif modelname ==  'WatsonSHStickIsoVIsoDot_B0':
        strings = ['ficvf', 'di', 'dh', 'kappa', 'fiso', 'diso', 'irfrac', 'b0', 'theta', 'phi']
    elif modelname ==  'WatsonSHStickTortIsoV_B0':
        strings = ['ficvf', 'di', 'kappa', 'fiso', 'diso', 'b0', 'theta', 'phi']
    elif modelname ==  'WatsonSHStickTortIsoVIsoDot_B0':
        strings = ['ficvf', 'di', 'kappa', 'fiso', 'diso', 'irfrac', 'b0', 'theta', 'phi']
    elif modelname ==  'BinghamStickTortIsoV_B0':
        strings = ['ficvf', 'di', 'kappa', 'beta', 'psi', 'fiso', 'diso', 'b0', 'theta', 'phi']
    elif modelname ==  'WatsonSHCylSingleRadTortIsoV_GPD_B0':
        strings = ['ficvf', 'di', 'rad', 'kappa', 'fiso', 'diso', 'b0', 'theta', 'phi']
    elif modelname ==  'CylSingleRadIsoDotTortIsoV_GPD_B0':
        strings = ['ficvf', 'di', 'rad', 'irfrac', 'fiso', 'diso', 'b0', 'theta', 'phi']
    else:
        msg = 'Parameter strings yet to be defined for this model: %s' % modelname
        raise ValueError(msg)
    return strings

def GetParameterIndex(modelname, parametername):
    strings = GetParameterStrings(modelname)

    for i in range(len(strings)):
        if strings[i] == parametername :
            idx = i
            return idx    
    return -1

def NumFreeParams(modelName):
    return len(GetParameterStrings(modelName))

def MakeModel(modelname):
    model = {}
    model['name'] = modelname
    model['numParams'] = NumFreeParams(modelname)
    model['tissuetype'] = 'invivo'
    model['GS'] = {'fixed': np.zeros((model['numParams'])),'fixedvals':np.zeros((model['numParams']))}
    model['GD'] = {'fixed' : np.zeros((model['numParams'])),'fixedvals':np.zeros((model['numParams'])),'type':'single','multistart':{'perturbation':np.zeros((model['numParams'])),'noOfRuns':10}}
    model['MCMC'] = {'fixed':np.ones((model['numParams'] + 1)),'steplengths':0.05*np.ones((model['numParams']+1)),'burnin':2000,'interval':200,'samples':40}
    model['noOfStages'] = 2
    model['sigma'] = {'perVoxel':1,'minSNR':0.02,'scaling':100}
    
    irfracIdx = GetParameterIndex(modelname, 'irfrac')
    if irfracIdx > 0:
        model['tissuetype'] = 'exvivo'
        
    # fix intrinsic diffusivity
    diIdx = GetParameterIndex(modelname, 'di')
    model['GS']['fixed'][diIdx] = 1
    model['GD']['fixed'][diIdx] = 1
    if model['tissuetype'] == 'invivo':
        model['GS']['fixedvals'][diIdx] = 1.7E-9
        model['GD']['fixedvals'][diIdx] = 1.7E-9
    else:
        model['GS']['fixedvals'][diIdx] = 0.6E-9
        model['GD']['fixedvals'][diIdx] = 0.6E-9
        
    # fix isotropic diffusivity
    disoIdx = GetParameterIndex(modelname, 'diso')
    if disoIdx > 0:
        model['GS']['fixed'][disoIdx] = 1
        model['GD']['fixed'][disoIdx] = 1
        if model['tissuetype'] == 'invivo':
            model['GS']['fixedvals'][disoIdx] = 3.0E-9
            model['GD']['fixedvals'][disoIdx] = 3.0E-9
        else:
            model['GS']['fixedvals'][disoIdx] = 2.0E-9
            model['GD']['fixedvals'][disoIdx] = 2.0E-9
            
    # fix B0
    # fixed value is estimated from the b0 images voxel-wise
    b0Idx = GetParameterIndex(modelname, 'b0')
    if b0Idx > 0:
        model['GS']['fixed'][b0Idx] = 1
        model['GD']['fixed'][b0Idx] = 1
    
    return model

def scheme2noddi(scheme):
    protocol = {}
    protocol['pulseseq'] = 'PGSE'
    protocol['schemetype'] = 'multishellfixedG'
    protocol['teststrategy'] = 'fixed'
    bval = scheme.bvals.copy()
    
    # set total number of measurements
    protocol['totalmeas'] = len(bval)
    
    # set the b=0 indices
    protocol['b0_Indices'] = bval==0
    protocol['numZeros'] = sum(protocol['b0_Indices'])
    
    # find the unique non-zero b-values
    B = np.unique(bval[bval>0])
    
    # set the number of shells
    protocol['M'] = len(B)
    protocol['N'] = np.zeros((len(B)))
    for i in range(len(B)):
        protocol['N'][i] = sum(bval==B[i])
    
    # maximum b-value in the s/mm^2 unit
    maxB = np.max(B)
    
    # set maximum G = 40 mT/m
    Gmax = 0.04
    
    # set smalldel and delta and G
    GAMMA = 2.675987E8
    tmp = np.power(3*maxB*1E6/(2*GAMMA*GAMMA*Gmax*Gmax),1.0/3.0)
    protocol['udelta'] = np.zeros((len(B)))
    protocol['usmalldel'] = np.zeros((len(B)))
    protocol['uG'] = np.zeros((len(B)))
    for i in range(len(B)):
        protocol['udelta'][i] = tmp
        protocol['usmalldel'][i] = tmp
        protocol['uG'][i] = np.sqrt(B[i]/maxB)*Gmax
    
    protocol['delta'] = np.zeros(bval.shape)
    protocol['smalldel'] = np.zeros(bval.shape)
    protocol['G'] = np.zeros(bval.shape)
    
    for i in range(len(B)):
        tmp = np.nonzero(bval==B[i])
        for j in range(len(tmp[0])):
            protocol['delta'][tmp[0][j]] = protocol['udelta'][i]
            protocol['smalldel'][tmp[0][j]] = protocol['usmalldel'][i]
            protocol['G'][tmp[0][j]] = protocol['uG'][i]
    
    # load bvec
    protocol['grad_dirs'] = scheme.gradients.copy()
    
    # make the gradient directions for b=0's [1 0 0]
    for i in range(len(protocol['b0_Indices'])):
        protocol['grad_dirs'][protocol['b0_Indices'][i],:] = [1, 0, 0]
    
    # make sure the gradient directions are unit vectors
    for i in range(protocol['totalmeas']):
        protocol['grad_dirs'][i,:] = protocol['grad_dirs'][i,:]/vector_norm(protocol['grad_dirs'][i,:])
        
    return protocol

def CylNeumanLePar_PGSE(d, G, delta, smalldel):
    # Line bellow used in matlab version removed as CylNeumanLePar_PGSE is called from SynthMeasWatsonSHCylNeuman_PGSE which already casts x to d, R and kappa -> x replaced by d in arguments
    #d=x[0]
    
    # Radial wavenumbers
    GAMMA = 2.675987E8
    modQ = GAMMA*smalldel*G
    modQ_Sq = modQ*modQ
    
    # diffusion time for PGSE, in a matrix for the computation below.
    difftime = (delta-smalldel/3)
    
    # Parallel component
    LE =-modQ_Sq*difftime*d
    
    # Compute the Jacobian matrix
    #if(nargout>1)
    #    % dLE/d
    #    J = -modQ_Sq*difftime
    #end
    
    return LE

def CylNeumanLePerp_PGSE(d, R, G, delta, smalldel, roots):
    
    # When R=0, no need to do any calculation
    if (R == 0.00):
        LE = np.zeros(G.shape) # np.size(R) = 1
        return LE
    else:
        msg = "Python implementation for function dipy.reconst.noddi.CylNeumanLePerp_PGSE not yet validated"
        raise ValueError(msg)
    
    """
    # Check the roots array is correct
    if np.abs(roots[0] - 1.8412)>0.0001:
        msg = 'Looks like the roots array is wrong.  First value should be 1.8412, but is %f' % roots[0]
        raise ValueError(msg)
    
    # Radial wavenumbers
    GAMMA = 2.675987E8
    
    # number of gradient directions, i.e. number of measurements
    l_q=G.shape[0]
    l_a=len(R)
    k_max=len(roots)
    
    R_mat=matlib.repmat(R,l_q, 1)
    R_mat=R_mat.reshape(-1)
    R_mat=matlib.repmat(R_mat,1,1, k_max)
    R_matSq=np.power(R_mat,2)
    
    root_m=np.reshape(roots,(1, 1, k_max))
    alpha_mat=matlib.repmat(root_m,l_q*l_a, 1, 1)/R_mat
    amSq=np.power(alpha_mat,2)
    amP6=np.power(amSq,3)
    
    deltamx=matlib.repmat(delta,1,l_a)
    deltamx_rep = deltamx.reshape(-1)
    deltamx_rep = matlib.repmat(deltamx_rep,(1, 1, k_max))
    
    smalldelmx=matlib.repmat(smalldel,(1,l_a))
    smalldelmx_rep = smalldelmx.reshape(-1)
    smalldelmx_rep = matlib.repmat(smalldelmx_rep,(1, 1, k_max))
    
    Gmx=matlib.repmat(G,(1,l_a))
    GmxSq = np.power(Gmx,2)
    
    # Perpendicular component (Neuman model)
    sda2 = smalldelmx_rep*amSq
    bda2 = deltamx_rep*amSq
    emdsda2 = np.exp(-d*sda2)
    emdbda2 = np.exp(-d*bda2)
    emdbdmsda2 = np.exp(-d*(bda2 - sda2))
    emdbdpsda2 = np.exp(-d*(bda2 + sda2))
    
    sumnum1 = 2*d*sda2
    # the rest can be reused in dE/dR
    sumnum2 = - 2 + 2*emdsda2 + 2*emdbda2
    sumnum2 = sumnum2 - emdbdmsda2 - emdbdpsda2
    sumnum = sumnum1 + sumnum2
    
    sumdenom = np.power(d,2)*amP6*(R_matSq*amSq - 1)
    
    # Check for zeros on top and bottom
    #sumdenom(find(sumnum) == 0) = 1;
    sumterms = sumnum/sumdenom
    
    testinds = sumterms[:,:,-1]>0
    test = sumterms[testinds,0]/sumterms[testinds,-1]
    if(np.min(test)<1E4):
        warnings('Ratio of largest to smallest terms in Neuman model sum is <1E4.  May need more terms.')
    
    s = np.sum(sumterms,2)
    s = np.reshape(s,(l_q,l_a))
    if(np.min(s)<0):
        warnings('Negative sums found in Neuman sum.  Setting to zero.')
        s[s<0]=0
    
    LE = -2*np.power(GAMMA,2)*GmxSq*s
    
    return LE"""

def LegendreGaussianIntegral(Lpmp, n):
    if n > 6:
        msg = 'The maximum value for n is 6, which correspondes to the 12th order Legendre polynomial'
        raise ValueError(msg)
    exact = Lpmp>0.05
    approx = Lpmp<=0.05
        
    mn = n + 1
    
    I = np.zeros((len(Lpmp),mn))
    sqrtx = np.sqrt(Lpmp[exact])
    I[exact,0] = np.sqrt(np.pi)*scipy.special.erf(sqrtx)/sqrtx
    dx = 1.0/Lpmp[exact]
    emx = -np.exp(-Lpmp[exact])
    for i in range(1,mn):
        I[exact,i] = emx + (i-0.5)*I[exact,i-1]
        I[exact,i] = I[exact,i]*dx
    
    # Computing the legendre gaussian integrals for large enough Lpmp
    L = np.zeros((len(Lpmp),n+1))
    for i in range(n+1):
        if i == 0:
            L[exact,0] = I[exact,0]
        elif i == 1:
            L[exact,1] = -0.5*I[exact,0] + 1.5*I[exact,1]
        elif i == 2:
            L[exact,2] = 0.375*I[exact,0] - 3.75*I[exact,1] + 4.375*I[exact,2]
        elif i == 3:
            L[exact,3] = -0.3125*I[exact,0] + 6.5625*I[exact,1] - 19.6875*I[exact,2] + 14.4375*I[exact,3]
        elif i == 4:
            L[exact,4] = 0.2734375*I[exact,0] - 9.84375*I[exact,1] + 54.140625*I[exact,2] - 93.84375*I[exact,3] + 50.2734375*I[exact,4]
        elif i == 5:
            L[exact,5] = -(63./256.)*I[exact,0] + (3465./256.)*I[exact,1] - (30030./256.)*I[exact,2] + (90090./256.)*I[exact,3] - (109395./256.)*I[exact,4] + (46189./256.)*I[exact,5]
        elif i == 6:
            L[exact,6] = (231./1024.)*I[exact,0] - (18018./1024.)*I[exact,1] + (225225./1024.)*I[exact,2] - (1021020./1024.)*I[exact,3] + (2078505./1024.)*I[exact,4] - (1939938./1024.)*I[exact,5] + (676039./1024.)*I[exact,6]
    
    # Computing the legendre gaussian integrals for small Lpmp
    x2=np.power(Lpmp[approx],2)
    x3=x2*Lpmp[approx]
    x4=x3*Lpmp[approx]
    x5=x4*Lpmp[approx]
    x6=x5*Lpmp[approx]
    for i in range(n+1):
        if i == 0:
            L[approx,0] = 2 - 2*Lpmp[approx]/3 + x2/5 - x3/21 + x4/108
        elif i == 1:
            L[approx,1] = -4*Lpmp[approx]/15 + 4*x2/35 - 2*x3/63 + 2*x4/297
        elif i == 2:
            L[approx,2] = 8*x2/315 - 8*x3/693 + 4*x4/1287
        elif i == 3:
            L[approx,3] = -16*x3/9009 + 16*x4/19305
        elif i == 4:
            L[approx,4] = 32*x4/328185
        elif i == 5:
            L[approx,5] = -64*x5/14549535
        elif i == 6:
            L[approx,6] = 128*x6/760543875
    
    return L

#def erfi(x):
#    return not(np.isreal(x))*(-(cnp.sqrt(-np.power(x,2))/(x+np.isreal(x)))*gammainc(-np.power(x,2),1/2))+np.isreal(x)*np.real(-np.sqrt(-1)*np.sign(x)*((x<5.7)*gammainc(-np.power(x,2),1/2))+(x>=5.7)*np.exp(np.power(x,2))/x/np.sqrt(np.pi))

def WatsonSHCoeff(kappa):
    """
    # Implementation for multiple concentration parameters needs to be debuged before making it available
    # Originaly designed to compute the SH coefficients of a Watson distribution with multiple concentration parameters
    # kappa should be a column vector
    if kappa.shape[1] != 1:
        msg = 'dipy/reconst/noddi.py: WatsonSHCoeff(kappa) needs kappa to be a column vector.'
        raise ValueError(msg)
    large = kappa>30
    exact = kappa>0.1
    approx = kappa<=0.1
    
    # The maximum order of SH coefficients (2n)
    n = 6
    
    # Computing the SH coefficients
    C = np.zeros((len(kappa),n+1))
    
    # 0th order is a constant
    C[:,0] = 2*np.sqrt(np.pi)
    
    # Precompute the special function values
    sk = np.sqrt(kappa[exact])
    sk2 = sk*kappa[exact]
    sk3 = sk2*kappa[exact]
    sk4 = sk3*kappa[exact]
    sk5 = sk4*kappa[exact]
    sk6 = sk5*kappa[exact]
    sk7 = sk6*kappa[exact]
    k2 = np.power(kappa,2)
    k3 = k2*kappa
    k4 = k3*kappa
    k5 = k4*kappa
    k6 = k5*kappa
    k7 = k6*kappa
    
    erfik = scipy.special.erfi(sk)
    ierfik = 1/erfik
    ek = np.exp(kappa[exact])
    dawsonk = 0.5*np.sqrt(np.pi)*erfik/ek
    
    # for large enough kappa
    C[exact,1] = 3*sk - (3 + 2*kappa[exact])*dawsonk
    C[exact,1] = np.sqrt(5)*C[exact,1]*ek
    C[exact,1] = C[exact,1]*ierfik/kappa[exact]
    
    C[exact,2] = (105 + 60*kappa[exact] + 12*k2[exact])*dawsonk
    C[exact,2] = C[exact,2] -105*sk + 10*sk2
    C[exact,2] = .375*C[exact,2]*ek/k2[exact]
    C[exact,2] = C[exact,2]*ierfik
    
    C[exact,3] = -3465 - 1890*kappa[exact] - 420*k2[exact] - 40*k3[exact]
    C[exact,3] = C[exact,3]*dawsonk
    C[exact,3] = C[exact,3] + 3465*sk - 420*sk2 + 84*sk3
    C[exact,3] = C[exact,3]*np.sqrt(13*np.pi)/64/k3[exact]
    C[exact,3] = C[exact,3]/dawsonk
    
    C[exact,4] = 675675 + 360360*kappa[exact] + 83160*k2[exact] + 10080*k3[exact] + 560*k4[exact]
    C[exact,4] = C[exact,4]*dawsonk
    C[exact,4] = C[exact,4] - 675675*sk + 90090*sk2 - 23100*sk3 + 744*sk4
    C[exact,4] = np.sqrt(17)*C[exact,4]*ek
    C[exact,4] = C[exact,4]/512/k4[exact]
    C[exact,4] = C[exact,4]*ierfik
    
    C[exact,5] = -43648605 - 22972950*kappa[exact] - 5405400*k2[exact] - 720720*k3[exact] - 55440*k4[exact] - 2016*k5[exact]
    C[exact,5] = C[exact,5]*dawsonk
    C[exact,5] = C[exact,5] + 43648605*sk - 6126120*sk2 + 1729728*sk3 - 82368*sk4 + 5104*sk5
    C[exact,5] = np.sqrt(21*np.pi)*C[exact,5]/4096/k5[exact]
    C[exact,5] = C[exact,5]/dawsonk
    
    C[exact,6] = 7027425405 + 3666482820*kappa[exact] + 872972100*k2[exact] + 122522400*k3[exact]  + 10810800*k4[exact] + 576576*k5[exact] + 14784*k6[exact]
    C[exact,6] = C[exact,6]*dawsonk
    C[exact,6] = C[exact,6] - 7027425405*sk + 1018467450*sk2 - 302630328*sk3 + 17153136*sk4 - 1553552*sk5 + 25376*sk6
    C[exact,6] = 5*C[exact,6]*ek
    C[exact,6] = C[exact,6]/16384/k6[exact]
    C[exact,6] = C[exact,6]*ierfik
    
    # for very large kappa
    if large.sum() > 0:
        lnkd = np.log(kappa[large]) - np.log(30)
        lnkd2 = lnkd*lnkd
        lnkd3 = lnkd2*lnkd
        lnkd4 = lnkd3*lnkd
        lnkd5 = lnkd4*lnkd
        lnkd6 = lnkd5*lnkd
        C[large,1] = 7.52308 + 0.411538*lnkd - 0.214588*lnkd2 + 0.0784091*lnkd3 - 0.023981*lnkd4 + 0.00731537*lnkd5 - 0.0026467*lnkd6
        C[large,2] = 8.93718 + 1.62147*lnkd - 0.733421*lnkd2 + 0.191568*lnkd3 - 0.0202906*lnkd4 - 0.00779095*lnkd5 + 0.00574847*lnkd6
        C[large,3] = 8.87905 + 3.35689*lnkd - 1.15935*lnkd2 + 0.0673053*lnkd3 + 0.121857*lnkd4 - 0.066642*lnkd5 + 0.0180215*lnkd6
        C[large,4] = 7.84352 + 5.03178*lnkd - 1.0193*lnkd2 - 0.426362*lnkd3 + 0.328816*lnkd4 - 0.0688176*lnkd5 - 0.0229398*lnkd6
        C[large,5] = 6.30113 + 6.09914*lnkd - 0.16088*lnkd2 - 1.05578*lnkd3 + 0.338069*lnkd4 + 0.0937157*lnkd5 - 0.106935*lnkd6
        C[large,6] = 4.65678 + 6.30069*lnkd + 1.13754*lnkd2 - 1.38393*lnkd3 - 0.0134758*lnkd4 + 0.331686*lnkd5 - 0.105954*lnkd6
    
    # for small kappa
    C[approx,1] = 4/3*kappa[approx] + 8/63*k2[approx]
    C[approx,1] = C[approx,1]*np.sqrt(np.pi/5)
    
    C[approx,2] = 8/21*k2[approx] + 32/693*k3[approx]
    C[approx,2] = C[approx,2]*(np.sqrt(np.pi)*0.2)
    
    C[approx,3] = 16/693*k3[approx] + 32/10395*k4[approx]
    C[approx,3] = C[approx,3]*np.sqrt(np.pi/13)
    
    C[approx,4] = 32/19305*k4[approx]
    C[approx,4] = C[approx,4]*np.sqrt(np.pi/17)
    
    C[approx,5] = 64*np.sqrt(np.pi/21)*k5[approx]/692835
    
    C[approx,6] = 128*np.sqrt(np.pi)*k6[approx]/152108775
    
    return C
    """
    if isinstance(kappa,np.ndarray):
        msg = 'dipy/reconst/noddi.py : WatsonSHcoeff() not implemented for kappa array input yet.'
        raise ValueError(msg)
        
    # In the scope of AMICO only a single value is used for kappa
    n = 6
    
    C = np.zeros((n+1))
    # 0th order is a constant
    C[0] = 2*np.sqrt(np.pi)
    
    # Precompute the special function values
    sk = np.sqrt(kappa)
    sk2 = sk*kappa
    sk3 = sk2*kappa
    sk4 = sk3*kappa
    sk5 = sk4*kappa
    sk6 = sk5*kappa
    sk7 = sk6*kappa
    k2 = np.power(kappa,2)
    k3 = k2*kappa
    k4 = k3*kappa
    k5 = k4*kappa
    k6 = k5*kappa
    k7 = k6*kappa
    
    erfik = scipy.special.erfi(sk)
    ierfik = 1/erfik
    ek = np.exp(kappa)
    dawsonk = 0.5*np.sqrt(np.pi)*erfik/ek
    
    if kappa > 0.1:
    
        # for large enough kappa
        C[1] = 3*sk - (3 + 2*kappa)*dawsonk
        C[1] = np.sqrt(5)*C[1]*ek
        C[1] = C[1]*ierfik/kappa
        
        C[2] = (105 + 60*kappa + 12*k2)*dawsonk
        C[2] = C[2] -105*sk + 10*sk2
        C[2] = .375*C[2]*ek/k2
        C[2] = C[2]*ierfik
        
        C[3] = -3465 - 1890*kappa - 420*k2 - 40*k3
        C[3] = C[3]*dawsonk
        C[3] = C[3] + 3465*sk - 420*sk2 + 84*sk3
        C[3] = C[3]*np.sqrt(13*np.pi)/64/k3
        C[3] = C[3]/dawsonk
        
        C[4] = 675675 + 360360*kappa + 83160*k2 + 10080*k3 + 560*k4
        C[4] = C[4]*dawsonk
        C[4] = C[4] - 675675*sk + 90090*sk2 - 23100*sk3 + 744*sk4
        C[4] = np.sqrt(17)*C[4]*ek
        C[4] = C[4]/512/k4
        C[4] = C[4]*ierfik
        
        C[5] = -43648605 - 22972950*kappa - 5405400*k2 - 720720*k3 - 55440*k4 - 2016*k5
        C[5] = C[5]*dawsonk
        C[5] = C[5] + 43648605*sk - 6126120*sk2 + 1729728*sk3 - 82368*sk4 + 5104*sk5
        C[5] = np.sqrt(21*np.pi)*C[5]/4096/k5
        C[5] = C[5]/dawsonk
        
        C[6] = 7027425405 + 3666482820*kappa + 872972100*k2 + 122522400*k3  + 10810800*k4 + 576576*k5 + 14784*k6
        C[6] = C[6]*dawsonk
        C[6] = C[6] - 7027425405*sk + 1018467450*sk2 - 302630328*sk3 + 17153136*sk4 - 1553552*sk5 + 25376*sk6
        C[6] = 5*C[6]*ek
        C[6] = C[6]/16384/k6
        C[6] = C[6]*ierfik
    
    # for very large kappa
    if kappa>30:
        lnkd = np.log(kappa) - np.log(30)
        lnkd2 = lnkd*lnkd
        lnkd3 = lnkd2*lnkd
        lnkd4 = lnkd3*lnkd
        lnkd5 = lnkd4*lnkd
        lnkd6 = lnkd5*lnkd
        C[1] = 7.52308 + 0.411538*lnkd - 0.214588*lnkd2 + 0.0784091*lnkd3 - 0.023981*lnkd4 + 0.00731537*lnkd5 - 0.0026467*lnkd6
        C[2] = 8.93718 + 1.62147*lnkd - 0.733421*lnkd2 + 0.191568*lnkd3 - 0.0202906*lnkd4 - 0.00779095*lnkd5 + 0.00574847*lnkd6
        C[3] = 8.87905 + 3.35689*lnkd - 1.15935*lnkd2 + 0.0673053*lnkd3 + 0.121857*lnkd4 - 0.066642*lnkd5 + 0.0180215*lnkd6
        C[4] = 7.84352 + 5.03178*lnkd - 1.0193*lnkd2 - 0.426362*lnkd3 + 0.328816*lnkd4 - 0.0688176*lnkd5 - 0.0229398*lnkd6
        C[5] = 6.30113 + 6.09914*lnkd - 0.16088*lnkd2 - 1.05578*lnkd3 + 0.338069*lnkd4 + 0.0937157*lnkd5 - 0.106935*lnkd6
        C[6] = 4.65678 + 6.30069*lnkd + 1.13754*lnkd2 - 1.38393*lnkd3 - 0.0134758*lnkd4 + 0.331686*lnkd5 - 0.105954*lnkd6
    
    if kappa <= 0.1:
        # for small kappa
        C[1] = 4/3*kappa + 8/63*k2
        C[1] = C[1]*np.sqrt(np.pi/5)
        
        C[2] = 8/21*k2 + 32/693*k3
        C[2] = C[2]*(np.sqrt(np.pi)*0.2)
        
        C[3] = 16/693*k3 + 32/10395*k4
        C[3] = C[3]*np.sqrt(np.pi/13)
        
        C[4] = 32/19305*k4
        C[4] = C[4]*np.sqrt(np.pi/17)
        
        C[5] = 64*np.sqrt(np.pi/21)*k5/692835
        
        C[6] = 128*np.sqrt(np.pi)*k6/152108775
    
    return C   


def SynthMeasWatsonSHCylNeuman_PGSE(x, grad_dirs, G, delta, smalldel, fibredir, roots):
    d=x[0]
    R=x[1]
    kappa=x[2]
    
    l_q = grad_dirs.shape[0]
    
    # Parallel component
    LePar = CylNeumanLePar_PGSE(d, G, delta, smalldel)
    
    # Perpendicular component
    LePerp = CylNeumanLePerp_PGSE(d, R, G, delta, smalldel, roots)
    
    ePerp = np.exp(LePerp)
    
    # Compute the Legendre weighted signal
    Lpmp = LePerp - LePar
    lgi = LegendreGaussianIntegral(Lpmp, 6)
    
    # Compute the spherical harmonic coefficients of the Watson's distribution
    coeff = WatsonSHCoeff(kappa)
    coeffMatrix = matlib.repmat(coeff, l_q, 1)
    
    # Compute the dot product between the symmetry axis of the Watson's distribution
    # and the gradient direction
    #
    # For numerical reasons, cosTheta might not always be between -1 and 1
    # Due to round off errors, individual gradient vectors in grad_dirs and the
    # fibredir are never exactly normal.  When a gradient vector and fibredir are
    # essentially parallel, their dot product can fall outside of -1 and 1.
    #
    # BUT we need make sure it does, otherwise the legendre function call below
    # will FAIL and abort the calculation!!!
    #
    cosTheta = np.dot(grad_dirs,fibredir)
    badCosTheta = abs(cosTheta)>1
    cosTheta[badCosTheta] = cosTheta[badCosTheta]/abs(cosTheta[badCosTheta])
    
    # Compute the SH values at cosTheta
    sh = np.zeros(coeff.shape)
    shMatrix = matlib.repmat(sh, l_q, 1)
    for i in range(7):
        shMatrix[:,i] = np.sqrt((i+1 - .75)/np.pi)
        # legendre function returns coefficients of all m from 0 to l
        # we only need the coefficient corresponding to m = 0
        # WARNING: make sure to input ROW vector as variables!!!
        # cosTheta is expected to be a COLUMN vector.
        tmp = np.zeros((l_q))
        for pol_i in range(l_q):
            tmp[pol_i] = scipy.special.lpmv(0, 2*i, cosTheta[pol_i])
        shMatrix[:,i] = shMatrix[:,i]*tmp
    
    E = np.sum(lgi*coeffMatrix*shMatrix, 1)
    # with the SH approximation, there will be no guarantee that E will be positive
    # but we need to make sure it does!!! replace the negative values with 10% of
    # the smallest positive values
    E[E<=0] = np.min(E[E>0])*0.1
    E = 0.5*E*ePerp
    
    return E

def WatsonHinderedDiffusionCoeff(dPar, dPerp, kappa):
    
    dw = np.zeros((2,1))
    dParMdPerp = dPar - dPerp
    
    if kappa < 1e-5:
        dParP2dPerp = dPar + 2.*dPerp
        k2 = kappa*kappa
        dw[0] = dParP2dPerp/3.0 + 4.0*dParMdPerp*kappa/45.0 + 8.0*dParMdPerp*k2/945.0
        dw[1] = dParP2dPerp/3.0 - 2.0*dParMdPerp*kappa/45.0 - 4.0*dParMdPerp*k2/945.0
    else:
        sk = np.sqrt(kappa)
        dawsonf = 0.5*np.exp(-kappa)*np.sqrt(np.pi)*scipy.special.erfi(sk)
        factor = sk/dawsonf
        dw[0] = (-dParMdPerp+2.0*dPerp*kappa+dParMdPerp*factor)/(2.0*kappa)
        dw[1] = (dParMdPerp+2.0*(dPar+dPerp)*kappa-dParMdPerp*factor)/(4.0*kappa)
    
    return dw

def SynthMeasHinderedDiffusion_PGSE(x, grad_dirs, G, delta, smalldel, fibredir):
    
    dPar=x[0]
    dPerp=x[1]
    
    # Radial wavenumbers
    GAMMA = 2.675987E8
    modQ = GAMMA*smalldel*G
    modQ_Sq = np.power(modQ,2.0)
    
    # Angles between gradient directions and fibre direction.
    cosTheta = np.dot(grad_dirs,fibredir)
    cosThetaSq = np.power(cosTheta,2.0)
    sinThetaSq = 1.0-cosThetaSq
    
    # b-value
    bval = (delta-smalldel/3.0)*modQ_Sq
    
    # Find hindered signals
    E=np.exp(-bval*((dPar - dPerp)*cosThetaSq + dPerp))
    
    return E

def SynthMeasWatsonHinderedDiffusion_PGSE(x, grad_dirs, G, delta, smalldel, fibredir):
    
    dPar = x[0]
    dPerp = x[1]
    kappa = x[2]
    
    # get the equivalent diffusivities
    dw = WatsonHinderedDiffusionCoeff(dPar, dPerp, kappa)
    
    xh = np.column_stack([dw[0], dw[1]])
    E = SynthMeasHinderedDiffusion_PGSE(xh, grad_dirs, G, delta, smalldel, fibredir)
    
    return E

def SynthMeasIsoGPD(d, protocol):
    
    if (protocol['pulseseq'] == 'PGSE') or (protocol['pulseseq'] == 'STEAM'):
    
        GAMMA = 2.675987E8
        modQ = GAMMA*protocol['smalldel'].transpose()*protocol.G.transpose()
        modQ_Sq = np.power(modQ,2)
        difftime = (protocol['delta'].transpose()-protocol['smalldel']/3)
    
        E = np.exp(-difftime*modQ_Sq*d)
    
    else:
        msg = 'SynthMeasIsoGPD() : Protocol %s not translated from NODDI matlab code yet' % protocol['pulseseq']
        raise ValueError(msg)
    
    return E