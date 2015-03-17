#!/usr/bin/python
""" Classes and functions for fitting tensors """
from __future__ import division, print_function, absolute_import

import warnings

import numpy as np
import scipy
import os.path as op
import pkg_resources
import math

import scipy.optimize as opt

from .noddi import make_model, scheme2noddi, synth_meas_watson_SH_cyl_neuman_PGSE, synth_meas_watson_hindered_diffusion_PGSE
from ..core.gradients import gradient_table_from_camino, GradientTable
from ..core.geometry import vector_norm, cart2sphere
from ..io.gradients import write_gradient_to_camino_file

"""from dipy.utils.six.moves import range
from dipy.data import get_sphere

from ..core.sphere import Sphere
from .vec_val_sum import vec_val_vect
from ..core.onetime import auto_attr
from .base import ReconstModel, ReconstFit"""

def create_high_resolution_scheme( schemeFilename, filenameHR ):
    
    # load scheme
    scheme = gradient_table_from_camino(schemeFilename, b0_threshold=0)

    # create a high-resolution version of it (to be used with Camino)
    n = len( scheme.shells )
    gradients = np.zeros( (500*n, 3) )
    G = np.zeros( (500*n, 1) )
    big_delta = np.zeros( (500*n, 1) )
    small_delta = np.zeros( (500*n, 1) )
    TE = np.zeros( (500*n, 1) )
    bs       = np.zeros( (500*n, 1) )
    grad500 = np.loadtxt( '500_dirs.txt' )
    for i in range(grad500.shape[0]):
        grad500[i,:] = grad500[i,:] / vector_norm( grad500[i,:] )
        if grad500[i,1] < 0:
            grad500[i,:] = -grad500[i,:] # to ensure they are in the spherical range [0,180]x[0,180]

    row = 0
    for i in range(n):
        gradients[row:row+500,0:3] = grad500
        G[row:row+500]   = scheme.shells[i]['G']
        big_delta[row:row+500]   = scheme.shells[i]['Delta']
        small_delta[row:row+500]   = scheme.shells[i]['delta']
        TE[row:row+500]   = scheme.shells[i]['TE']
        bs[row:row+500]           = scheme.shells[i]['bval']
        row = row + 500
        
    HR_scheme = GradientTable(gradients, G, big_delta, small_delta, TE, b0_threshold=0)
    write_gradient_to_camino_file(HR_scheme,filenameHR)
    
    return HR_scheme

def create_ylm(lmax, colatitude, longitude):
    
    Ylm = np.zeros((np.size(longitude),(lmax+2)*(lmax+1)/2))
    for l in range(0,lmax+1,2):
        Pm = np.ones((np.size(colatitude),l+1))
        for li in range(l+1):
            Pm[:,li] = scipy.special.lpmv(li, l, np.cos(colatitude))
        lconstant = np.sqrt((2.0*l + 1)/(4.0*np.pi))
        center = (l+1)*(l+2)/2 - l - 1
        Ylm[:,center] = lconstant*Pm[:,0]
        for m in range(1,l+1):
            precoeff = lconstant * math.sqrt(float(math.factorial(l - m))/float(math.factorial(l + m)))
            if (m % 2) == 1:
                precoeff = -precoeff
            Ylm[:, center + m] = np.sqrt(2.0)*precoeff*Pm[:,m]*np.cos(m*longitude)
            Ylm[:, center - m] = np.sqrt(2.0)*precoeff*Pm[:,m]*np.sin(m*longitude)   
    
    return Ylm

def precompute_rotation_matrices(lmax):
    dirs_path = pkg_resources.resource_filename('dipy',op.join('data','500_dirs.txt'))
    grad500 = np.loadtxt( dirs_path )
    for i in range(len(grad500)):
        grad500[i,:] = grad500[i,:] / vector_norm( grad500[i,:] )
    _,colatitude,longitude = cart2sphere(grad500[:,0], grad500[:,1],grad500[:,2])
    Ylm = create_ylm( lmax, colatitude, longitude )
    fit = np.dot(np.linalg.pinv( np.dot(Ylm.T,Ylm)) , Ylm.T)
    
    Ylm_rot = np.zeros((181,181,longitude.shape[0],(lmax+2)*(lmax+1)/2))
    for ox in range(Ylm_rot.shape[0]):
        for oy in range(Ylm_rot.shape[1]):
            Ylm_rot[ox,oy] = create_ylm( lmax, ox/180.0*np.pi, oy/180.0*np.pi )
    
    AUX = {'Ylm_rot':Ylm_rot, 'fit':fit}
    
    return AUX

def rotate_kernel( K, AUX, idx_IN, idx_OUT, isIsotropic ):
    
    if not isIsotropic:
        # fit SH and rotate kernel to 181*181 directions
        KRlm = np.zeros((AUX['fit'].shape[0],181,181), dtype='single')
        for ox in range(180+1):
            for oy in range(180+1):
                Ylm_rot = AUX['Ylm_rot'][ox, oy]
                for s in range(len(idx_IN)):
                    Klm = AUX['fit'] * K(idx_IN[s]) # fit SH of shell to rotate
                    Rlm = np.zeros(Klm.shape)
                    idx = 1
                    for l in range(0,AUX['lmax'],2):
                        const = np.sqrt(4.0*np.pi/2.0*l+1.0) * Klm[int((l*l+l+2.0)/2)]
                        for m in range(-l,l+1):
                            Rlm[idx] = const * Ylm_rot(idx)
                            idx = idx+1
                    KRlm[idx_OUT[s],ox,oy] = Rlm.astype('single')
    else:
        # simply fit SH
        KRlm = np.zeros((AUX['fit'],1,1), dtype='single')
        Ylm_rot = AUX['Ylm_rot'][0,0]
        for s in range(len(idx_IN)):
            KRlm[idx_OUT[s],0,0] = (AUX['fit'] * K[idx_IN[s]]).astype('single')               
    
    return KRlm
    
def generate_kernels_NODDI(config, data_path, schemeHR, AUX, idx_IN, idx_OUT):
    
    ATOMS_path = op.join(data_path,config.protocol,'common')
    
    noddi = make_model( 'WatsonSHStickTortIsoV_B0' ) # CHECK: might be unnecessary for our purpose -> ask Alessandro
    dPar = config.kernels['dPar'] * 1E-9
    dIso = config.kernels['dIso'] * 1E-9
    noddi['GS']['fixedvals'][1] = dPar
    noddi['GD']['fixedvals'][1] = dPar
    noddi['GS']['fixedvals'][4] = dIso
    noddi['GD']['fixedvals'][4] = dIso
    
    protocolHR = scheme2noddi( schemeHR )
    
    IC_KAPPAs = 1 / np.tan(config.kernels['IC_ODs']*np.pi/2)
    idx = 1
    for ii in range( len(IC_KAPPAs)):
        kappa = IC_KAPPAs[ii]
        signal_ic = synth_meas_watson_SH_cyl_neuman_PGSE( np.array([dPar, 0, kappa]), protocolHR['grad_dirs'], np.squeeze(protocolHR['gradient_strength']), np.squeeze(protocolHR['delta']), np.squeeze(protocolHR['smalldel']), np.array([0,0,1]), 0 )
        
        for v_ic in config.kernels['IC_VFs']:
            print( '\t\t- A_%03d... ' % idx )
            
            # generate
            dPerp = dPar * (1 - v_ic)
            signal_ec = synth_meas_watson_hindered_diffusion_PGSE( np.array([dPar, dPerp, kappa]), protocolHR['grad_dirs'], np.squeeze(protocolHR['gradient_strength']), np.squeeze(protocolHR['delta']), np.squeeze(protocolHR['smalldel']), np.array([0,0,1]) )
            signal = v_ic*signal_ic + (1-v_ic)*signal_ec

            # rotate and save
            lm = rotate_kernel( signal, AUX, idx_IN, idx_OUT, False )
            save( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), '-v6', 'lm' )

            idx = idx+1
            

class amico_conf(object):
    def __init__(self, data_path, protocol, subject, model):
        self.protocol         = protocol
        self.subject          = subject
        self.DATA_path        =  op.join( data_path, self.protocol, self.subject )
        self.dwiFilename      =  op.join( self.DATA_path, 'DWI.nii' )
        self.maskFilename     =  op.join( self.DATA_path, 'roi_mask.nii' )
        #self.dim              = []
        #self.pixdim           = []
        self.OUTPUT_path      = op.join(self.DATA_path,'AMICO')
        self.schemeFilename    = op.join( self.DATA_path, 'DWI.scheme' )
        #self.scheme            = []
        #self.kernels = []
        #self.OPTIMIZATION = []
        self.kernels = None
        self.set_model(model)
 
    def set_model(self,model_name):
        self.kernels = {}
        self.kernels['model']  = model_name.upper()
        self.kernels['IC_ODs'] = np.concatenate((np.array([0.03, 0.06]),np.linspace(0.09,0.99,10)))
        
        self.OPTIMIZATION = {}
        self.OPTIMIZATION['SPAMS_param']         = {}
        #self.OPTIMIZATION['LS_param']            = optimset('TolX',1e-4)
        self.OPTIMIZATION['SPAMS_param']['mode'] = 2
        self.OPTIMIZATION['SPAMS_param']['pos'] = True
    
        if self.kernels['model'] == 'NODDI':
                self.kernels['dPar']   = 1.7                                        # units of 1E-9 (m/s)
                self.kernels['dIso']   = 3.0                                       # units of 1E-9 (m/s)
                self.kernels['IC_VFs'] = np.linspace(0.1, 0.99,12)
                self.kernels['IC_ODs'] = np.concatenate((np.array([0.03, 0.06]),np.linspace(0.09,0.99,10)))
                self.OPTIMIZATION['SPAMS_param']['lambda1']  = 5e-1
                self.OPTIMIZATION['SPAMS_param']['lambda2'] = 1e-3
        elif self.kernels['model'] == 'ACTIVEAX':
                self.kernels['dPar']   = 0.6                                        # units of 1E-9 (m/s)
                self.kernels['dIso']   = 2.0                                        # units of 1E-9 (m/s)
                self.kernels['IC_Rs']  = np.concatenate((np.array([0.01]),np.linspace(0.5,10,20)))                # units of 1E-6 (micrometers)
                self.kernels['IC_VFs'] = np.arange(0.3,0.9,0.1)
    
                self.OPTIMIZATION['SPAMS_param']['lambda1']  = 0.25
                self.OPTIMIZATION['SPAMS_param']['lambda2'] = 4
    
        else:
                msg = '\t[KERNELS_Generate] Model "%s" not recognized' % self.kernels.model
                raise ValueError(msg)

"""

config = amico_conf('/home/david/Desktop/Test_folder','NODDI','subj_1','noddi')
data_path = '/home/david/Desktop/Test_folder/NODDI'

schemeHR = create_high_resolution_scheme( '/home/david/Desktop/Test_folder/NODDI/scheme_for_dipy.txt', '/home/david/Desktop/Test_folder/NODDI/dipy_HR_scheme.txt' )


Note:
To make IC_VFs and IC_ODs that have singleton dimension:
                self.kernels['IC_VFs'] = np.linspace(0.1, 0.99,12)[:,None]
                self.kernels['IC_ODs'] = np.concatenate((np.array([[0.03], [0.06]]),np.linspace(0.09,0.99,10)[:,None]))

"""
    