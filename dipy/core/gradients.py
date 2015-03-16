from __future__ import division, print_function, absolute_import

from ..utils.six import string_types

import numpy as np

from ..io import gradients as io
from .onetime import auto_attr
from .geometry import vector_norm


class GradientTable(object):
    """Diffusion gradient information

    Parameters
    ----------
    gradients : array_like (N, 3)
        N diffusion gradients
    b0_threshold : float
        Gradients with b-value less than or equal to `b0_threshold` are
        considered as b0s i.e. without diffusion weighting.

    Attributes
    ----------
    gradients : (N,3) ndarray
        diffusion gradients
    bvals : (N,) ndarray
        The b-value, or magnitude, of each gradient direction.
    qvals: (N,) ndarray
        The q-value for each gradient direction. Needs big and small
        delta.
    bvecs : (N,3) ndarray
        The direction, represented as a unit vector, of each gradient.
    b0s_mask : (N,) ndarray
        Boolean array indicating which gradients have no diffusion
        weighting, ie b-value is close to 0.
    b0_threshold : float
        Gradients with b-value less than or equal to `b0_threshold` are
        considered to not have diffusion weighting.

    See Also
    --------
    gradient_table

    """
    def __init__(self, gradients, G = None, big_delta=None, small_delta=None,
                 TE = None, b0_threshold=0):
        """Constructor for GradientTable class"""
        gradients = np.asarray(gradients)
        if gradients.ndim != 2 or gradients.shape[1] != 3:
            raise ValueError("gradients should be an (N, 3) array")
        self.gradients = gradients
        # Avoid nan gradients. Set these to 0 instead:
        self.gradients = np.where(np.isnan(gradients), 0., gradients)
        self.G = G
        self.big_delta = big_delta
        self.small_delta = small_delta
        self.TE = TE
        self.b0_threshold = b0_threshold
        self.nS = gradients.shape[0]
        if G != None and big_delta != None and small_delta != None and TE!=None:
            self.version = 1
        else:
            self.version = 0

    @auto_attr
    def bvals(self):
        if self.version:
            return np.power( 267.513e6 * self.G * self.small_delta, 2) * (self.big_delta - self.small_delta/3) * 1e-6
        else:
            return vector_norm(self.gradients)

    @auto_attr
    def qvals(self):
        tau = self.big_delta - self.small_delta / 3.0
        return np.sqrt(self.bvals / tau) / (2 * np.pi)

    @auto_attr
    def b0s_mask(self):
        return self.bvals <= self.b0_threshold
    
    @auto_attr
    def dwi_mask(self):
        return -self.b0s_mask

    @auto_attr
    def bvecs(self):
        # To get unit vectors we divide by bvals, where bvals is 0 we divide by
        # 1 to avoid making nans
        denom = self.bvals + (self.bvals == 0)
        denom = denom.reshape((-1, 1))
        return self.gradients / denom
    
    @auto_attr
    def dwi_count(self):
        return sum(self.dwi_mask)
    
    @auto_attr
    def b0_count(self):
        return sum(self.b0s_mask)
    
    @auto_attr
    def shells(self):
        if self.version:
            scheme = np.column_stack((self.G, self.big_delta, self.small_delta, self.TE))
        else:
            scheme = self.bvals.copy()
        _, idx_u = np.unique(np.ascontiguousarray(scheme).view(np.dtype((np.void, scheme.dtype.itemsize * scheme.shape[1]))), return_index=True)
        idx_u = np.sort(idx_u)
        shells = []
        for i in idx_u:
            if self.bvals[i] > self.b0_threshold:
                tmp = {'bval':self.bvals[i]}
                if self.version:
                    tmp['G'] = scheme[i,0]
                    tmp['Delta'] = scheme[i,1]
                    tmp['delta'] = scheme[i,2]
                    tmp['TE'] = scheme[i,3]
                else:
                    tmp['G'] = np.nan
                    tmp['Delta'] = np.nan
                    tmp['delta'] = np.nan
                    tmp['TE'] = np.nan
                shells.append(tmp)
        return shells

    @property
    def info(self):
        print('B-values shape (%d,)' % self.bvals.shape)
        print('         min %f ' % self.bvals.min())
        print('         max %f ' % self.bvals.max())
        print('B-vectors shape (%d, %d)' % self.bvecs.shape)
        print('         min %f ' % self.bvecs.min())
        print('         max %f ' % self.bvecs.max())
        
    def write_to_camino_file(self,filename):
        if self.version:
            scheme = np.column_stack((self.gradients, self.G, self.big_delta, self.small_delta, self.TE))
            header = 'VERSION: STEJSKALTANNER'
        else:
            scheme = np.column_stack((self.gradients,self.bvals))
            header = 'VERSION: BVECTOR'
        np.savetxt(filename,scheme, header = header, fmt = '%15e')


def gradient_table_from_bvals_bvecs(bvals, bvecs, b0_threshold=0, atol=1e-2,
                                  **kwargs):
    """Creates a GradientTable from a bvals array and a bvecs array

    Parameters
    ----------
    bvals : array_like (N,)
        The b-value, or magnitude, of each gradient direction.
    bvecs : array_like (N, 3)
        The direction, represented as a unit vector, of each gradient.
    b0_threshold : float
        Gradients with b-value less than or equal to `bo_threshold` are
        considered to not have diffusion weighting.
    atol : float
        Each vector in `bvecs` must be a unit vectors up to a tolerance of
        `atol`.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword inputs are passed to GradientTable.

    Returns
    -------
    gradients : GradientTable
        A GradientTable with all the gradient information.

    See Also
    --------
    GradientTable, gradient_table

    """
    bvals = np.asarray(bvals, np.float)
    bvecs = np.asarray(bvecs, np.float)
    dwi_mask = bvals > b0_threshold

    # check that bvals is (N,) array and bvecs is (N, 3) unit vectors
    if bvals.ndim != 1 or bvecs.ndim != 2 or bvecs.shape[0] != bvals.shape[0]:
        raise ValueError("bvals and bvecs should be (N,) and (N, 3) arrays "
                         "respectively, where N is the number of diffusion "
                         "gradients")

    bvecs_close_to_1 = abs(vector_norm(bvecs) - 1) <= atol
    if bvecs.shape[1] != 3 or not np.all(bvecs_close_to_1[dwi_mask]):
        raise ValueError("bvecs should be (N, 3), a set of N unit vectors")

    bvecs = np.where(bvecs_close_to_1[:, None], bvecs, 0)
    bvals = bvals * bvecs_close_to_1
    gradients = bvals[:, None] * bvecs

    grad_table = GradientTable(gradients, b0_threshold=b0_threshold, **kwargs)
    grad_table.bvals = bvals
    grad_table.bvecs = bvecs
    grad_table.b0s_mask = ~dwi_mask

    return grad_table

def gradient_table_from_camino(filename, b0_threshold):
    camino = np.loadtxt(filename,skiprows=1)
    if camino.shape[1] > 4:
        bvecs = camino[:,0:3]
        G = camino[:,3]
        big_delta = camino[:,4]
        small_delta = camino[:,5]
        TE = camino[:,6]
        return GradientTable(bvecs, G, big_delta, small_delta, TE, b0_threshold)
    else:
        bvecs = camino[:,0:3]
        bvals = camino[:,3]
        bvecs = bvecs/vector_norm(bvecs)*bvals
        return GradientTable(bvecs, b0_threshold = b0_threshold)
        

def gradient_table(bvals, bvecs=None, G = None, big_delta=None, small_delta=None,
                   TE = None, b0_threshold=0, atol=1e-2):
    """A general function for creating diffusion MR gradients.

    It reads, loads and prepares scanner parameters like the b-values and
    b-vectors so that they can be useful during the reconstruction process.

    Parameters
    ----------

    bvals : can be any of the four options

        1. an array of shape (N,) or (1, N) or (N, 1) with the b-values.
        2. a path for the file which contains an array like the above (1).
        3. an array of shape (N, 4) or (4, N) or (N, 7) or (7, N). Then this 
           parameter is considered to be a b-table which contains both bvals 
           and bvecs. In this case the next parameter is skipped.
        4. a path for the file which contains an array like the one at (3).

    bvecs : can be any of two options

        1. an array of shape (N, 3) or (3, N) with the b-vectors.
        2. a path for the file which contains an array like the previous.

    big_delta : float
        acquisition timing duration (default None)

    small_delta : float
        acquisition timing duration (default None)

    b0_threshold : float
        All b-values with values less than or equal to `bo_threshold` are
        considered as b0s i.e. without diffusion weighting.

    atol : float
        All b-vectors need to be unit vectors up to a tolerance.

    Returns
    -------
    gradients : GradientTable
        A GradientTable with all the gradient information.

    Examples
    --------
    >>> from dipy.core.gradients import gradient_table
    >>> bvals=1500*np.ones(7)
    >>> bvals[0]=0
    >>> sq2=np.sqrt(2)/2
    >>> bvecs=np.array([[0, 0, 0],
    ...                 [1, 0, 0],
    ...                 [0, 1, 0],
    ...                 [0, 0, 1],
    ...                 [sq2, sq2, 0],
    ...                 [sq2, 0, sq2],
    ...                 [0, sq2, sq2]])
    >>> gt = gradient_table(bvals, bvecs)
    >>> gt.bvecs.shape == bvecs.shape
    True
    >>> gt = gradient_table(bvals, bvecs.T)
    >>> gt.bvecs.shape == bvecs.T.shape
    False

    Notes
    -----
    1. Often b0s (b-values which correspond to images without diffusion
       weighting) have 0 values however in some cases the scanner cannot
       provide b0s of an exact 0 value and it gives a bit higher values
       e.g. 6 or 12. This is the purpose of the b0_threshold in the __init__.
    2. We assume that the minimum number of b-values is 7.
    3. B-vectors should be unit vectors.

    """

    # If you provided strings with full paths, we go and load those from
    # the files:
    if isinstance(bvals, string_types):
        bvals, _ = io.read_bvals_bvecs(bvals, None)
    if isinstance(bvecs, string_types):
        _, bvecs = io.read_bvals_bvecs(None, bvecs)

    bvals = np.asarray(bvals)
    # If bvecs is None we expect bvals to be an (N, 4) or (4, N) or (N, 7) or (7, N) array.
    if bvecs is None:
        if bvals.shape[-1] == 4:
            bvecs = bvals[:, 0:3]
            bvals = np.squeeze(bvals[:, 3])
        elif bvals.shape[0] == 4:
            bvecs = bvals[0:3, :].T
            bvals = np.squeeze(bvals[3, :])
        elif bvals.shape[-1] == 7:
            bvecs = bvals[:, 0:3]
            G = bvals[:,3]
            big_delta = bvals[:,4]
            small_delta = bvals[:,5]
            TE = bvals[:,6]
        elif bvals.shape[0] == 7:
            bvecs = bvals[0:3,:]
            G = bvals[3,:]
            big_delta = bvals[4,:]
            small_delta = bvals[5,:]
            TE = bvals[6,:]
        else:
            raise ValueError("input should be bvals and bvecs OR an (N, 4)"
                             " array containing both bvals and bvecs")
    else:
        bvecs = np.asarray(bvecs)
        if (bvecs.shape[1] > bvecs.shape[0])  and bvecs.shape[0]>1:
            bvecs = bvecs.T
    if bvals.shape[-1] == 4 or bvals.shape[0] == 4:
        return gradient_table_from_bvals_bvecs(bvals, bvecs, G = G, big_delta=big_delta,
                                           small_delta=small_delta, TE = TE,
                                           b0_threshold=b0_threshold,
                                           atol=atol)
    else:
        return GradientTable(bvecs, G, big_delta, small_delta, TE, b0_threshold)
