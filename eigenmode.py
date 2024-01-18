import lapy
from lapy import Solver, TriaMesh
from utils_vepcon import *


def get_tria_from_src(src):
    """
    get the TriaMesh class from the source space
    input: source space
    output: TriaMesh lh and rh
    
    """
    face_lh = src[0]['tris']
    coord_lh = src[0]['rr']*1000
    tria_lh = TriaMesh(coord_lh, face_lh, fsinfo=None)

    face_rh = src[1]['tris']
    coord_rh = src[1]['rr']*1000 
    tria_rh = TriaMesh(coord_rh, face_rh, fsinfo=None)
    
    return tria_lh,tria_rh
    
    

def calc_eig_from_src(src, num_modes,cut=True):
    """Calculate the eigenvalues and eigenmodes of the src's surface, .

    Parameters
    ----------
    src : source space

    num_modes : int
        Number of eigenmodes to be calculated

    cut : bool | if ture : The number of vertices of emodes is the same as the number of vertices of src
    Returns
    ------
    evals_lh evals_rh : array (num_modes x 1)
        Eigenvalues
    emodes_lh emodes_rh : array (number of surface points x num_modes)
        Eigenmodes
    """
    tria_lh,tria_rh = get_tria_from_src(src)
    

    fem_lh = Solver(tria_lh)
    evals_lh, emodes_lh = fem_lh.eigs(k=num_modes)
    
    fem_rh = Solver(tria_rh)
    evals_rh, emodes_rh = fem_rh.eigs(k=num_modes)
    if cut:
        emodes_lh = emodes_lh[src[0]['vertno'],:]
        emodes_rh = emodes_rh[src[1]['vertno'],:]
        
    return evals_lh, evals_rh,emodes_lh,emodes_rh


def source_localization(evoked,whitener,Lambda,evals,emodes,leadfield):
    """
    input: evoked_whitened; whitener; Lambda; evals; emodes; leadfield
            lambda : the hyperparameters,The recommended value is 0.04, but it needs to be changed according to the specific conditions of the data set.
            evals : Eigenvalues obtained when decomposing the geometry surf
            emodes : the eigenmodes obtained when decomposing the geometry surf and In the operation, the eigenmodes of the left brain and the right brain are combined.
    
    output: ex_data,theta
            ex_data : The data of STC class obtained after source localization calculation through eigenmode
            theta :specifically reflects the coefficient distribution and changes of the eigenmode at different time points.
    """
    # 调整先验
    evals = (evals - evals.mean()) / evals.std()
    evals = np.log(1 + np.exp(-evals))
    lam = np.diag(1/evals)
    
    # 白化EEG信号 cov 2293
    evoked_whitened =  np.sqrt(evoked.nave) * np.dot(whitener, evoked.data)
    
    # 白化导联矩阵 
    G =  whitener @ leadfield @ emodes
    
    theta = np.linalg.inv(G.T @ G + Lambda * lam ) @ G.T  @ evoked_whitened
    ex_data = emodes @ theta
    return ex_data , theta

