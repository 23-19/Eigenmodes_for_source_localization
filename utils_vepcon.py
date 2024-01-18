import numpy as np
import nibabel as nib
from scipy import linalg
import mne
from mne.io.constants import FIFF
from mne.coreg import Coregistration
from mne.viz import Brain
import os.path as op
from scipy import spatial
from mne.minimum_norm import make_inverse_operator, apply_inverse
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd


def pre_procss_for_epochs(subject_dir,subject,epochs,write= True,trans_cal = False):
    """
    if write = Ture:
        input: subject's dir ; epoch class data
        write: source space ; trans ; bem ; fwd ; noise_cov
        ouput: source space ; trans ; bem ; fwd ; noise_cov ; leadfield
    if write = False:
        input: subject's dir ; epoch class data
        ouput: source space ; trans ; bem ; fwd ; noise_cov ; leadfield
    
    trans can be obtained using the automatic approximation method
    but for accuracy, 
    it is still recommended to use the gui method
    
    """
    
    if write : 
        # source space
        src= mne.setup_source_space(subject,spacing='ico5',surface='white',subjects_dir=subject_dir,add_dist='patch')
        
        if trans_cal:
            fiducials = 'estimated'
            coreg = Coregistration(epochs.info, subject,subjects_dir=subject_dir,fiducials=fiducials)
            coreg.fit_fiducials(verbose=True)
            coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)
            coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters
            coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
            dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
            print(
                f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
                f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
            )
            trans = coreg.trans
        else:
            trans = mne.read_trans(subject_dir+'/'+subject+'/trans-trans.fif')
        
        # bem
        conductivity = (0.3, 0.006, 0.3)  # for three layers
        model = mne.make_bem_model(subject=subject, ico=4, conductivity=conductivity, subjects_dir=subject_dir)
        bem = mne.make_bem_solution(model)
        
        # forward model
        fwd = mne.make_forward_solution(
        info = epochs.info,
        trans = trans,
        src = src,
        bem = bem,
        meg = False,
        eeg=True,
        mindist=0.0,
        n_jobs=8,
        verbose=True,
        )
        fwd_fixed = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, use_cps=True
        )
        
        # noise cov
        epochs.apply_baseline((None, 0))
        noise_cov = mne.compute_covariance(epochs,tmin=None,tmax = 0.0,method=["shrunk", "empirical"], rank=None, verbose=True,n_jobs=8)
        
        #leadfield
        leadfield = fwd_fixed['sol']['data']
        
        #info
        info = epochs.info
        
        # write
        mne.write_source_spaces(subject_dir+'/'+subject+'/src-src.fif',src,overwrite=True)
        mne.write_trans(subject_dir+'/'+subject+'/trans-trans.fif',trans,overwrite=True)
        mne.write_bem_solution(subject_dir+'/'+subject+ '/bem_solution.fif',bem,overwrite=True)
        mne.write_forward_solution(subject_dir+'/'+subject+'/fwd-fwd.fif',fwd_fixed,overwrite=True)
        mne.write_cov(subject_dir+'/'+subject+'/cov-cov.fif',noise_cov,overwrite=True)
        mne.io.write_info(subject_dir+'/'+subject+'/info-info.fif',info,data_type = 5)
        
    else:
        src = mne.read_source_spaces(subject_dir+'/'+subject+'/src-src.fif')
        trans = mne.read_trans(subject_dir+'/'+subject+'/trans-trans.fif')
        bem = mne.read_bem_solution(subject_dir+'/'+subject+ '/bem_solution.fif')
        fwd = mne.read_forward_solution(subject_dir+'/'+subject+'/fwd-fwd.fif')
        noise_cov = mne.read_cov(subject_dir+'/'+subject+'/cov-cov.fif') 
        info = epochs.info   
        leadfield = fwd['sol']['data']
        
    return src,trans,bem,fwd,noise_cov,leadfield,info

