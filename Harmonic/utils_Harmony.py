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
import numpy as np
import math 
from scipy.special import lpmv


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
        # fwd_fixed = mne.convert_forward_solution(
        # fwd, surf_ori=True, force_fixed=True, use_cps=True
        # )
        
        # noise cov
        epochs.apply_baseline((None, 0))
        noise_cov = mne.compute_covariance(epochs,tmin=None,tmax = 0.0,method=["shrunk", "empirical"], rank=None, verbose=True,n_jobs=8)
        
        #leadfield
        leadfield = fwd['sol']['data']
        
        #info
        info = epochs.info
        
        # write
        mne.write_source_spaces(subject_dir+'/'+subject+'/src-src.fif',src,overwrite=True)
        mne.write_trans(subject_dir+'/'+subject+'/trans-trans.fif',trans,overwrite=True)
        mne.write_bem_solution(subject_dir+'/'+subject+ '/bem_solution.fif',bem,overwrite=True)
        mne.write_forward_solution(subject_dir+'/'+subject+'/fwd-fwd.fif',fwd,overwrite=True)
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


def read_sphere_and_convert(filepath):
    """
    从 FreeSurfer 球面文件中读取顶点坐标并转换为球坐标系。

    :param filepath: 球面文件的路径
    :return: 转换后的球坐标（半径，天顶角，方位角）
    """
    def cartesian_to_spherical(coords):
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)  # 天顶角
        phi = np.arctan2(y, x) # 方位角
        return r, theta, phi

    # 读取球面文件
    verts, _ = mne.read_surface(filepath)

    # 转换为球坐标
    r, theta, phi = cartesian_to_spherical(verts)

    return r, theta, phi


def sensor_coords(info):
    """
    
    get the sensor coords from MNE's info 
    and 
    convert it from m to mm
    
    """
    
    coords = []
    for ch in info['chs']:
        if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH:
            coords.append(ch['loc'][:3]*1000)  
    return np.array(coords).T  


def cal_coord(coords, r):
    """
    Calculate the spherical coordinates (theta and phi) for each point in coords.

    Parameters:
    coords (np.array): A 2D numpy array of shape (3, N) representing the x, y, z coordinates.
    r (float): The radius used in the calculation.

    Returns:
    tuple: Two numpy arrays representing theta and phi values for each point.
    """
    I = coords.shape[1]
    theta = np.zeros(I)
    phi = np.zeros(I)

    for i in range(I):
        x = coords[0, i]
        y = coords[1, i]
        z = coords[2, i]
        
        theta[i] = np.arccos(z / r)

        if x == 0 and y == 0:
            phi[i] = 0
        else:
            phi[i] = np.arctan2(y, x)

    return theta, phi


def head_basis(I, N, theta, phi):
    """
    计算头部谐波基函数。

    :param I: 传感器数量
    :param N: 谐波的最大阶数
    :param theta: 传感器位置的天顶角，应为 [0, π] 范围内的弧度
    :param phi: 传感器位置的方位角，应为 [0, 2π] 范围内的弧度
    :return: 头部谐波基函数的矩阵
    """


    Y = np.zeros((I, (N + 1)**2))
    k = 0
    for n in range(N + 1):
        for m in range(-n, n + 1):
            # 计算归一化的伴随勒让德多项式
            P_m_n = lpmv(abs(m), n, np.cos(theta))
            # 计算归一化因子
            norm_factor = ((-1)**abs(m) * np.sqrt((2 * n + 1) * math.factorial(n - abs(m)) / (4 * np.pi * math.factorial(n + abs(m)))))
            
            if m >= 0:
                Y[:, k] = norm_factor * P_m_n * np.cos(m * phi)
            else:
                Y[:, k] = norm_factor * P_m_n * np.sin(abs(m) * phi)

            k += 1

    return Y




def get_coeff_hsph(Y):
    """
    
    Y : Head harmonic basis function 
    get the Coefficient matrix in head harmonic space
    
    """
    I = Y.shape[0]
    Yalp = np.array([np.outer(Y[i, :].T, Y[i, :]) for i in range(I)])
    b = np.array([np.sum(Yalp[i, :, :] * np.eye(Y.shape[1])) for i in range(I)])
    M = np.sum(Yalp[:, np.newaxis, :, :] * Yalp[np.newaxis, :, :, :], axis=(2, 3))
    d, vects = np.linalg.eig(M)
    indx = np.argsort(d)[::-1]
    d = d[indx]
    vects = vects[:, indx]
    rank = np.sum(d > 0.001)
    d = np.diag(d[:rank])
    u = vects[:, :rank]
    Gamma = np.diag(np.dot(u, np.dot(np.linalg.inv(d), np.dot(u.T, b))))
    return Gamma

def source_localization(evoked,whitener,Lambda,basis_function_lh,basis_function_rh,src,leadfield):
    
    # 调整先验
 
    lam = np.eye(basis_function_lh.shape[1]*2)
    
    # 白化EEG信号 cov 2293
    evoked_whitened =  np.sqrt(evoked.nave) * np.dot(whitener, evoked.data)
    

    # 白化导联矩阵 
    G_lh =  whitener @ leadfield[:,:len(src[0]['vertno'])] @ basis_function_lh
    G_rh =  whitener @ leadfield[:,len(src[0]['vertno']):] @ basis_function_rh
    G = np.hstack((G_lh, G_rh))
    
    theta = np.linalg.inv(G.T @ G + Lambda * lam * lam ) @ G.T  @ evoked_whitened
    
    half_size = theta.shape[0] // 2

    theta_lh = theta[:half_size, :]
    theta_rh = theta[half_size:, :]

    result_lh = basis_function_lh @ theta_lh
    result_rh = basis_function_rh @ theta_rh

    ex_data = np.vstack((result_lh, result_rh))
    
    return ex_data, theta 
