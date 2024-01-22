from utils_Harmony import *

subject_dir='E:/database/derivatives/freesurfer-7.1.1'
subject='sub-11'
raw_path = 'E:/database/derivatives/eeglab-v14.1.1/sub-11/eeg/sub-11_task-motion_desc-preproc_eeg.set'
epochs = mne.io.read_epochs_eeglab(raw_path)
epochs.apply_baseline((None,0.0))
epochs.set_eeg_reference(ref_channels='average', projection=True)
epochs.apply_proj()
evoked = epochs.average()


src,trans,bem,fwd,noise_cov,leadfield,info = pre_procss_for_epochs(subject_dir,subject,epochs,write=False,trans_cal=False)


fwd_fixed = mne.convert_forward_solution(
fwd, surf_ori=True, force_fixed=True, use_cps=True
)
leadfield = fwd_fixed['sol']['data']

evoked = epochs.average().pick("eeg")
inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)
method = 'MNE'
snr = 5.
lambda2 = 1. / snr ** 2
stc = apply_inverse(evoked, inverse_operator, lambda2, method=method, pick_ori=None)
stc.plot(subjects_dir = subject_dir,subject = subject,surface='white',hemi = 'both')


r_lh, theta_lh, phi_lh = read_sphere_and_convert(subject_dir+'/'+subject+'/surf/lh.sphere')
r_rh, theta_rh, phi_rh = read_sphere_and_convert(subject_dir+'/'+subject+'/surf/rh.sphere')

N = 6
basis_function_lh = head_basis(theta_lh.shape[0], N, theta_lh, phi_lh)
basis_function_rh = head_basis(theta_rh.shape[0], N, theta_rh, phi_rh)
basis_function_lh = basis_function_lh[src[0]['vertno'],:]
basis_function_rh = basis_function_rh[src[1]['vertno'],:]

W, _ = mne.minimum_norm.inverse.compute_whitener(noise_cov, info)

ex_data,theta  = source_localization(evoked,W,1e16,basis_function_lh,basis_function_rh,src,leadfield)
stc_1 = stc
stc_1.data = ex_data
stc_1.plot(subjects_dir=subject_dir,subject = subject,hemi = 'both',surface = 'white')