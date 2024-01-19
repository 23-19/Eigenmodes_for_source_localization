from utils_vepcon import *
from eigenmode import *

# Before calculation
# command use mne.gui to calculate trans first

subject_dir='E:/database/derivatives/freesurfer-7.1.1'
subject='sub-11'
path = 'E:/database/derivatives/eeglab-v14.1.1/sub-11/eeg/sub-11_task-motion_desc-preproc_eeg.set'
epochs = mne.io.read_epochs_eeglab(path)
epochs.apply_baseline((None,0.0))
epochs.set_eeg_reference(ref_channels='average', projection=True)
epochs.apply_proj()


src,trans,bem,fwd,noise_cov,leadfield,info = pre_procss_for_epochs(subject_dir,subject,epochs,write=True,trans_cal=True)


fwd_fixed = mne.convert_forward_solution(
fwd, surf_ori=True, force_fixed=True, use_cps=True
)
leadfield = fwd_fixed['sol']['data']

evoked = epochs.average().pick("eeg")
evoked= mne.set_eeg_reference(evoked, ref_channels='average', projection=True)[0]

inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)
method = 'MNE'
snr = 5.
lambda2 = 1. / snr ** 2
stc = apply_inverse(evoked, inverse_operator, lambda2, method=method, pick_ori=None)

stc.plot(hemi = 'both',surface = 'white')

W, _ = mne.minimum_norm.inverse.compute_whitener(noise_cov, info)

evals_lh, evals_rh,emodes_lh,emodes_rh = calc_eig_from_src(src, 150,cut=True)
emodes = np.vstack((emodes_lh, emodes_rh))

source,theta = source_localization(evoked,W,0.04,evals_lh,emodes,leadfield)

stc_1 = stc
stc_1.data = source

stc_1.plot(hemi = 'both',surface = 'white')

print("\nPress Enter twice to exit the program.")
enter_count = 0
while True:
    if input() == '':
        enter_count += 1
        if enter_count == 2:
            print("Exiting program.")
            break
    else:
        enter_count = 0
        

