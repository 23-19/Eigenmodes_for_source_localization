from utils_vepcon import *
from eigenmode import *


subject_dir='E:/OSE_dataset/002/'
subject = 'OSE_002'

raw = mne.io.read_raw_brainvision('E:/OSE_dataset/002/EEG/Somatosensory/run01.vhdr',preload=True)
mat_data_dir = 'E:/OSE_dataset/002/EEG/Somatosensory/allruns.pos.mat'

channel_types = {'EOG1': 'eog', 'EOG2': 'eog'}
raw.set_channel_types(channel_types)
mat_data = scipy.io.loadmat(mat_data_dir)    
electrode_names = mat_data['name'] 
electrode_positions = mat_data['pos']
eeg_electrode_names = raw.ch_names

electrode_names_list = [str(e[0]) for e in electrode_names.ravel()]

sorted_positions = []

for eeg_name in eeg_electrode_names:

    if eeg_name in electrode_names_list:
        index = electrode_names_list.index(eeg_name)
        sorted_positions.append(electrode_positions[index])
    else:
        print(f"{eeg_name} not found in electrode_names_list")

montage = mne.channels.make_dig_montage(ch_pos=dict(zip(eeg_electrode_names, sorted_positions)), coord_frame='head')
raw.set_montage(montage)
raw.filter(l_freq=1, h_freq=40)
picks = mne.pick_types(raw.info, eeg=True,eog = True)
ica = mne.preprocessing.ICA(n_components=50, random_state=97, max_iter='auto')
ica.fit(raw, picks=picks, decim=3,)
#exclude = [0,1,2,7,5,8,12,15,16,17,20,21,25,27,28,31,34,35,38,47,48,] # Audio
#exclude = [0,1,2,6,10,14,15,17,19,24,26,29,33,34] # Motor
exclude = [0,5,6,7,11,18,19,20,22,25,26,27,30,31,33,34,39,43] # Somatosensory
raw = ica.apply(raw, exclude=exclude)
events ,event_id = mne.events_from_annotations(raw)
print(event_id)


# For different tasks, the event id needs to be changed to adapt
epochs = mne.Epochs(raw,events,event_id=1,tmin = -0.3 ,tmax = 0.5,proj = True,picks='eeg',baseline=(None,0),preload=True)
epochs.apply_baseline((None, 0))
epochs.set_eeg_reference(ref_channels='average', projection=True)


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


evals_lh, evals_rh,emodes_lh,emodes_rh = calc_eig_from_src(src, 300,cut=True)
W, _ = mne.minimum_norm.inverse.compute_whitener(noise_cov, info)


source,theta  = source_localization(evoked,W,1e16,evals_lh,evals_rh,emodes_lh,emodes_rh,src,leadfield)

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