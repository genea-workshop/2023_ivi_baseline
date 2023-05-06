import h5py
import numpy as np

# NOTE: Since the data is augmented in such a way that every recording is
#       present twice (once as the main agent, # and once as the interlocutor),
#       it's enough to look at the main-agent half for audio normalisation
h5 = h5py.File("trn_main-agent_v0.h5", "r")

mfcc = np.concatenate([h5[key]['audio']['mfcc'][:] for key in h5.keys()])
mel = np.concatenate([h5[key]['audio']['melspectrogram'][:] for key in h5.keys()])
prosody = np.concatenate([h5[key]['audio']['prosody'][:] for key in h5.keys()])

np.save("mfcc_mean.npy", np.mean(mfcc, axis=0))
np.save("mfcc_std.npy", np.std(mfcc, axis=0) + 1e-5)

np.save("mel_mean.npy", np.mean(mel, axis=0))
np.save("mel_std.npy", np.std(mel, axis=0) + 1e-5)

np.save("prosody_mean.npy", np.mean(prosody, axis=0))
np.save("prosody_std.npy", np.std(prosody, axis=0) + 1e-5)

h5.close()
