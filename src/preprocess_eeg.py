import numpy as np
import mne
from sklearn.preprocessing import StandardScaler

def load_eeg_data(eeg_path):
    raw = mne.io.read_raw_fif(eeg_path, preload=True)
    return raw

def preprocess_eeg_data(raw):
    raw.filter(1., 50.)
    epochs = mne.make_fixed_length_epochs(raw, duration=1., overlap=0.5)
    data = epochs.get_data()
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    return data

if __name__ == "__main__":
    eeg_path = 'data/eeg/sample_eeg.fif'
    raw = load_eeg_data(eeg_path)
    eeg_data = preprocess_eeg_data(raw)
    np.save('data/eeg/preprocessed_sample_eeg.npy', eeg_data)
