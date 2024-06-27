import numpy as np
from tensorflow.keras.models import load_model

def load_models():
    mri_model = load_model('models/mri_model.h5')
    eeg_model = load_model('models/eeg_model.h5')
    return mri_model, eeg_model

def predict(mri_data, eeg_data, mri_model, eeg_model):
    mri_prediction = mri_model.predict(mri_data)
    eeg_prediction = eeg_model.predict(eeg_data)
    
    combined_prediction = (mri_prediction + eeg_prediction) / 2 
    return combined_prediction

if __name__ == "__main__":
    mri_data = np.load('data/mri/preprocessed_sample_mri.npy')
    eeg_data = np.load('data/eeg/preprocessed_sample_eeg.npy')
    
    mri_model, eeg_model = load_models()
    result = predict(mri_data, eeg_data, mri_model, eeg_model)
    
    print(f"Combined prediction: {result}")
