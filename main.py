from src.preprocess_mri import preprocess_mri_data, load_mri_data
from src.preprocess_eeg import preprocess_eeg_data, load_eeg_data
from src.integrate_models import load_models, predict
import numpy as np

def main():
    # Preprocess MRI Data
    mri_path = 'data/mri/sample_mri.nii'
    mri_image = load_mri_data(mri_path)
    mri_image = preprocess_mri_data(mri_image)
    np.save('data/mri/preprocessed_sample_mri.npy', mri_image)
    
    # Preprocess EEG Data
    eeg_path = 'data/eeg/sample_eeg.fif'
    raw = load_eeg_data(eeg_path)
    eeg_data = preprocess_eeg_data(raw)
    np.save('data/eeg/preprocessed_sample_eeg.npy', eeg_data)
    
    # Load models and predict
    mri_model, eeg_model = load_models()
    mri_data = np.load('data/mri/preprocessed_sample_mri.npy')
    eeg_data = np.load('data/eeg/preprocessed_sample_eeg.npy')
    result = predict(mri_data, eeg_data, mri_model, eeg_model)
    
    print(f"NeuroAIDS Detection Result: {result}")

if __name__ == "__main__":
    main()
