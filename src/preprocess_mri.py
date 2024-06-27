import nibabel as nib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_mri_data(mri_path):
    mri_image = nib.load(mri_path).get_fdata()
    return mri_image

def preprocess_mri_data(mri_image):
    scaler = StandardScaler()
    mri_image = scaler.fit_transform(mri_image.reshape(-1, mri_image.shape[-1])).reshape(mri_image.shape)
    return mri_image

if __name__ == "__main__":
    mri_path = 'data/mri/sample_mri.nii'
    mri_image = load_mri_data(mri_path)
    mri_image = preprocess_mri_data(mri_image)
    np.save('data/mri/preprocessed_sample_mri.npy', mri_image)
