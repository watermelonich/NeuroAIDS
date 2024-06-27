import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense

def load_data():
    mri_data = np.load('data/mri/preprocessed_sample_mri.npy')
    labels = np.load('data/mri/labels.npy')  # Assuming you have labels
    return mri_data, labels

def build_mri_model(input_shape):
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape),
        MaxPooling3D((2, 2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    mri_data, labels = load_data()
    model = build_mri_model(mri_data.shape[1:])
    model.fit(mri_data, labels, epochs=10, batch_size=2, validation_split=0.2)
    model.save('models/mri_model.h5')
