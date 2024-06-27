import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def load_data():
    eeg_data = np.load('data/eeg/preprocessed_sample_eeg.npy')
    labels = np.load('data/eeg/labels.npy')  # Assuming you have labels
    return eeg_data, labels

def build_eeg_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    eeg_data, labels = load_data()
    model = build_eeg_model(eeg_data.shape[1:])
    model.fit(eeg_data, labels, epochs=10, batch_size=2, validation_split=0.2)
    model.save('models/eeg_model.h5')
