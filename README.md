# NeuroAIDS Model Detector

This repository contains code and resources for detecting NeuroAIDS symptoms through MRI and EEG scans. The project aims to integrate MRI and EEG data to identify key symptoms such as brain atrophy, white matter lesions, presence of masses or tumors, and seizures or epilepsy, providing a comprehensive diagnostic tool for NeuroAIDS.

## Project Overview

NeuroAIDS, or HIV-associated neurocognitive disorders (HAND), encompasses various neurological complications arising from HIV infection. This project utilizes machine learning and deep learning techniques to detect:

- **Brain Atrophy**
- **White Matter Lesions**
- **Presence of Masses or Tumors** (MRI scans)
- **Seizures and Epilepsy** (EEG scans)

### Features

- **MRI Data Analysis**: Detects structural abnormalities such as brain atrophy, white matter lesions, and masses/tumors.
- **EEG Data Analysis**: Identifies patterns indicative of seizures and epilepsy.
- **Multi-Modal Integration**: Combines results from MRI and EEG analysis to provide a comprehensive diagnosis.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/watermelonich/neuroaids.git
    cd neuroaids
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Preprocessing Data

#### MRI Data

1. Place your MRI files in the `data/mri/` directory.
2. Run the preprocessing script:
    ```sh
    python src/preprocess_mri.py
    ```

#### EEG Data

1. Place your EEG files in the `data/eeg/` directory.
2. Run the preprocessing script:
    ```sh
    python src/preprocess_eeg.py
    ```

### Training Models

#### MRI Model

1. Ensure your preprocessed MRI data and labels are available in `data/mri/`.
2. Run the MRI model training script:
    ```sh
    python src/train_mri_model.py
    ```

#### EEG Model

1. Ensure your preprocessed EEG data and labels are available in `data/eeg/`.
2. Run the EEG model training script:
    ```sh
    python src/train_eeg_model.py
    ```

### Integrating Models and Predicting

1. Ensure your preprocessed MRI and EEG data are available in the respective directories.
2. Run the model integration and prediction script:
    ```sh
    python src/integrate_models.py
    ```

### Full Pipeline

To run the entire pipeline from preprocessing to prediction, use:
```sh
python main.py
```

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the contributors and the community for their support and contributions.