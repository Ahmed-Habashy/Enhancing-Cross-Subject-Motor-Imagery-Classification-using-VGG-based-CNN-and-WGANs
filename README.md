# Enhancing-Cross-Subject-Motor-Imagery-Classification-using-VGG-based-CNN-and-WGANs

### README

This repository contains the code and data for the paper "Toward Calibration-Free Motor Imagery Brain-Computer Interfaces: A VGG-based Convolutional Neural Network and WGAN Approach"by A. G. Habashi, Ahmed M. Azab, Seif Eldawlatly, and Gamal M. Aly. The paper proposes a novel framework for cross-subject motor imagery (MI) classification using electroencephalogram (EEG) signals. The framework employs a Wasserstein Generative Adversarial Network (WGAN) for data augmentation and a modified VGG-based Convolutional Neural Network (CNN) for classification.

#### Data

The code is implemented using one benchmark dataset (BCI Competition IV-2B) as an example. However, the evaluation in the paper was conducted on three benchmark datasets (BCI Competition IV-2B, IV-1, and IV-2A). The data can be downloaded from the BCI Competition IV website: [https://www.bbci.de/competition/iv/](https://www.bbci.de/competition/iv/)

#### Preprocessing

The preprocessing steps involve filtering the EEG data, extracting MI task segments, computing Short-Time Fourier Transform (STFT) spectrograms, and merging and resizing the spectrograms into grayscale images. The preprocessing scripts for each dataset are provided in the repository:

*   `preprocessing dataset 2B.py`
*   `preprocessing dataset 1.py`
*   `preprocessing dataset 2A.py`

#### WGAN Training

The WGAN is trained to generate synthetic EEG spectrum images that are similar to the real data. The WGAN architecture and training process are implemented in the `func_cnn_Wgan.docx` script.

#### CNN Training and Testing

The modified VGG-based CNN classifier is trained on the augmented dataset (real and synthetic images). The training and testing process is implemented in the `Gray cross-WGAN-CNN.docx` script. The script also includes the implementation of two other CNN classifiers (VGG-based CNN and Audio-spectrum CNN) for comparison.

#### Evaluation

The framework is evaluated using a leave-one-subject-out cross-validation scheme. The classification accuracy is used as the performance metric. The results demonstrate that the proposed framework outperforms state-of-the-art methods in cross-subject MI classification.

#### Requirements

*   Python 3.9.7
*   TensorFlow
*   Keras
*   NumPy
*   SciPy
*   Matplotlib
*   gumpy

#### Usage

1.  Download the datasets from the BCI Competition IV website.
2.  Preprocess the datasets using the provided scripts.
3.  Train the WGAN using the `func_cnn_Wgan.py` script.
4.  Train and test the CNN classifiers using the `Gray cross-WGAN-CNN.py` script.

#### Note

The code provided in this repository is for research purposes only. It may require modifications to be used in real-world applications.
