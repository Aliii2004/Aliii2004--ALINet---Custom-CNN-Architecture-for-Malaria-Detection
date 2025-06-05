# ALINet: Custom CNN Architecture for Malaria Detection

## Overview
ALINet is a custom Convolutional Neural Network (CNN) architecture designed for automated malaria detection from cell images. This project implements a deep learning solution to assist in the diagnosis of malaria through cell image analysis.

## Project Structure
```
├── pytorch_cnn_alinet_ai.ipynb   # Main notebook containing model implementation
├── requirements.txt              # Project dependencies
└── README.md                    # Project documentation
```

## Requirements
- Python 3.8+
- PyTorch 2.0.0+
- CUDA (optional, for GPU acceleration)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ALINet-Custom-CNN-Architecture-for-Malaria-Detection.git
cd ALINet-Custom-CNN-Architecture-for-Malaria-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Architecture
ALINet consists of:
- 3 Convolutional blocks (Conv2D + BatchNorm + ReLU + MaxPool)
- Fully connected layers with dropout
- Output layer for binary classification

## Usage
1. Open `pytorch_cnn_alinet_ai.ipynb` in Jupyter Notebook or JupyterLab
2. Follow the notebook cells for:
   - Data preparation
   - Model training
   - Evaluation
   - Making predictions

## Dataset
The model is designed to work with the malaria cell images dataset. Each image should be preprocessed to 64x64 pixels in RGB format.

## Performance
- Validation Accuracy: 97.89%
- Accuracy: 98.22% on test set
- Precision: 98.22%
- Recall: 98.22%
- F1-Score: 98.22%

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions and feedback, please open an issue in the repository.
