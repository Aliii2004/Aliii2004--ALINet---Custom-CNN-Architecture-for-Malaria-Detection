# ALINet: Custom CNN Architecture for Malaria Detection

## Overview
ALINet is a custom Convolutional Neural Network (CNN) architecture designed for automated malaria detection from cell images. This project implements a deep learning solution to assist in the diagnosis of malaria through cell image analysis.

## Project Structure
```
├── pytorch_cnn_alinet.py   # Main notebook containing model implementation
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
ALINet implements a sophisticated CNN architecture specifically designed for malaria cell classification:

### Input Processing
- Input: RGB images (3 channels)
- Image resolution: 64x64 pixels
- Normalization: Standard scaling with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Convolutional Blocks (4 blocks)
Each convolutional block follows a consistent pattern:

**Block 1: Initial Feature Extraction**
- Input: 3 channels → Output: 64 channels
- First Conv2D: 3×3 kernel, padding=1
- Second Conv2D: 3×3 kernel, padding=1
- Each Conv2D followed by BatchNorm2D + ReLU
- MaxPool2D: 2×2, stride=2
- Output size: 32×32

**Block 2: Feature Enhancement**
- Input: 64 channels → Output: 128 channels
- Two Conv2D layers: 3×3 kernel, padding=1
- BatchNorm2D + ReLU after each Conv2D
- MaxPool2D: 2×2, stride=2
- Output size: 16×16

**Block 3: Deep Feature Extraction**
- Input: 128 channels → Output: 256 channels
- Two Conv2D layers: 3×3 kernel, padding=1
- BatchNorm2D + ReLU after each Conv2D
- MaxPool2D: 2×2, stride=2
- Output size: 8×8

**Block 4: Final Feature Processing**
- Input: 256 channels → Output: 512 channels
- Two Conv2D layers: 3×3 kernel, padding=1
- BatchNorm2D + ReLU after each Conv2D
- MaxPool2D: 2×2, stride=2
- Output size: 4×4

### Fully Connected Layers (Classifier)
1. **Flatten Layer**
   - Transforms feature maps to 1D vector
   - Input: 512×4×4 = 8192 features

2. **Dense Layer 1**
   - 8192 → 4096 neurons
   - ReLU activation
   - Dropout rate: 0.5

3. **Dense Layer 2**
   - 4096 → 1024 neurons
   - ReLU activation
   - Dropout rate: 0.5

4. **Output Layer**
   - 1024 → 2 neurons
   - Softmax activation
   - Binary classification: infected/uninfected

### Architecture Benefits
- **Deep Feature Extraction**: Progressive increase in channels (3→64→128→256→512)
- **Regularization**: Multiple dropout layers prevent overfitting
- **Training Stability**: BatchNormalization after each convolution
- **Spatial Information**: Proper padding maintains feature details
- **Computational Efficiency**: Balanced depth and width for optimal performance

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
