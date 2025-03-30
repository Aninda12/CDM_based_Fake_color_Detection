# Fake Colorized Image Detection

## Overview
This project focuses on detecting fake colorized images using a **Channel Difference Map (CDM)** based model. It involves data preprocessing, model training, evaluation, and batch image detection using deep learning techniques.

## Features
- **CDM-Based Image Processing**: Extracts critical color differences to enhance fake detection.
- **Deep Learning Classification**: Utilizes a trained neural network to classify images as real or fake.
- **Visualization Tools**: Includes activation mapping and ROC curve plotting.
- **Batch Image Detection**: Supports testing multiple images at once.

## Project Structure
```
IPPROJECT/
├── .venv/                      # Virtual environment
├── data/
│   ├── fake/                   # Folder for fake images
│   └── real/                   # Folder for real images
├── src/
│   ├── __pycache__/
│   ├── cdm_utils.py            # Utilities for generating CDM features
│   ├── data_utils.py           # Data loading and preprocessing
│   ├── evaluation.py           # Model evaluation and testing
│   ├── models.py               # Neural network architectures
│   ├── training.py             # Model training script
│   ├── visualization.py        # Visualization utilities
├── outputs/                    # Model weights and logs
├── main.py                     # Main script for training and testing
├── test_script.py              # Custom user input testing script
├── req.txt                     # Required dependencies
├── README.md                   # Project documentation
└── structure.txt                # Project structure description
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ayushkumar912/CDM_based_Fake_Image_Detection.git
   cd CDM_based_Fake_Image_Detection
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   .venv\Scripts\activate     # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r req.txt
   ```

## Usage
### Training the Model
Run the training script to train the detection model:
```bash
python main.py --mode train --epochs 50 --batch_size 32
```

### Testing with Custom Input
To classify a single image with user input:
```bash
python test_script.py
```
It will prompt you to enter the image path and return the classification result.

### Batch Testing
To test multiple images at once:
```bash
python main.py --mode test --test_folder ./data/test_images/ --model_path outputs/models/detection_model.h5
```

## Results
- **ROC Curve & Activation Mapping**: The visualization script can generate these insights.
- **Classification Accuracy**: Outputs confidence scores for real and fake predictions.

## Contributors
- **Ayush Gangwar**  
  GitHub: [AYUSH-GANGWAR9](https://github.com/AYUSH-GANGWAR9)

## License
This project is open-source and available under the [MIT License](LICENSE).

