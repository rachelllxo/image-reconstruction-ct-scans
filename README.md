# CT Scan Reconstruction and Enhancement using Deep Learning



A deep learning and computer vision based medical imaging project for enhancing and reconstructing CT scan slices using image preprocessing, uncertainty estimation, and trust map visualization.

## Overview

This project focuses on improving the quality and interpretability of CT scan images using image enhancement and reconstruction techniques. The system applies preprocessing operations such as denoising and adaptive histogram equalization, followed by reconstruction and uncertainty-aware trust map generation.

An interactive Streamlit web application allows users to upload CT scan images and visualize:

- Input CT Scan
- Reconstructed/Enhanced Scan
- Trust Map showing reconstruction confidence

---

## Features

- CT scan image enhancement
- Noise reduction using Non-Local Means Denoising
- Contrast enhancement using CLAHE
- Trust map generation using edge detection
- Monte Carlo Dropout based uncertainty estimation
- Interactive Streamlit web interface
- Real-time image visualization

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Streamlit

---

## System Workflow

1. User uploads a CT scan image.
2. Image is resized and preprocessed.
3. Noise reduction is applied.
4. Contrast enhancement improves image visibility.
5. Deep learning reconstruction is performed.
6. Multiple stochastic predictions are generated using Monte Carlo Dropout.
7. Variance-based uncertainty estimation is computed.
8. Trust map is generated to visualize confidence regions.
9. Results are displayed through the Streamlit interface.

---

## Project Structure

```bash
├── app.py
├── model.py
├── trust.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/ct-scan-reconstruction.git
cd ct-scan-reconstruction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

---

## Input

- JPG / PNG CT scan slice images

---

## Output

The application displays:

- Original CT Scan
- Enhanced/Reconstructed Image
- Trust Map

---

## Trust Map

The trust map represents the confidence of reconstruction by estimating uncertainty across multiple stochastic forward passes using Monte Carlo Dropout.

Higher trust regions indicate stable and reliable reconstruction.

---

## Future Improvements

- Integration of advanced CNN/U-Net architectures
- Real low-dose CT reconstruction
- 3D volumetric reconstruction
- Segmentation-assisted reconstruction
- Attention mechanisms for feature enhancement
- Deployment using Docker or cloud platforms

---

## Applications

- Medical image enhancement
- Radiology assistance systems
- Low-dose CT reconstruction research
- Intelligent diagnostic imaging systems
- AI-assisted healthcare tools

---

## Author

Rachel E  
B.Tech CSE (AI & ML)  
Periyar Maniammai Institute of Science and Technology
