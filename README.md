# Mapping Morphological Signatures with Rotation-Invariant Scattering Networks

This repository contains the official codebase for the Bsc dissertation: **"Mapping Morphological Signatures: Using Rotation-Invariant Morlet Scattering Networks to Identify and Interpret Galaxies."** This project implements an unsupervised, $SE(2)$ rotation-invariant scattering network to extract geometrically stable morphological features from astronomical surveys. It evaluates the critical trade-off between explicit mathematical invariance (Rigid Motion) and learned invariance (ResNet50V2 CNNs), utilizing the Galaxy10 DECals dataset as a benchmark for the Hubble Flow continuum.

### The Dataset: Galaxy10 DECals
This pipeline is built to process the **[Galaxy10 DECals Dataset](https://astronn.readthedocs.io/en/latest/galaxy10.html)**, an astrophysical dataset containing 17,736 images across 10 broad morphological classes. 

<p align="center">
  <img src="https://astronn.readthedocs.io/en/latest/_images/galaxy10_example.png" alt="Galaxy10 DECals Classes" width="800"/>
  <br>
  <em>Example samples from the Galaxy10 DECals dataset demonstrating the morphological continuum from smooth ellipticals to disturbed mergers.</em>
</p>

---

## Installation & Setup

It is highly recommended to run this pipeline within an isolated Python Virtual Environment to prevent dependency conflicts (especially regarding GPU-accelerated libraries like CuPy and TensorFlow).

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```


2. Create and activate a virtual environment:

On macOS and Linux:

```
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:

```
pip install --upgrade pip
pip install -r requirements.txt
```

(Note: For GPU acceleration, ensure you have the correct CUDA toolkit installed for your system to support CuPy and TensorFlow/PyTorch).

Repository Structure
The codebase is organized into sequential stages of the analytical pipeline: Pre-processing, Embedding Extraction, and Topological Analysis.

.
├── .gitignore                  # Standard Git ignore file
├── requirements.txt            # Python dependencies (NumPy, SciPy, CuPy, UMAP, HDBSCAN, etc.)
│
├── Pre-processing & Utilities
│   ├── pre_process_data.py     # Cosmological scaling (7.35/z), sky subtraction, and Arcsinh stretching
│   ├── sky.py                  # Custom Sigma-clipping algorithms for sky background calculation
│   ├── sky_subtract.ipynb      # Jupyter notebook for prototyping/visualizing background noise removal
│   └── view_data.ipynb         # Interactive HDF5 dataset explorer and visualization prototype
│
├── Wavelet & Scattering Core
│   ├── create_wavelets.py      # Generates the Morlet wavelet filter banks (phi and psi filters)
│   ├── 1dwavelets.py           # Foundational 1D wavelet exploration and testing utilities
│   ├── rigid_motion_embedding.py # Core $SE(2)$ invariant joint rigid motion scattering implementation
│   └── kymatio_embedding.py    # Baseline extraction using the standard (covariant) Kymatio library
│
├── Deep Learning Baseline
│   ├── cnn_embedding.py        # Extracts 2048-d feature vectors from the ResNet50V2 baseline
│   └── modelCNNpaper.py        # CNN architecture from A post-merger enhancement only in star-forming Type 2 Seyfert galaxies: the deep learning view
│
└── Manifold Analysis
    └── cluster_pipeline.py     # UMAP projection, HDBSCAN clustering, Silhouette sc


