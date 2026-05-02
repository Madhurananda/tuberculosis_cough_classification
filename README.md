# Automatic Cough Classification for Tuberculosis Screening

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

This repository contains Python scripts for a tuberculosis (TB) cough classification system based on acoustic analysis of cough recordings. The system extracts handcrafted audio features and applies classical machine learning models combined with patient-level decision rules to produce a binary TB diagnosis.

---

## Overview

The pipeline consists of the following stages:

1. **Cough extraction** – segmentation of cough events from ELAN-annotated audio recordings (main microphone and stethoscope channels).
2. **Feature extraction** – computation of MFCCs (including delta and delta-delta), log filterbanks, zero-crossing rate (ZCR), kurtosis, and log energy. Frame-level features can be grouped and averaged.
3. **Patient-level splitting** – stratified dataset partitioning ensuring no subject appears in multiple splits.
4. **Classifier training** – Logistic Regression, SVM, MLP, and KNN with hyperparameter optimisation via grid search.
5. **Decision rules** – patient-level aggregation using TBI_A, TBI_S, and ADS strategies.
6. **Evaluation** – ROC curves, AUC, sensitivity, specificity, accuracy, and Cohen’s kappa.
7. **Feature reduction** – Sequential Forward Selection (SFS) to identify optimal feature subsets.

---

## Dependencies

Requires **Python 3.12** and the following packages:

```
librosa>=0.10.0
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
soundfile>=0.12.0
matplotlib>=3.7.0
tqdm>=4.65.0
openpyxl>=3.1.0
keras>=2.13.0
tensorflow>=2.13.0
natsort>=8.4.0
```

Install with:

```bash id="g7m2ks"
pip install -r requirements.txt
```

---

## Data Preparation

Organize recordings and annotations as follows:

```
../data/raw_data/sorted_recording/
├── Wu0376/
│   ├── Wu0376_Tr1.wav
│   ├── Wu0376_Tr2.wav
│   ├── Wu0376_Tr3.wav
│   └── Wu0376.eaf
├── ...
└── Data_overview.xlsx
```

Steps:

* Parse `.eaf` files to extract cough segments
* Save segmented cough audio into:

  * `coughs_mc/`
  * `coughs_st-1/`
  * `coughs_st-2/`
* Generate a serialized dictionary (`patients_information`) containing waveforms and metadata

---

## Feature Extraction

Run:

```bash id="y4a1pt"
python feat_extract_mc.py
```

Key parameters:

* `N_MFCCs` – number of MFCC coefficients (e.g., 13, 26, 39)
* `N_frame` – frame length (e.g., 2048)
* `B` – number of grouped frames
* `Avg` – whether to average grouped frames

Output files are stored in:

```
../data/feature_data/features_dataset/
```

Example:

```
mc_features_MFCC=26_Frame=2048_B=1_Avg=True.csv
```

Features include ZCR, kurtosis, log energy, MFCCs, delta and delta-delta coefficients, and TB labels.

---

## Train/Test Splitting

Run:

```bash id="y9ks0w"
python split_dataset_recordings.py
```

Configuration:

* Set `val_sizes_split` in `config.py` (e.g., `[0, 0.1, 0.15, 0.2, 0.25]`)

Output:

```
../data/feature_data/splits/val_size=<val_size>/
```

Each split ensures patient independence across training, validation, and test sets.

---

## Classification & Evaluation

Available classifiers:

```bash id="6z5l4b"
python LR_TB_Class.py
python SVM_TB_Class.py
python MLP_TB_Class.py
python MLP_TF_TB_Class.py
python KNN_TB_Class.py
```

Each script:

* Iterates over feature sets and validation splits
* Performs hyperparameter optimisation (GridSearchCV)
* Applies patient-level aggregation rules (TBI_A, TBI_S, ADS)
* Reports AUC, sensitivity, specificity, accuracy, and kappa
* Saves ROC curves and results to:

```
../data/results/<classifier>_classifier/
```

---

## Feature Reduction (SFS)

Run:

```bash id="o3gq7s"
python feature_reduction.py
```

Functionality:

* Sequential Forward Selection using bootstrap AUC
* Outputs feature subset performance (mean ± error)
* Generates plots showing performance vs. number of features

---

## Results Visualisation

* `predres_plots.py` – histograms of prediction scores
* `plot_ROC_allTools.py` – ROC comparisons
* `audio_analysis.py` – waveform and spectrogram visualization

---

## Citation

If you use this repository, please cite:

Pahar, M., Klopper, M., Reeve, B., Warren, R., Theron, G. and Niesler, T., 2021. *Automatic cough classification for tuberculosis screening in a real-world environment*. Physiological Measurement, 42(10), p.105014.

```bibtex id="j8r3dn"
@article{pahar2021automatic,
  title={Automatic cough classification for tuberculosis screening in a real-world environment},
  author={Pahar, Madhurananda and Klopper, Melissa and Reeve, Byron and Warren, Robin and Theron, Grant and Niesler, Thomas},
  journal={Physiological Measurement},
  volume={42},
  number={10},
  pages={105014},
  year={2021},
  publisher={IOP Publishing}
}
```

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

## Notes

* Designed for real-world TB screening scenarios using cough audio
* Performance depends on recording quality and annotation accuracy
* Patient-level aggregation is critical for reliable diagnosis

---
