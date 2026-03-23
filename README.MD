Speech Understanding Assignment 1

End-to-End Speech Processing, Representation Learning & Ethical AI Pipeline

## Overview

This project implements a complete speech processing pipeline covering:

Q1: Manual feature extraction (MFCC, spectral analysis, voiced/unvoiced detection, phonetic mapping)
Q2: Speaker recognition with baseline vs disentangled representation learning
Q3: Ethical AI pipeline (dataset audit, privacy-preserving transformation, fairness-aware training)

The system is designed to be:

Reproducible (Docker)
Automated (main pipeline execution)
Structured for research-style reporting
## Project Structure
speech_assignment/
│
├── main.py                  # Runs entire pipeline (Q1 + Q2 + Q3)
├── Dockerfile
├── requirements.txt
├── README.md
│
├── q1/
│   ├── mfcc_manual.py
│   ├── leakage_snr.py
│   ├── voiced_unvoiced.py
│   └── phonetic_mapping.py
│
├── q2/
│   ├── train.py
│   ├── eval.py
│   └── configs/config.yaml
│
├── q3/
│   ├── audit.py
│   ├── privacymodule.py
│   ├── pp_demo.py
│   ├── train_fair.py
│   └── evaluation_scripts/
│       └── proxy_metrics.py
│
├── data/
│   ├── q1_audio/
│   ├── q2_audio/
│   ├── q3_audio/
│   ├── q2_metadata.csv
│   ├── q3_metadata.csv
│   └── q3_train.csv
│
└── outputs/
## Setup Instructions
Option 1: Local (Conda)
conda create -n speech_assign python=3.10 -y
conda activate speech_assign
pip install -r requirements.txt
Option 2: Docker (Recommended)

Build image:

docker build --no-cache -t speech-assignment .

Run full pipeline:

docker run --rm -v ${PWD}:/app speech-assignment
## Run the Full Pipeline
python main.py

This executes:

## Q1
Manual MFCC extraction
Spectral leakage & SNR comparison
Voiced/Unvoiced detection
Phonetic boundary mapping (Wav2Vec2)
## Q2
Baseline speaker model
Reduced disentangled model
Training + evaluation
## Q3
Dataset bias audit
Privacy-preserving transformation
Fairness-aware training
Proxy evaluation
## Data Requirements
## Q1
Place at least one WAV file in:

data/q1_audio/
## Q2
data/q2_audio/
├── Audio1.wav
├── Audio2.wav
├── Audio3.wav

data/q2_metadata.csv:

file_name,speaker_id,environment_id
Audio1.wav,0,0
Audio2.wav,1,1
Audio3.wav,2,0
## Q3
data/q3_audio/
├── Audio1.wav
├── Audio2.wav
├── Audio3.wav

data/q3_metadata.csv:

file_name,gender,age_group,dialect
Audio1.wav,male,young,standard
Audio2.wav,female,adult,regional
Audio3.wav,male,adult,regional

data/q3_train.csv:

file_name,label,group_id
Audio1.wav,0,0
Audio2.wav,1,1
Audio3.wav,2,0

## Outputs

Generated inside:

outputs/

Includes:

MFCC plots
Spectral comparison graphs
Voiced/unvoiced segmentation
Phonetic mapping + RMSE
Q2 training history & evaluation
Q3 audit plots
Privacy transformation audio
Fairness model outputs

## Methodology Summary
Q1: Signal Processing
Manual MFCC pipeline (no built-in MFCC)
Windowing comparison (Rectangular, Hamming, Hanning)
Cepstrum-based voiced/unvoiced segmentation
Wav2Vec2-based phonetic boundary reference
Q2: Representation Learning
Baseline: Fully connected speaker classifier
Proposed: Reduced disentangled model
Speaker head
Environment head
Loss:
Loss = Speaker Loss - λ * Environment Loss
Q3: Ethical AI
Dataset Audit
Distribution analysis (gender, age, dialect)
Missing data detection
Privacy Module
Neural transformation to obscure identity
Fairness Training
Loss = Task Loss + λ * Fairness Loss
Evaluation
Proxy SNR
Output comparison

## Notes
All file paths are dynamic (Docker-safe)
Filenames must match CSV exactly (case-sensitive in Docker)
First run downloads Wav2Vec2 model (~378MB)

## Key Features
Fully automated pipeline (main.py)
Docker reproducibility
Modular structure (Q1, Q2, Q3)
Research-style implementation
Beginner-friendly yet extensible

## Future Improvements
Use larger datasets (LibriSpeech, Common Voice)
Replace proxy metrics with DNSMOS/FAD
Add GPU acceleration
Improve disentanglement architecture
Real voice anonymization (formant shifting, VC models)

## Author
Purushothaman Shanmugam
M23DE1033
M-Tech Data Engineerig
Indian Institute of Technology Jodhpur

## License

This project is for academic use.