# Q2 Reproduction Guide

## Files
- `train.py`: trains baseline and reduced disentangled speaker model
- `eval.py`: evaluates saved checkpoints
- `configs/config.yaml`: main experiment configuration
- `results/`: stores CSV tables and plots

## Run
```bash
python q2/train.py
python q2/eval.py