# CCD-GFSOD：Centroid Calibration and Dynamic Constraint for Generalized Few Shot Object Detection
<img width="1800" height="1200" alt="总框架图" src="https://github.com/user-attachments/assets/27937ab0-e505-4371-a76e-f328630ebcbd" />
a novel framework CCD-GFSOD, to reduce novel classes feature instability and alleviate confusion among similar categories. 



# Quick Start

1. Check Requirement
- Linux with Python >= 3.6
- PyTorch >= 1.6 & torchvision that matches the PyTorch version.
- CUDA 10.1, 10.2
- GCC >= 4.9

2. Build DeFRCN
- Clone Code

```bash
  git clone https://github.com/er-muyue/DeFRCN.git
  cd DeFRCN
```

- Create a virtual environment (optional)

```bash
  virtualenv defrcn
  cd /path/to/venv/defrcn
  source ./bin/activate
```

- Install PyTorch 1.6.0 with CUDA 10.1

```bash
  pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

- Install Detectron2

```bash
  python3 -m pip install detectron2==0.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
```

- Install other requirements.

```bash
  python3 -m pip install -r requirements.txt
```

3. Prepare Data and Weights
