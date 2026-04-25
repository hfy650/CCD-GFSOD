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
- **Data Preparation**

  - We evaluate our models on two datasets for both FSOD and G-FSOD settings:

    | Dataset   | Size  | GoogleDrive | BaiduYun | Note                  |
    |-----------|-------|-------------|----------|-----------------------|
    | VOC2007   | 0.8G  | download    | download | -                     |
    | VOC2012   | 3.5G  | download    | download | -                     |
    | vocsplit  | <1M   | download    | download | refer from TFA        |
    | COCO      | ~19G  | -           | -        | download from offical |
    | cocosplit | 174M  | download    | download | refer from TFA        |

  - Unzip the downloaded data-source to `datasets` and put it into your project directory:
    ...
datasets
  | -- coco (trainval2014/*.jpg, val2014/*.jpg, annotations/*.json)
  | -- cocosplit
  | -- VOC2007
  | -- VOC2012
  | -- vocsplit
defrcn
tools
...

4. Training and Evaluation
For ease of training and evaluation over multiple runs, we integrate the whole pipeline of few-shot object detection into one script `run_*.sh`, including base pre-training and novel-finetuning (G-FSOD).

- To reproduce the results on VOC, `EXP_NAME` can be any string (e.g CCD-GFSOD, or something) and `SPLIT_ID` must be `1` or `2` or `3` (we consider 3 random splits like other papers).

```bash
  bash run_voc.sh EXP_NAME SPLIT_ID (1, 2 or 3)
```

- To reproduce the results on COCO, `EXP_NAME` can be any string (e.g CCD-GFSOD, or something)

```bash
  bash run_coco.sh EXP_NAME
```

- Please read the details of few-shot object detection pipeline in `run_*.sh`, you need change `IMAGENET_PRETRAIN*` to your path.
