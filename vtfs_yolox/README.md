# YOLOX for VIA Traffic Sign dataset

These instructions guide you to train and infer the YOLOX model on [VIA Traffic Sign dataset](https://github.com/makerhanoi/via-datasets).

## 1. Prepare environment

- Clone and install YOLOX using instructions from the official repository:

```
git clone https://github.com/Megvii-BaseDetection/YOLOX
git checkout ac58e0a
```

- Create a conda environment (or virtualenv) and activate it (optional):

```
conda create -n tfs python=3.9 -y
conda activate tfs
```

- Install YOLOX:

```
pip install torch==2.0.1
pip install -e .
```

- Clone source code for training with VIA Traffic Sign dataset:

```
cd YOLOX
git clone https://github.com/vietanhdev/vtfs_yolox.git
```

## 2. Dataset preparation

Download and extract VIA traffic sign dataset:

```
bash vtfs_yolox/download_data.sh
```

## 3. Training

```
export YOLOX_DATADIR=datasets/vtfs/
python3 tools/train.py -f vtfs_yolox/exps/tfs_nano.py -b 8
```
