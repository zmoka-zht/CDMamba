<div align="center">
    <h2>
        CDMamba: Remote Sensing Image Change Detection with Mamba
    </h2>
</div>
<br>

<div align="center">
  <img src="resources/CDMamba.png" width="800"/>
</div>
<br>
<div align="center">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2406.04207">
    <span style="font-size: 20px; ">arXiv</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="resources/RSMamba.pdf">
    <span style="font-size: 20px; ">PDF</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
</div>
<br>
<br>

[![GitHub stars](https://badgen.net/github/stars/zmoka-zht/CDMamba)](https://github.com/zmoka-zht/CDMamba)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2406.04207-b31b1b.svg)](https://arxiv.org/abs/2406.04207)


<br>
<br>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>


## Introduction

This repository is the code implementation of the paper [CDMamba: Remote Sensing Image Change Detection with Mamba](https://arxiv.org/abs/2406.04207)

The current branch has been tested on Linux system, PyTorch 2.1.0 and CUDA 12.1, supports Python 3.10.

If you find this project helpful, please give us a star ⭐️, your support is our greatest motivation.


## Updates

🌟 **2024.06.20** Released the RSMamba project.

## TODO

- [X] Open-sourced the [weight files]

## Table of Contents

- [Introduction](#Introduction)
- [TODO](#TODO)
- [Table of Contents](#Table-of-Contents)
- [Installation](#Installation)
- [Dataset Preparation](#Dataset Preparation)
- [Model Training and Testing](#Model Training and Testing)
- [Citation](#Citation)
- [License](#License)
- [Contact Us](#Contact-Us)

## Installation

### Requirements

- Linux system, Windows is not tested, depending on whether `causal-conv1d` and `mamba-ssm` can be installed
- Python 3.8+, recommended 3.10
- PyTorch 2.0 or higher, recommended 2.1.0
- CUDA 11.7 or higher, recommended 12.1

### Environment Installation

It is recommended to use Miniconda for installation. The following commands will create a virtual environment named `cd_mamba` and install PyTorch. In the following installation steps, the default installed CUDA version is **12.1**. If your CUDA version is not 12.1, please modify it according to the actual situation.

Note: If you are experienced with PyTorch and have already installed it, you can skip to the next section. Otherwise, you can follow the steps below.

<details open>

**Step 0**: Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).

**Step 1**: Create a virtual environment named `cd_mamba` and activate it.

```shell
conda create -n cd_mamba python=3.10
conda activate cd_mamba
```

**Step 2**: Install dependencies.

```shell
pip install -r requirements.txt
```
**Note**: If importing mamba fails, try the following
```shell
pip install causal-conv1d==1.2.0.post2
pip install mamba-ssm==1.2.0.post1
```

</details>


### Install CDMamba


You can download or clone the CDMamba repository.

```shell
git clone git@github.com:zmoka-zht/CDMamba.git
cd CDMamba
```

## Dataset Preparation


### Remote Sensing Change Detection Dataset

We provide the method of preparing the remote sensing change detection dataset used in the paper.

#### WHU-CD Dataset

- Data download link: [WHU-CD Dataset PanBaiDu](https://pan.baidu.com/s/1nh7znToO4XwaZHIOo7gCmw). Code:t2sb


#### LEVIR-CD Dataset 

- Data download link: [LEVIR-CD Dataset PanBaiDu](https://pan.baidu.com/s/1s5352sCRLxu50w2cEfSvWA). Code:qlvs


#### LEVIR+-CD Dataset

- Data download link: [LEVIR+-CD Dataset PanBaiDu](https://pan.baidu.com/s/1ymcsUei7oDyyMUBbpUTGAw ). Code: xtj8


#### Organization Method

You can also choose other sources to download the data, but you need to organize the dataset in the following format：

```
${DATASET_ROOT} # Dataset root directory, for example: /home/username/data/LEVIR-CD
├── A
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├──...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├──...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── B
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├──...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├──...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── label
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├──...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├──...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── list
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
```

## Model Training and Testing

All configuration for model training and testing are stored in the local folder `config`

#### Example of Training on LEVIR-CD Dataset

```shell
python train.py --config/mamba/levir_cdmamba.json 
```

#### Example of Testing on LEVIR-CD Dataset

```shell
python test.py --config/mamba/levir_test_cdmamba.json 
```

## Citation

If you use the code or performance benchmarks of this project in your research, please refer to the following bibtex citation of CDMamba.

```
@article{zhang2024cdmamba,
  title={CDMamba: Remote Sensing Image Change Detection with Mamba},
  author={Zhang, Haotian and Chen, Keyan and Liu, Chenyang and Chen, Hao and Zou, Zhengxia and Shi, Zhenwei},
  journal={arXiv preprint arXiv:2406.04207},
  year={2024}
}
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Contact Us

If you have any other questions❓, please contact us in time 👬
