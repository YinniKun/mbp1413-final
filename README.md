<!--
 * @Author: Chris Xiao yl.xiao@mail.utoronto.ca
 * @Date: 2024-03-31 01:14:18
 * @LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
 * @LastEditTime: 2024-04-02 02:25:38
 * @FilePath: /mbp1413-final/README.md
 * @Description: README file for the final project of MBP1413 Winter 2024
 * I Love IU
 * Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
-->
# Evaluating and Optimizing Training and Interference Performances Between Variations of UNET Models for Nuclei Detection and Segmentation

This is the repo for the final project of MBP1413 Winter 2024, by Richard, Sylvia, and Chris.

## Description

In this project, we looked at the performance between different models of UNet (specifically, UNET and UNETR) for general nuclei segmentation from various imaging modalities and tried to optimize the model performances with various data processing methods and hyperparameter tuning.

For details of the findings, please refer to the report (named `final-report.pdf`) and the presentation (named `final-presentation.pdf`) for this project in this ```/documents```. The codes used to generate the results used in the report and presentation are also found in this repo.

## Data Availability

The data used to train the models, as well as the unprocessed results (such as loss curves, model validation results, etc.) of this project can be found [at this Google Drive](https://drive.google.com/drive/folders/1Gf1jCM_4Zove3mqOsA3wmld1ZJUHG08A?usp=sharing)

## Environment Installation (tested on Ubuntu 22.04)

### Prerequisites

For Mac Users

```bash
brew install graphviz
```

For Debian System Users (Ubuntu, etc.)

```bash
sudo apt install graphviz
```

For Redhat System Users (CentOS, Fedora, etc.)

```bash
sudo yum install graphviz
```

### Conda Environment Setup

```bash
git clone https://github.com/YinniKun/mbp1413-final.git
cd mbp1413-final
conda-env create -f environment.yml
conda activate monai
```

## Usage

To run locally, use:

```bash
python main.py
-c /path/to/config/yaml/file
-m train # default is train, can be train or test
-d # flag for downloading dataset. Action won't be triggered if not using this flag
-r # flag for resuming the training process. Action won't be triggered if not using this flag
-e 200 # the epochs number, default is 200
-l 0.005 # the learning rate, default is 0.005 
-mo unetr # the model to be trained/tested, default is unet
-sch # flag for using lr_scheduler
-opt SGD # deafult is Adam, can be SGD or Adam
-sa # flag for saving the model architecture plot
```

Detailed information can be shown using ```-h```\
To run on Compute Canada, use:

```bash
sbatch run.sh
```
