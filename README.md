# mbp1413-final
Final project for MBP1413 Winter 2024, by Richard, Sylvia, and Chris. 

## Description
In this project, we look at the performance between different models of UNet (specifically, UNet and UNet-R) for general nuclei segmentation from various imaging modalities.

This repo contains the report (named `final-report.pdf`) and the presentation (named `final-presentation.pdf`) for this project. The codes used to generate the results used in the report and presentation are also found in this repo.

## Environment Installation
```bash
cd /path/to/mbp1413-final
conda-env create -f environment.yml
conda activate monai
```

## Usage
To run locally, use:
```python
python main.py
-c /path/to/config/yaml/file
-m mode # default is "train", can be "train" or "test"
-d #flag for downloading dataset. Action won't be triggered if not using this flag
-r #flag for resuming the training process. Action won't be triggered if not using this flag
-e 200 #the epochs number, default is 200
-l 0.001 #the learning rate, default is 0.001 
-mo "unetr" #the model to be trained/tested, default is "unet"
```
To run on Compute Canada, use:
```bash
sbatch run.sh
```

## Data Availability
- The data used to train the models can be found [here](https://drive.google.com/file/d/1R6yO4DtEmsLRtkCroTeSNeRE78wZSfWi/view?usp=drive_link)
- The unprocessed results (such as loss curves, model validation results, etc.) of this project can be found [here](https://drive.google.com/drive/folders/1Gf1jCM_4Zove3mqOsA3wmld1ZJUHG08A?usp=share_link)

