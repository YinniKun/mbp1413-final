# mbp1413-final
Final project for MBP1413 Winter 2024, by Richard, Sylvia, and Chris. 

## Description
In this project, we look at the performance between different models of UNet (specifically, UNet and UNet-R) for general nuclei segmentation from various imaging modalities.

This repo contains the report (named `final-report.pdf`) and the presentation (named `final-presentation.pdf`) for this project. The codes used to generate the results used in the report and presentation are also found in this repo.

## Environment Installation
```
cd /path/to/mbp1413-final
conda-env create -f environment.yml
conda activate monai
```

## Usage
```python
python main.py
-c /path/to/config/yaml/file
-m mode. # default is "train", can be "train" or "test"
-d #flag for downloading dataset. Action won't be triggered if not using this flag
-r #flag for resuming the training process. Action won't be triggered if not using this flag
```
