# mbp1413-final
Final project for MBP1413 Winter 2024, by Richard, Sylvia, and Chris

## Installation
```
cd /path/to/mbp1413-final
conda-env create -f environment.yml
conda activate monai
```

## Usage
```
python main.py
-c /path/to/config/yaml/file
-m running mode. default is "train", can be "train" or "test"
-d download dataset, action wont be triggerd if not using flag
-r resume training process, action wont be triggerd if not using flag
```
