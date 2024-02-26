#!/usr/bin/env bash
#SBATCH --job-name=run-model
#SBATCH --account=def-gregorys
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/project/6068045/zf2dong/mbp1413/mbp1413-final/outs/%x_%j.out

source /project/6068045/zf2dong/mbp1413/env/pytorch/bin/activate
 
python main.py -c config.yaml -e 50 -l 0.001
python main.py -c config.yaml -r -e 100  -l 0.001
python main.py -c config.yaml -r -e 150 -l 0.001
python main.py -c config.yaml -r -e 150 -l 0.001

python main.py -c config.yaml -e 50 -l 0.005
python main.py -c config.yaml -r -e 100  -l 0.005
python main.py -c config.yaml -r -e 150 -l 0.005
python main.py -c config.yaml -r -e 150 -l 0.005

python main.py -c config.yaml -e 50 -l 0.01
python main.py -c config.yaml -r -e 100  -l 0.01
python main.py -c config.yaml -r -e 150 -l 0.01
python main.py -c config.yaml -r -e 150 -l 0.01