#!/usr/bin/env bash
#SBATCH --job-name=run-model
#SBATCH --account=def-gregorys
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/project/6068045/zf2dong/mbp1413/mbp1413-final/outs/%x_%j.out

source /project/6068045/zf2dong/mbp1413/env/pytorch/bin/activate

print("For UNet") 
python main.py -c config.yaml -e 50 -l 0.001
python main.py -c config.yaml -m "test" -e 50 -l 0.001
python main.py -c config.yaml -e 100  -l 0.001
python main.py -c config.yaml -m "test" -e 100  -l 0.001
python main.py -c config.yaml -e 150 -l 0.001
python main.py -c config.yaml -m "test" -e 150 -l 0.001
python main.py -c config.yaml -e 200 -l 0.001
python main.py -c config.yaml -m "test" -e 200 -l 0.001

python main.py -c config.yaml -e 50 -l 0.005
python main.py -c config.yaml -m "test" -e 50 -l 0.005
python main.py -c config.yaml -e 100  -l 0.005
python main.py -c config.yaml -m "test" -e 100 -l 0.005
python main.py -c config.yaml -e 150 -l 0.005
python main.py -c config.yaml -m "test" -e 150 -l 0.005
python main.py -c config.yaml -e 200 -l 0.005
python main.py -c config.yaml -m "test" -e 200 -l 0.005

python main.py -c config.yaml -e 50 -l 0.01
python main.py -c config.yaml -m "test" -e 50 -l 0.01
python main.py -c config.yaml -e 100  -l 0.01
python main.py -c config.yaml -m "test" -e 100 -l 0.01
python main.py -c config.yaml -e 150 -l 0.01
python main.py -c config.yaml -m "test" -e 150 -l 0.01
python main.py -c config.yaml -e 200 -l 0.01
python main.py -c config.yaml -m "test" -e 200 -l 0.01

print("For UNetR")
python main.py -c config.yaml -e 50 -l 0.001 -mo "unetr"
python main.py -c config.yaml -m "test" -e 50 -l 0.001 -mo "unetr"
python main.py -c config.yaml -e 100  -l 0.001 -mo "unetr"
python main.py -c config.yaml -m "test" -e 100  -l 0.001 -mo "unetr"
python main.py -c config.yaml -e 150 -l 0.001 -mo "unetr"
python main.py -c config.yaml -m "test" -e 150 -l 0.001 -mo "unetr"
python main.py -c config.yaml -e 200 -l 0.001 -mo "unetr"
python main.py -c config.yaml -m "test" -e 200 -l 0.001 -mo "unetr"

python main.py -c config.yaml -e 50 -l 0.005 -mo "unetr"
python main.py -c config.yaml -m "test" -e 50 -l 0.005 -mo "unetr"
python main.py -c config.yaml -e 100  -l 0.005 -mo "unetr"
python main.py -c config.yaml -m "test" -e 100 -l 0.005 -mo "unetr"
python main.py -c config.yaml -e 150 -l 0.005 -mo "unetr"
python main.py -c config.yaml -m "test" -e 150 -l 0.005 -mo "unetr"
python main.py -c config.yaml -e 200 -l 0.005 -mo "unetr"
python main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr"

python main.py -c config.yaml -e 50 -l 0.01 -mo "unetr"
python main.py -c config.yaml -m "test" -e 50 -l 0.01 -mo "unetr"
python main.py -c config.yaml -e 100  -l 0.01 -mo "unetr"
python main.py -c config.yaml -m "test" -e 100 -l 0.01 -mo "unetr"
python main.py -c config.yaml -e 150 -l 0.01 -mo "unetr"
python main.py -c config.yaml -m "test" -e 150 -l 0.01 -mo "unetr"
python main.py -c config.yaml -e 200 -l 0.01 -mo "unetr"
python main.py -c config.yaml -m "test" -e 200 -l 0.01 -mo "unetr"