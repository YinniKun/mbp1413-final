#!/usr/bin/env bash
#SBATCH --job-name=run-model
#SBATCH --account=def-gregorys
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/project/6068045/zf2dong/mbp1413/mbp1413-final/outs/%x_%j.out

source /project/6068045/zf2dong/mbp1413/env/pytorch/bin/activate

# echo "For UNet" 
# python -u main.py -c config.yaml -e 50 -l 0.001
# python -u main.py -c config.yaml -m "test" -e 50 -l 0.001
# python -u main.py -c config.yaml -e 100  -l 0.001
# python -u main.py -c config.yaml -m "test" -e 100  -l 0.001
# python -u main.py -c config.yaml -e 150 -l 0.001
# python -u main.py -c config.yaml -m "test" -e 150 -l 0.001
# python -u main.py -c config.yaml -e 200 -l 0.001
# python -u main.py -c config.yaml -m "test" -e 200 -l 0.001

# python -u main.py -c config.yaml -e 50 -l 0.005
# python -u main.py -c config.yaml -m "test" -e 50 -l 0.005
# python -u main.py -c config.yaml -e 100  -l 0.005
# python -u main.py -c config.yaml -m "test" -e 100 -l 0.005
# python -u main.py -c config.yaml -e 150 -l 0.005
# python -u main.py -c config.yaml -m "test" -e 150 -l 0.005
# python -u main.py -c config.yaml -e 200 -l 0.005
# python -u main.py -c config.yaml -m "test" -e 200 -l 0.005

# python -u main.py -c config.yaml -e 50 -l 0.01
# python -u main.py -c config.yaml -m "test" -e 50 -l 0.01
# python -u main.py -c config.yaml -e 100  -l 0.01
# python -u main.py -c config.yaml -m "test" -e 100 -l 0.01
# python -u main.py -c config.yaml -e 150 -l 0.01
# python -u main.py -c config.yaml -m "test" -e 150 -l 0.01
# python -u main.py -c config.yaml -e 200 -l 0.01
# python -u main.py -c config.yaml -m "test" -e 200 -l 0.01

# echo "Done for UNet"
# echo "For UNetR"
# python -u main.py -c config.yaml -e 50 -l 0.001 -mo "unetr"
# python -u main.py -c config.yaml -m "test" -e 50 -l 0.001 -mo "unetr"
# python -u main.py -c config.yaml -e 100  -l 0.001 -mo "unetr"
# python -u main.py -c config.yaml -m "test" -e 100  -l 0.001 -mo "unetr"
# python -u main.py -c config.yaml -e 150 -l 0.001 -mo "unetr"
# python -u main.py -c config.yaml -m "test" -e 150 -l 0.001 -mo "unetr"
# python -u main.py -c config.yaml -e 200 -l 0.001 -mo "unetr"
# python -u main.py -c config.yaml -m "test" -e 200 -l 0.001 -mo "unetr"

# python -u main.py -c config.yaml -e 50 -l 0.005 -mo "unetr"
# python -u main.py -c config.yaml -m "test" -e 50 -l 0.005 -mo "unetr"
# python -u main.py -c config.yaml -e 100  -l 0.005 -mo "unetr"
# python -u main.py -c config.yaml -m "test" -e 100 -l 0.005 -mo "unetr"
# python -u main.py -c config.yaml -e 150 -l 0.005 -mo "unetr"
# python -u main.py -c config.yaml -m "test" -e 150 -l 0.005 -mo "unetr"
# python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr"
# python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr"

python -u main.py -c config.yaml -e 50 -l 0.01 -mo "unetr"
python -u main.py -c config.yaml -m "test" -e 50 -l 0.01 -mo "unetr"
python -u main.py -c config.yaml -e 100  -l 0.01 -mo "unetr"
python -u main.py -c config.yaml -m "test" -e 100 -l 0.01 -mo "unetr"
python -u main.py -c config.yaml -e 150 -l 0.01 -mo "unetr"
python -u main.py -c config.yaml -m "test" -e 150 -l 0.01 -mo "unetr"
python -u main.py -c config.yaml -e 200 -l 0.01 -mo "unetr"
python -u main.py -c config.yaml -m "test" -e 200 -l 0.01 -mo "unetr"
echo "Done for UNetR"
