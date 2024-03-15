#!/usr/bin/env bash
#SBATCH --job-name=train-model-all
#SBATCH --account=def-gregorys
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/project/6068045/zf2dong/mbp1413/mbp1413-final/outs/%x_%j.out

source /project/6068045/zf2dong/mbp1413/env/pytorch/bin/activate

echo "Exp 2"
echo "Training and testing UNet"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "Adam" -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "Adam" -sch
echo "Training and testing UNeTr"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "Adam" -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "Adam" -sch

echo "Exp 3"
echo "Training and testing UNet"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "Adam" -no
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "Adam" -no
echo "Training and testing UNeTr"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "Adam" -no
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "Adam" -no

echo "Exp 4"
echo "Training and testing UNet"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "Adam" -no -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "Adam" -no -sch
echo "Training and testing UNeTr"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "Adam" -no -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "Adam" -no -sch

echo "Exp 5"
echo "Training and testing UNet"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "SGD" 
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "SGD"
echo "Training and testing UNeTr"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "SGD" 
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "Adam"

echo "Exp 6"
echo "Training and testing UNet"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "SGD" -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "SGD" -sch
echo "Training and testing UNeTr"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "SGD" -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "Adam" -sch

echo "Exp 7"
echo "Training and testing UNet"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "SGD" -no
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "SGD" -no
echo "Training and testing UNeTr"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "SGD" -no
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "Adam" -no

echo "Exp 8"
echo "Training and testing UNet"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "SGD" -no -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "SGD" -no -sch
echo "Training and testing UNeTr"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "SGD" -no -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "Adam" -no -sch

echo "Exp 1"
echo "Training and testing UNet"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "Adam"
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "Adam"
echo "Training and testing UNeTr"
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "Adam"
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "Adam"
