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
echo "Exp 1"
echo "Training and testing UNet"
start=`date +%s`
python -u main.py -c config.yaml -m train -d -e 200 -l 0.005 -mo unet -sch -opt Adam -sa
python -u main.py -c config.yaml -m test -e 200 -l 0.005 -mo unet -sch -opt Adam -sa
end=`date +%s`
runtime1_1=$((end-start))
echo "Training and testing UNeTr"
start=`date +%s`
python -u main.py -c config.yaml -m train -d -e 200 -l 0.005 -mo unetr -sch -opt Adam -sa
python -u main.py -c config.yaml -m test -e 200 -l 0.005 -mo unetr -sch -opt Adam -sa
end=`date +%s`
runtime1_2=$((end-start))

echo "============Summary============"
echo "******UNet time in secs******"
echo $runtime1_1
echo "******UNetR time in secs******"
echo $runtime1_2
