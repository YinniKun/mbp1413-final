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
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "Adam" -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "Adam" -sch
end=`date +%s`
runtime2_1=$((end-start))
echo "Training and testing UNeTr"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "Adam" -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "Adam" -sch
end=`date +%s`
runtime2_2=$((end-start))

echo "Exp 3"
echo "Training and testing UNet"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "Adam" -no
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "Adam" -no
end=`date +%s`
runtime3_1=$((end-start))
echo "Training and testing UNeTr"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "Adam" -no
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "Adam" -no
end=`date +%s`
runtime3_2=$((end-start))

echo "Exp 4"
echo "Training and testing UNet"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "Adam" -no -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "Adam" -no -sch
end=`date +%s`
runtime4_1=$((end-start))
echo "Training and testing UNeTr"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "Adam" -no -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "Adam" -no -sch
end=`date +%s`
runtime4_2=$((end-start))

echo "Exp 5"
echo "Training and testing UNet"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "SGD" 
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "SGD"
end=`date +%s`
runtime5_1=$((end-start))
echo "Training and testing UNeTr"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "SGD" 
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "SGD"
end=`date +%s`
runtime5_2=$((end-start))

echo "Exp 6"
echo "Training and testing UNet"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "SGD" -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "SGD" -sch
end=`date +%s`
runtime6_1=$((end-start))
echo "Training and testing UNeTr"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "SGD" -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "SGD" -sch
end=`date +%s`
runtime6_2=$((end-start))

echo "Exp 7"
echo "Training and testing UNet"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "SGD" -no
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "SGD" -no
end=`date +%s`
runtime7_1=$((end-start))
echo "Training and testing UNeTr"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "SGD" -no
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "SGD" -no
end=`date +%s`
runtime7_2=$((end-start))

echo "Exp 8"
echo "Training and testing UNet"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "SGD" -no -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "SGD" -no -sch
end=`date +%s`
runtime8_1=$((end-start))
echo "Training and testing UNeTr"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "SGD" -no -sch
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "SGD" -no -sch
end=`date +%s`
runtime8_2=$((end-start))

echo "Exp 1"
echo "Training and testing UNet"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unet" -opt "Adam"
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unet" -opt "Adam"
end=`date +%s`
runtime1_1=$((end-start))
echo "Training and testing UNeTr"
start=`date +%s`
python -u main.py -c config.yaml -e 200 -l 0.005 -mo "unetr" -opt "Adam"
python -u main.py -c config.yaml -m "test" -e 200 -l 0.005 -mo "unetr" -opt "Adam"
end=`date +%s`
runtime1_2=$((end-start))

echo "============Summary============"
echo "******UNet time in secs******"
echo $runtime1_1
echo $runtime2_1
echo $runtime3_1
echo $runtime4_1
echo $runtime5_1
echo $runtime6_1
echo $runtime7_1
echo $runtime8_1
echo "******UNetR time in secs******"
echo $runtime1_2
echo $runtime2_2
echo $runtime3_2
echo $runtime4_2
echo $runtime5_2
echo $runtime6_2
echo $runtime7_2
echo $runtime8_2
