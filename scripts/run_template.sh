#!/bin/bash
#PBS -N flower_predict
#PBS -q inferno
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -A GT-saluru8-CODA20
#PBS -j oe
#PBS -o flower_predict.log
#PBS -e flower_predict.log

date

module load anaconda3
cd /storage/coda1/p-saluru8/ideas0/schockalingam6/arrays_class/union/ml_integ/ensembleGRN/scripts/
pwd
python ensemble_arabidopsis_ML.py edge_networks/flower-iqr-ids-edges.csv ensemble_preds1/flower-iqr-preds.tsv

date

