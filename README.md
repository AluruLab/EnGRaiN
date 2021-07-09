# EnGRaiN
----

EnGRaiN is the first supervised machine learning method to construct ensemble networks. 
To benefit from the typical accuracy advantages of supervised learning methods while 
taking into account the impossibility of knowing true networks for training, 
we devised a method that uses small training datasets of true positives and true negatives among gene pairs. 


Dependencies
----
EnGRaiN requires python v 3.1 or above and depends upon the following python 
libraries:

  - pandas 
  - json
  - numpy 
  - matplotlib 
  - xgboost 
  - sklearn

The libraries can be installed via `pip` or `conda`.

EnGRaiN script
----
The source code for EnGRaiN is made available as `engrain_ensemble.py` python script
in the src/ folder of this reporsitory. This script has the following usage.

    python src/engrain_ensemble.py -h 

    usage: engrain_ensemble.py [-h]
              {sim,ravgu,ravg,xtissue,pred_stk,pred_ens,grids_rocpr,grids_tpfp}
              ...

    Train with a subset of network and predict Ensemble

    positional arguments:
      {sim,ravgu,ravg,xtissue,pred_stk,pred_ens,grids_rocpr,grids_tpfp}
      sim                 Run Simulated Datasets
      ravgu               Run Rank Avg.&ScaleSum of union networks for A.
                          thaliana Datasets
      ravg                Run Rank Avg.&ScaleSum of networks for A. thaliana
                          Datasets
      xtissue             Run Cross-tissue comparisions A. thaliana Datasets
      pred_stk            Stacked Predictions for A. thaliana Networks
      pred_ens            Ensemble Predictions for A. thaliana Networks
      grids_rocpr         Run Grid Search with XGBoost params for A. thaliana
                          Datasets
      grids_tpfp          Run Grid Search with XGBoost params for A. thaliana
                          Datasets

    optional arguments:
      -h, --help            show this help message and exit


Simulated Dataset Runs
---

In our paper, we demonstrate the effectiveness of EnGRaiN using simulated datasets.
The smaller networks from simulated datsets are made available in the data/ sub-directory in this reporsitory.
The larger networks from simulated datasets that are used in the paper can be downloaded via the links provided in [data/README.md](/data/README.md) file.


The EnGRaiN script requires a JSON input file with the required input configurations.
The input configurations for these runs are present in results/config folder. 


To run the latest version of the simulated analysis:

1. `cd` to the results folder.
2. Link the data direcotry : `ln -s ../data/`
3. Download the [yeast-edge-weights-v5.csv.gz](https://www.dropbox.com/s/c7rhjs75oek1wia/yeast-edge-weights-v5.csv.gz?dl=0)  dataset to the `data` folder.
4. Run the command `python engrain_ensemble.py sim v5`.
5. AUROC/AUPR Output will generated as a table in the standard output.

A. thaliana Dataset Runs
---
To evaluate EnGRaiN, we also used a curated collection of \textit{A. thaliana} 
datasets, that we created from microarray datasets available from 
public repositories.

Tissue-specific Network data used for evaluation are available at the data/athaliana_raw directory. 
Note that this includes only the scores for positives and negatives.
AUROC/AUPR can be computed using this data with the help of the input config files 
in the runs/ens_grid_search directory.

The final Arabidopsis Ensemble network constructed using EnGRain can be downloaded from [here](https://www.dropbox.com/s/8fu4i5q8ynmpxu6/EnGRaiN-Athaliana-Ensemble.zip?dl=0).


Runs on Simulated Data  
----
The source code for the containers and aggregation scripts for simulated runs are in the github repo [srirampc/ardmore](https://github.com/srirampc/ardmore).

Microarray Data Processing  
----
The scripts for Microarray data processing workflow are available in the github repo [srirampc/tanyard](https://github.com/srirampc/tanyard).

