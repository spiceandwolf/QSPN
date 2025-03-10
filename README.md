# QSPN_code

***QSPN_code*** is the code of paper: A Unified Model for Cardinality Estimation by Learning from Data and Queries via Sum-Product Networks...

## Configuration

**Step 1:** Download this code.

    git clone https://github.com/rucjrliu/QSPN_code.git

**Step 2:** Install dependencies of the environment qspn.

    cd QSPN_code/
    conda install -f environment.yml

**In the following, all running should be at this directory 'QSPN_code/'.**

**Step 3:** Downloading our experimental data.

Download file **qspn_data_models.tar** from OneDrive sharing link: https://1drv.ms/u/c/f9c0a1a8c6911768/EfoPGQrHElVKmxYIaGbkueIB1B7XUeAvMK2Ns6PAaCrpYw?e=Q5VOF4

    unzip qspn_data_models.tar
    mv qspn_data_models/* .

## Quick Start

### QSPN Single-Table CardEst

We first show the commands of conducting the experiments in our paper.

Dataset: GAS/Census/Forest/Power

    python scripts/run_qspn.py --dataset gas --inference
    python scripts/run_qspn.py --dataset census13 --inference
    python scripts/run_qspn.py --dataset forest10 --inference
    python scripts/run_qspn.py --dataset power7 --inference

The result will be outputed to the console, including Model Size, Mean Inference Time, Q-error of each query, 50th/90th/95th/99th/Max/Mean Q-error of CardEst on the workload.

You can also construct *QSPN* models by yourself, like

    python scripts/run_qspn.py --dataset gas --train
    python scripts/run_qspn.py --dataset census13 --train
    python scripts/run_qspn.py --dataset forest10 --train
    python scripts/run_qspn.py --dataset power7 --train

The Model Construction Time and Model Size will be outputed to the console. And the *QSPN* model file: *'models/single_tables/qspn/qspn_5.0_0.1_0.3_QSplit_\{gas/census13/forest10/power7\}_template_0.5_0.5.pkl'* will be overwritten by the new constructed *QSPN* model.

### QSPN Update CardEst

Take dataset Power for instance, we show how to run three update methods: *NoTrain*, *ReTrain* and our *AdaIncr*.

    python scripts/run_qspn.py --dataset power7 --skew 0.0 --corr 0.0 --update-method notrain --update-query-root update --update-skew 0.5 --update-corr 0.5
    python scripts/run_qspn.py --dataset power7 --skew 0.0 --corr 0.0 --update-method retrain --update-query-root update --update-skew 0.5 --update-corr 0.5
    python scripts/run_qspn.py --dataset power7 --skew 0.0 --corr 0.0 --update-method adaincr --update-query-root update --update-skew 0.5 --update-corr 0.5

where '--skew 0.0' and '--corr 0.0' means using the model constructed on the original workload with no access on correlated columns; '--update-method notrain/retrain/adaincr' means testing different update method *(NoTrain/ReTrain/AdaIncr)*; '--update-query-root update --update-skew 0.5 --update-corr 0.5' means testing the model on Hybrid-Update workload ('--update-query-root update': the workload contains data-update; '--update-skew 0.5 --update-corr 0.5': the workload contains query-update).

If you want to test update method *AdaIncr* on Data-Update-Only workload, the command is

    python scripts/run_qspn.py --dataset power7 --skew 0.0 --corr 0.0 --update-method notrain --update-query-root update --update-skew 0.0 --update-corr 0.0

If you want to test update method *AdaIncr* on Query-Update-Only workload, the command is

    python scripts/run_qspn.py --dataset power7 --skew 0.0 --corr 0.0 --update-method notrain --update-query-root template --update-skew 0.5 --update-corr 0.5

You can also change the dataset by change parameter '--dataset power7' to '--dataset gas/census13/forest10' and so on.

The result will be outputed to file *'qspn_update_expr.log'*, including workload executing time and Mean Q-error on each interval of the whole update workload (we set $10$ checkpoints).

### M-QSPN Multi-Table CardEst

Dataset: IMDB, Workload: JOB-light

    python scripts/run_mqspn.py --dataset job --workload-trainset mscn_queries_neurocard_format --workload-testset job-light --inference

The result will be outputed to the console, including Model Size, Mean Inference Time, Q-error of each query, 50th/90th/95th/99th/Max/Mean Q-error of CardEst on the workload.

If you want construction a *M-QSPN* model by yourself, such as a model with binning size as $1000$, the command is

    python scripts/run_mqspn.py --dataset job --workload-trainset mscn_queries_neurocard_format --workload-testset job-light --model-binning-size 1000 --train

Also, if you want to test this model (with not default binning size: $1000$), the command is

    python scripts/run_mqspn.py --dataset job --workload-trainset mscn_queries_neurocard_format --workload-testset job-light --model-binning-size 1000 --inference

## Code Structure

```sh
|-- data # folder of datasets for Entity Resolution
|-- models # folder of constructed QSPN & M-QSPN models
|-- qspn # folder of QSPN & M-QSPN source code
|-- scripts # folder of scripts to run CardEst
    |-- run_qspn.py # Single-Table CardEst & Model Update by QSPN model
    |-- run_mqspn.py # Multi-Table CardEst by M-QSPN model
    |-- update_eval.py # (Please Do NOT Run) Source code for model update method, called by run_qspn.py
|-- settings.py # (Please Do NOT Modify) Configuration of Paths
```

<!-- ## Advanced Setting

### Single-Table Dataset

File format...

### Single-Table Workload

File format...

### Multi-Table Dataset

File format...

### Multi-Table Workload

File format...

### Running Parameter

#### QSPN Paramter

...

#### M-QSPN Parater

Actually, the whole command of running or constructing *QSPN* on dataset Forest and our workload is:

    python scripts/run_qspn.py --dataset forest10 --query-root template --skew 0.5 --corr 0.5 --inference
    python scripts/run_qspn.py --dataset forest10 --query-root template --skew 0.5 --corr 0.5 --train

which allows users to use their own dataset and workload by change the value of parameter '--dataset', '--query-root', '--skew', '--corr'

## Acknowledgement

... -->