# QSPN_code

## Introduction

***QSPN_code*** is the code of paper: A Unified Model for Cardinality Estimation by Learning from Data and Queries via Sum-Product Networks.

## Environment

**Step 1:** Download this code.

    git clone https://github.com/rucjrliu/QSPN_code.git

**Step 2:** Install dependencies of the environment qspn.

    cd QSPN_code/
    conda install -f environment.yml

**Step 3:** Downloading our experimental data.

Download file **qspn_data_models.tar** from OneDrive sharing link: https://1drv.ms/u/c/f9c0a1a8c6911768/EfoPGQrHElVKmxYIaGbkueIB1B7XUeAvMK2Ns6PAaCrpYw?e=Q5VOF4

    unzip qspn_data_models.tar
    mv qspn_data_models/* .

In the following, all running should be at this directory 'QSPN_code/'

## QSPN Single-Table CardEst

We first show the command of conducting the experiments in our paper.

Dataset: GAS

    python scripts/run_qspn.py --dataset gas --inference

Dataset: Census

    python scripts/run_qspn.py --dataset census13 --inference

Dataset: Forest

    python scripts/run_qspn.py --dataset forest10 --inference

Dataset: Power

    python scripts/run_qspn.py --dataset power7 --inference

You can also construct *QSPN* models by yourself, like

    python scripts/run_qspn.py --dataset gas --train
    python scripts/run_qspn.py --dataset census13 --train
    python scripts/run_qspn.py --dataset forest10 --train
    python scripts/run_qspn.py --dataset power7 --train

Actually, the whole command of running or constructing *QSPN* on dataset Forest and our workload is:

    python scripts/run_qspn.py --dataset forest10 --query-root template --skew 0.5 --corr 0.5 --inference
    python scripts/run_qspn.py --dataset forest10 --query-root template --skew 0.5 --corr 0.5 --train

which allows users to use their own dataset and workload by change the value of parameter '--dataset', '--query-root', '--skew', '--corr'

## QSPN Update CardEst

Take dataset Power for instance, we show how to run three update methods: *NoTrain*, *ReTrain* and our *AdaIncr*.

    python scripts/run_qspn.py --dataset power7 --skew 0.0 --corr 0.0 --update-method notrain --update-query-root update --update-skew 0.5 --update-corr 0.5
    python scripts/run_qspn.py --dataset power7 --skew 0.0 --corr 0.0 --update-method retrain --update-query-root update --update-skew 0.5 --update-corr 0.5
    python scripts/run_qspn.py --dataset power7 --skew 0.0 --corr 0.0 --update-method adaincr --update-query-root update --update-skew 0.5 --update-corr 0.5

where '--skew 0.0' and '--corr 0.0' means using the model constructed on workload with no access on correlated columns; '--update-method notrain/retrain/adaincr' means testing different update method; '--update-query-root update --update-skew 0.5 --update-corr 0.5' means testing the model on Hybrid-Update workload ('--update-query-root update': the workload contains data-update; '--update-skew 0.5 --update-corr 0.5': the workload contains query-update).

If you want to test update method *AdaIncr* on Data-Update-Only workload, the command is

    python scripts/run_qspn.py --dataset power7 --skew 0.0 --corr 0.0 --update-method notrain --update-query-root update --update-skew 0.0 --update-corr 0.0

If you want to test update method *AdaIncr* on Query-Update-Only workload, the command is

    python scripts/run_qspn.py --dataset power7 --skew 0.0 --corr 0.0 --update-method notrain --update-query-root template --update-skew 0.5 --update-corr 0.5

## M-QSPN Multi-Table CardEst

Dataset: IMDB, Workload: JOB-light

    python scripts/run_mqspn.py --dataset job --workload-trainset mscn_queries_neurocard_format --workload-testset job-light --inference

If you want construction a *M-QSPN* model by yourself, such as a model with binning size as $1000$, the command is

    python scripts/run_mqspn.py --dataset job --workload-trainset mscn_queries_neurocard_format --workload-testset job-light --model-binning-size 1000 --train

Also, if you want to test this model (with not default binning size: $1000$), the command is

    python scripts/run_mqspn.py --dataset job --workload-trainset mscn_queries_neurocard_format --workload-testset job-light --model-binning-size 1000 --inference