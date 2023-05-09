#!/usr/bin/env bash

python token_classification.py --out_dir runs/flights_activeaed_0 --dataset inconsistencies_flights --scorer aum --n_epochs 40
python token_classification.py --out_dir runs/flights_activeaed_1 --dataset inconsistencies_flights --scorer aum --n_epochs 40 --seed 1
python token_classification.py --out_dir runs/flights_activeaed_2 --dataset inconsistencies_flights --scorer aum --n_epochs 40 --seed 2

python token_classification.py --out_dir runs/companies_activeaed_0 --dataset inconsistencies_companies --scorer aum --n_epochs 40
python token_classification.py --out_dir runs/companies_activeaed_1 --dataset inconsistencies_companies --scorer aum --n_epochs 40 --seed 1
python token_classification.py --out_dir runs/companies_activeaed_2 --dataset inconsistencies_companies --scorer aum --n_epochs 40 --seed 2

python token_classification.py --out_dir runs/forex_activeaed_0 --dataset inconsistencies_forex --scorer aum --n_epochs 40
python token_classification.py --out_dir runs/forex_activeaed_1 --dataset inconsistencies_forex --scorer aum --n_epochs 40 --seed 1
python token_classification.py --out_dir runs/forex_activeaed_2 --dataset inconsistencies_forex --scorer aum --n_epochs 40 --seed 2

python token_classification.py --out_dir runs/conll_activeaed_0 --dataset aed_conll --scorer aum --n_epochs 10
python token_classification.py --out_dir runs/conll_activeaed_1 --dataset aed_conll --scorer aum --n_epochs 10 --seed 1
python token_classification.py --out_dir runs/conll_activeaed_2 --dataset aed_conll --scorer aum --n_epochs 10 --seed 2

python token_classification.py --out_dir runs/gum_activeaed_0 --dataset aed_gum --scorer aum --n_epochs 10
python token_classification.py --out_dir runs/gum_activeaed_1 --dataset aed_gum --scorer aum --n_epochs 10 --seed 1
python token_classification.py --out_dir runs/gum_activeaed_2 --dataset aed_gum --scorer aum --n_epochs 10 --seed 2

python sequence_classification.py --out_dir runs/atis_activeaed_0 --dataset aed_atis --scorer aum --n_epochs 20
python sequence_classification.py --out_dir runs/atis_activeaed_1 --dataset aed_atis --scorer aum --n_epochs 20 --seed 1
python sequence_classification.py --out_dir runs/atis_activeaed_2 --dataset aed_atis --scorer aum --n_epochs 20 --seed 2

python sequence_classification.py --out_dir runs/imdb_activeaed_0 --dataset pervasive_imdb --scorer aum --n_epochs 5
python sequence_classification.py --out_dir runs/imdb_activeaed_1 --dataset pervasive_imdb --scorer aum --n_epochs 5 --seed 1
python sequence_classification.py --out_dir runs/imdb_activeaed_2 --dataset pervasive_imdb --scorer aum --n_epochs 5 --seed 2

python sequence_classification.py --out_dir runs/sst_activeaed_0 --dataset aed_sst --scorer aum --n_epochs 10
python sequence_classification.py --out_dir runs/sst_activeaed_1 --dataset aed_sst --scorer aum --n_epochs 10 --seed 1
python sequence_classification.py --out_dir runs/sst_activeaed_2 --dataset aed_sst --scorer aum --n_epochs 10 --seed 2


# Training-dynamic Methods (DM + AUM)

python token_classification_train.py --out_dir runs_camera/flights_train_0 --dataset inconsistencies_flights --scorer aum --n_epochs 40 --no_active
python token_classification_train.py --out_dir runs_camera/flights_train_1 --dataset inconsistencies_flights --scorer aum --n_epochs 40 --seed 1 --no_active
python token_classification_train.py --out_dir runs_camera/flights_train_2 --dataset inconsistencies_flights --scorer aum --n_epochs 40 --seed 2 --no_active

python token_classification_train.py --out_dir runs_camera/companies_train_0 --dataset inconsistencies_companies --scorer aum --n_epochs 40 --no_active
python token_classification_train.py --out_dir runs_camera/companies_train_1 --dataset inconsistencies_companies --scorer aum --n_epochs 40 --seed 1 --no_active
python token_classification_train.py --out_dir runs_camera/companies_train_2 --dataset inconsistencies_companies --scorer aum --n_epochs 40 --seed 2 --no_active

python token_classification_train.py --out_dir runs_camera/forex_train_0 --dataset inconsistencies_forex --scorer aum --n_epochs 40 --no_active
python token_classification_train.py --out_dir runs_camera/forex_train_1 --dataset inconsistencies_forex --scorer aum --n_epochs 40 --seed 1 --no_active
python token_classification_train.py --out_dir runs_camera/forex_train_2 --dataset inconsistencies_forex --scorer aum --n_epochs 40 --seed 2 --no_active

python token_classification_train.py --out_dir runs_camera/conll_train_0 --dataset aed_conll --scorer aum --n_epochs 10 --no_active
python token_classification_train.py --out_dir runs_camera/conll_train_1 --dataset aed_conll --scorer aum --n_epochs 10 --seed 1 --no_active
python token_classification_train.py --out_dir runs_camera/conll_train_2 --dataset aed_conll --scorer aum --n_epochs 10 --seed 2 --no_active

python token_classification_train.py --out_dir runs_camera/gum_train_0 --dataset aed_gum --scorer aum --n_epochs 10 --no_active
python token_classification_train.py --out_dir runs_camera/gum_train_1 --dataset aed_gum --scorer aum --n_epochs 10 --seed 1 --no_active
python token_classification_train.py --out_dir runs_camera/gum_train_2 --dataset aed_gum --scorer aum --n_epochs 10 --seed 2 --no_active


python sequence_classification_train.py --out_dir runs_camera/atis_train_0 --dataset aed_atis --scorer aum --n_epochs 20 --no_active
python sequence_classification_train.py --out_dir runs_camera/atis_train_1 --dataset aed_atis --scorer aum --n_epochs 20 --seed 1 --no_active
python sequence_classification_train.py --out_dir runs_camera/atis_train_2 --dataset aed_atis --scorer aum --n_epochs 20 --seed 2 --no_active

python sequence_classification_train.py --out_dir runs_camera/imdb_train_0 --dataset pervasive_imdb --scorer aum --n_epochs 5 --no_active
python sequence_classification_train.py --out_dir runs_camera/imdb_train_1 --dataset pervasive_imdb --scorer aum --n_epochs 5 --seed 1 --no_active
python sequence_classification_train.py --out_dir runs_camera/imdb_train_2 --dataset pervasive_imdb --scorer aum --n_epochs 5 --seed 2 --no_active

python sequence_classification_train.py --out_dir runs_camera/sst_train_0 --dataset aed_sst --scorer aum --n_epochs 10 --no_active
python sequence_classification_train.py --out_dir runs_camera/sst_train_1 --dataset aed_sst --scorer aum --n_epochs 10 --seed 1 --no_active
python sequence_classification_train.py --out_dir runs_camera/sst_train_2 --dataset aed_sst --scorer aum --n_epochs 10 --seed 2 --no_active


# CU
python token_classification.py --out_dir runs/flights_cu_0 --dataset inconsistencies_flights --scorer cu --n_epochs 40 --no_active --aggregation test
python token_classification.py --out_dir runs/flights_cu_1 --dataset inconsistencies_flights --scorer cu --n_epochs 40 --seed 1 --no_active --aggregation test
python token_classification.py --out_dir runs/flights_cu_2 --dataset inconsistencies_flights --scorer cu --n_epochs 40 --seed 2 --no_active --aggregation test

python token_classification.py --out_dir runs/companies_cu_0 --dataset inconsistencies_companies --scorer cu --n_epochs 40 --no_active --aggregation test
python token_classification.py --out_dir runs/companies_cu_1 --dataset inconsistencies_companies --scorer cu --n_epochs 40 --seed 1 --no_active --aggregation test
python token_classification.py --out_dir runs/companies_cu_2 --dataset inconsistencies_companies --scorer cu --n_epochs 40 --seed 2 --no_active --aggregation test

python token_classification.py --out_dir runs/forex_cu_0 --dataset inconsistencies_forex --scorer cu --n_epochs 40 --no_active --aggregation test
python token_classification.py --out_dir runs/forex_cu_1 --dataset inconsistencies_forex --scorer cu --n_epochs 40 --seed 1 --no_active --aggregation test
python token_classification.py --out_dir runs/forex_cu_2 --dataset inconsistencies_forex --scorer cu --n_epochs 40 --seed 2 --no_active --aggregation test

python token_classification.py --out_dir runs/conll_cu_0 --dataset aed_conll --scorer cu --n_epochs 10 --no_active --aggregation test
python token_classification.py --out_dir runs/conll_cu_1 --dataset aed_conll --scorer cu --n_epochs 10 --seed 1 --no_active --aggregation test
python token_classification.py --out_dir runs/conll_cu_2 --dataset aed_conll --scorer cu --n_epochs 10 --seed 2 --no_active --aggregation test

python token_classification.py --out_dir runs/gum_cu_0 --dataset aed_gum --scorer cu --n_epochs 10 --no_active --aggregation test
python token_classification.py --out_dir runs/gum_cu_1 --dataset aed_gum --scorer cu --n_epochs 10 --seed 1 --no_active --aggregation test
python token_classification.py --out_dir runs/gum_cu_2 --dataset aed_gum --scorer cu --n_epochs 10 --seed 2 --no_active --aggregation test

python sequence_classification.py --out_dir runs/atis_cu_0 --dataset aed_atis --scorer cu --n_epochs 20 --no_active --aggregation test
python sequence_classification.py --out_dir runs/atis_cu_1 --dataset aed_atis --scorer cu --n_epochs 20 --seed 1 --no_active --aggregation test
python sequence_classification.py --out_dir runs/atis_cu_2 --dataset aed_atis --scorer cu --n_epochs 20 --seed 2 --no_active --aggregation test

python sequence_classification.py --out_dir runs/imdb_cu_0 --dataset pervasive_imdb --scorer cu --n_epochs 5 --no_active --aggregation test
python sequence_classification.py --out_dir runs/imdb_cu_1 --dataset pervasive_imdb --scorer cu --n_epochs 5 --seed 1 --no_active --aggregation test
python sequence_classification.py --out_dir runs/imdb_cu_2 --dataset pervasive_imdb --scorer cu --n_epochs 5 --seed 2 --no_active --aggregation test

python sequence_classification.py --out_dir runs/sst_cu_0 --dataset aed_sst --scorer cu --n_epochs 10 --no_active --aggregation test
python sequence_classification.py --out_dir runs/sst_cu_1 --dataset aed_sst --scorer cu --n_epochs 10 --seed 1 --no_active --aggregation test
python sequence_classification.py --out_dir runs/sst_cu_2 --dataset aed_sst --scorer cu --n_epochs 10 --seed 2 --no_active --aggregation test
