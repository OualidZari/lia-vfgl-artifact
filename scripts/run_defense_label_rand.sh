#!/bin/bash

function run_experiment() {

    seed=$1
    fraction_data_gcn=$2
    dataset=$3
    architecture=$4
    label_defense_budget=$5
    # Change to the parent directory before running the Python script
    (cd .. && python main.py --dataset ${dataset} --device cuda --gpu 1 --seed ${seed} --lr 0.01 --epochs 300 --val_ratio 0.02 \
        --train_ratio 0.5 \
        --fraction_data_gcn ${fraction_data_gcn} \
        --architecture ${architecture} \
        --perform_attack_all_methods --attack_all_nodes \
        --attack_methods 'labels,gradients,output_server,forward_values' \
        --local_logging \
        --experiment_name "Defense Label Randomization results Cora" \
        --attack_epochs [-1] \
        --label_defense \
        --label_defense_budget ${label_defense_budget}
    )
}

export -f run_experiment
seeds=(42 12 36 15 11 99 04 09 98 10)
fraction_data_gcn=(0.5)
architectures=('gcn')
datasets=('Citeseer')
label_defense_budgets=(0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
num_runs=10
parallel --line-buffer -j ${num_runs} run_experiment ::: "${seeds[@]}" ::: "${fraction_data_gcn[@]}" ::: "${datasets[@]}" ::: "${architectures[@]}" ::: "${label_defense_budgets[@]}"