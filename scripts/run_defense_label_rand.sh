#!/bin/bash
function get_gpu_with_most_free_memory() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print NR-1 ":" $1}' | sort -rn -t: -k2 | head -n1 | cut -d: -f1
}

function run_experiment() {

    seed=$1
    fraction_data_gcn=$2
    dataset=$3
    architecture=$4
    label_defense_budget=$5

    # Get the GPU with the most free memory
    gpu=$(get_gpu_with_most_free_memory)
    
    # Map dataset names for experiment name
    if [ "$dataset" == "amazon_photo" ]; then
        experiment_dataset="Photo"
    elif [ "$dataset" == "amazon_computer" ]; then
        experiment_dataset="Computer"
    else
        experiment_dataset="${dataset}"
    fi
    # Change to the parent directory before running the Python script
    (cd .. && python main.py --dataset ${dataset} --device cuda --gpu ${gpu} --seed ${seed} --lr 0.01 --epochs 300 --val_ratio 0.02 \
        --train_ratio 0.5 \
        --fraction_data_gcn ${fraction_data_gcn} \
        --architecture ${architecture} \
        --perform_attack_all_methods --attack_all_nodes \
        --attack_methods 'labels,gradients,output_server,forward_values' \
        --local_logging \
        --experiment_name "Defense Label Randomization results ${experiment_dataset}" \
        --attack_epochs [-1] \
        --label_defense \
        --label_defense_budget ${label_defense_budget}
    )
}

export -f run_experiment
export -f get_gpu_with_most_free_memory
seeds=(42 12 36 15 11 99 04 09 98 10)
# seeds=(36)
fraction_data_gcn=(0.5)
architectures=('gcn')
datasets=('amazon_computer' 'amazon_photo' 'Cora' 'Citeseer')
label_defense_budgets=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# label_defense_budgets=(0.1 0.2)
num_runs=5
parallel --line-buffer -j ${num_runs} run_experiment ::: "${seeds[@]}" ::: "${fraction_data_gcn[@]}" ::: "${datasets[@]}" ::: "${architectures[@]}" ::: "${label_defense_budgets[@]}"