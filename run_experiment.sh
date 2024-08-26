#!/bin/bash

function run_experiment() {

    seed=$1
    fraction_data_gcn=$2
    dataset=$3
    architecture=$4


    python main.py --dataset ${dataset} --device cuda --gpu 3 --seed ${seed} --lr 0.01 --epochs 10 --val_ratio 0.02 \
        --train_ratio 0.5 \
        --fraction_data_gcn ${fraction_data_gcn} \
        --architecture ${architecture} \
        --perform_attack_all_methods --attack_all_nodes \
        --attack_methods 'labels,features,gradients,output_server,forward_values' \
        --local_logging \
        --experiment_name "Table 4 results Cora multiple seeds" \
        --attack_epochs [-1] \
        # --use_wandb
        # --gradient_defense \
        # --gradient_defense_noise_level ${gradient_defense_noise_level} \
        
        
        

    # --label_defense \
    # --label_defense_budget ${label_defense_budget}

    #--save_predictions
    # --store_gradients \
    # --store_forward_mlp \
    # --store_output_server \
    # --num_neg_samples 1000 \
    # --num_pos_samples 1000 \
    #--attack_methods 'gradients,labels,features,output_server,forward_values'
    #--attack_method ${attack_method} \

}
export -f run_experiment
#seeds=(42 12 36)
# seeds=(42 12 36 15 11 99 04 09 98 10)
seeds=(42 12 36)
fraction_data_gcn=(0.5)
#fraction_data_gcn=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
# architectures=('gcn' 'gat' 'sage')
architectures=('gcn')
#datasets=('Citeseer')
datasets=('Cora')
# datasets=('Cora' 'amazon_photo' Twitch-DE' 'Twitch-EN' 'Twitch-ES' 'Twitch-FR')
# epsilons=(-1 1 2 3 4 5 6 7 8 9 10)
# label_defense_budget=(0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# gradient_defense_noise_levels=(0.00001 0.00002 0.00003 0.00004 0.00005)
# gradient_defense_noise_levels=(0.00001 0.0001 0.001 0.01 0.1 1 10)
# num_res_blocks=(1 2 3 4)
# clip_grad_norms=(0.01 0.1 0.5 1.0 5 10)
# n_parties=(2)
# train_ratios=(0.5)
# similarity_metric=('cosine' 'euclidean' 'correlation' 'chebyshev')
#train_ratios=(0.5) #default
#datasets=('Cora')
parallel -j 1 run_experiment ::: "${seeds[@]}" ::: "${fraction_data_gcn[@]}" ::: "${datasets[@]}" ::: "${architectures[@]}"
