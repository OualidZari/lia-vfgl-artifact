import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve
from utils.attack_utils import sample_pairs
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import wandb
import json
import os
import csv
from datetime import datetime

def perform_attack(args, gradients, data, attack_method, pos_samples, neg_samples, epoch):
    
    gradients = gradients.cpu()
    if args.attack_method == 'cosine':
        pairwise_grad_metric = cosine_similarity(gradients)
    else:
        pairwise_grad_metric = pdist(gradients, metric=args.attack_method)
        pairwise_grad_metric = squareform(pairwise_grad_metric)
    pos_samples = np.array(pos_samples)
    neg_samples = np.array(neg_samples)
    
    pos_grad_sim = pairwise_grad_metric[pos_samples[:, 0], pos_samples[:, 1]]
    neg_grad_sim = pairwise_grad_metric[neg_samples[:, 0], neg_samples[:, 1]]
    
    if args.store_scores:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") if args.local_logging else wandb.run.id
        log_dir = f"logs/{args.experiment_name}/{run_id}"
        os.makedirs(log_dir, exist_ok=True)
        
        with open(f'{log_dir}/epoch_{epoch}_pos.json', 'w') as f:
            json.dump(pos_grad_sim.tolist(), f)
        with open(f'{log_dir}/epoch_{epoch}_neg.json', 'w') as f:
            json.dump(neg_grad_sim.tolist(), f)
            
    return pos_grad_sim, neg_grad_sim

def perfrom_baseline_attack(args, inputs, pos_samples, neg_samples):
    pos_samples = np.array(pos_samples)
    neg_samples = np.array(neg_samples)
    inputs = inputs.cpu().detach().numpy()
    if args.attack_method == "baseline_normalization":
        norms = np.linalg.norm(inputs, axis=1, keepdims=True)
        V = np.where(norms != 0, inputs / norms, inputs)
        Q = 1 - np.matmul(V, V.T)
    else :
        Q = pdist(inputs, metric=args.attack_method)
        Q = squareform(Q)
    pos_sim = Q[pos_samples[:, 0], pos_samples[:, 1]]
    neg_sim = Q[neg_samples[:, 0], neg_samples[:, 1]]
    
    return pos_sim, neg_sim    

def calculate_auc(args, pos_scores, neg_scores, epoch):
    # Concatenate the positive and negative samples
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])

    # Compute the AUC
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Local logging
    if args.local_logging:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"logs/{args.experiment_name}/{run_id}"
        os.makedirs(log_dir, exist_ok=True)
        
        with open(f'{log_dir}/auc_results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # Write header if file is empty
                writer.writerow(['epoch', 'metric', 'value'])
            writer.writerow([epoch, 'auc', auc])

    # Wandb logging (if enabled)
    if args.use_wandb:
        wandb.log({"auc": auc}, step=epoch)

    return auc, fpr, tpr

def plot_roc_curve(args, fpr, tpr, auc, dataset_name, epoch):
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name}')
    plt.legend(loc="lower right")

    # Local logging
    if args.local_logging:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"logs/{args.experiment_name}/{run_id}"
        os.makedirs(log_dir, exist_ok=True)
        plt.savefig(f'{log_dir}/{dataset_name}_roc_curve_epoch_{epoch}.png')

    # Wandb logging (if enabled)
    if args.use_wandb:
        wandb.log({"roc_curve": wandb.Image(plt)}, step=epoch)

    plt.close()