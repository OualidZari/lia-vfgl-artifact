import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve
from utils.attack_utils import sample_pairs
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import wandb
import json

def perform_attack(args, gradients, data, attack_method, pos_samples, neg_samples, epoch):
    
    gradients = gradients.cpu()
    # if attack_method == 'cosine':
    #     pairwise_grad_metric = cosine_similarity(gradients)
    # if attack_method == 'correlation':
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
        run_id = wandb.run.id
        with open(f'scores_log/{run_id}_epoch_{epoch}_pos.json', 'w') as f:
            json.dump(pos_grad_sim.tolist(), f)
        with open(f'scores_log/{run_id}_epoch_{epoch}_neg.json', 'w') as f:
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

 
def calculate_auc(pos_scores, neg_scores, epoch):
    # Concatenate the positive and negative samples
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    
    #scores = (scores - scores.min() + 1e-16) / (scores.max() - scores.min() + 1e-16)

    # Compute the AUC
    auc = roc_auc_score(labels, scores)
    # if auc < 0.5:
    #     auc = 1 - auc
    fpr, tpr, thresholds = roc_curve(labels, scores)
    #wandb.log({"fpr": fpr, "tpr": tpr, "thresholds":thresholds}, step=epoch)
    wandb.log({"auc": auc}, step=epoch)

    return auc, fpr, tpr

def plot_roc_curve(fpr, tpr, auc, dataset_name, epoch):
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
    #plt.savefig(f'{dataset_name}_roc_curve.png')
    #wandb.log({"roc_curve": wandb.Image(plt)}, step=epoch)
    plt.close()
