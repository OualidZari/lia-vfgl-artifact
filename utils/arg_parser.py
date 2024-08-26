import argparse
import ast

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Citeseer', 
                        help="Dataset to use.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="Device to use for computation ('cpu' or 'cuda').")
    parser.add_argument("--seed", type=int, default=123, 
                        help="Random seed for reproducibility.")
    parser.add_argument("--lr", type=float, default=0.01, 
                        help="Learning rate for optimizers.")
    parser.add_argument("--epochs", type=int, default=300, 
                        help="Number of training epochs.")
    parser.add_argument("--train_ratio", type=float, default=0.5, 
                        help="Fraction of data to use for training.")
    parser.add_argument("--val_ratio", type=float, default=0.1, 
                        help="Fraction of data to use for validation.")
    parser.add_argument("--fraction_data_gcn", type=float, default=0.1, 
                        help="Fraction of features for the GCN model.")
    parser.add_argument("--perform_attack", action='store_true', 
                    help="If specified, perform the attack. Default is False (attack is not performed).")
    parser.add_argument("--attack_method", type=str, default='cosine',
                        help='Attack method')
    parser.add_argument("--num_pos_samples", type=int, default=None, 
                        help="Number of positive samples to evaluate the attack")
    parser.add_argument("--num_neg_samples", type=int, default=None, 
                        help="Number of negative samples to evaluate the attack")
    parser.add_argument("--perform_attack_baseline", action='store_true', 
                    help="If specified, perform the attack. Default is False (attack is not performed).")
    parser.add_argument("--perform_attack_all_methods", action='store_true', 
                    help="If specified, perform all the attack methods. Default is False (attack is not performed).")
    parser.add_argument("--attack_all_nodes", action='store_true', 
                    help="If specified, attack all the nodes in the graph. Default is False (attack is not performed).")
    parser.add_argument("--experiment_comment", type=str, 
                    help="Special experiment comment.")
    parser.add_argument("--store_scores", action='store_true', 
                    help="If specified, it saves the scores.")
    parser.add_argument("--use_wandb", action='store_true', 
                    help="If specified, wanbd is used.")
    parser.add_argument("--store_gradients", action='store_true', 
                    help="If specified, gradients are stored")
    parser.add_argument("--store_forward_mlp", action='store_true',
                        help="If specified, forward pass of MLP is stored")
    parser.add_argument("--store_output_server", action='store_true',
                        help="If specified, output of the server is saved")
    parser.add_argument("--active_attack", action='store_true',
                        help="If specified, active adv alters the mlp output")
    parser.add_argument('--attack_epochs', type=ast.literal_eval, default="[]",
                        help="List of epochs to perform the attack (default: [])")
    parser.add_argument('--attack_methods', default='gradients,labels', type=str, help='Comma-separated list of attack methods to evaluate')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--output_size', type=int, default=16)
    parser.add_argument('--architecture', type=str, default='gcn')
    parser.add_argument('--n_attacked_nodes', type=int, default=500)
    parser.add_argument('--sampling_strategy', type=str, default='all_nodes')
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--multi_party', action='store_true', help='Enable multi-party setup')
    parser.add_argument('--n_parties', type=int, default=2, help='Number of parties in the multi-party setup')
    parser.add_argument('--lapgraph', action='store_true', help='Use Lapgraph')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Epsilon for Lapgraph')
    parser.add_argument('--espilon_factor', type=float, default=0.01, help='Epsilon factor for Lapgraph')
    parser.add_argument('--label_defense', action='store_true', help='Enable label defense')
    parser.add_argument('--label_defense_budget', type=float, default=0.1, help='Label Defense budget')
    parser.add_argument('--gradient_defense', action='store_true', help='Enable gradient defense')
    parser.add_argument('--gradient_defense_noise_level', type=float, default=0.1, help='Gradient Defense noise level')
    parser.add_argument('--enable_clipping', action='store_true', help='Enable clipping')
    parser.add_argument('--clip_grad_norm', type=float, default=0.1, help='Clip value')
    parser.add_argument('--similarity_metric', type=str, default=None, help='Similarity metric')
    
    parser.add_argument('--use_resnet', action='store_true', help='Use ResNet-like architecture')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks')
    parser.add_argument('--hidden_size', type=int, default=64, help='Size of hidden layers')
    parser.add_argument('--local_logging', action='store_true', help='Enable local logging')
    parser.add_argument('--experiment_name', type=str, default='default_experiment', help='Name of the experiment for logging')
    return parser.parse_args()
