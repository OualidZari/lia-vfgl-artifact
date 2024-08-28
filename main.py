import random
from defense.label_defense import labels_defense
from defense.lapgraph import LapGraph
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from models import MLPModel, ActiveModel, GCN, GAT, SAGE
from utils import get_args, load_data, set_seed
from utils.data_loader import (
    get_training_graph,
    process_attack_data_online,
    get_attacked_nodes_data,
)
from attack import *
from tqdm import tqdm
from label_based_attack import compute_performance_metrics
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import json
import pickle
import os
import csv
from datetime import datetime


def main():
    args = get_args()
    set_seed(args.seed)
    if -1 in args.attack_epochs:
        args.attack_epochs = list(range(1, args.epochs + 1))

    device = torch.device(
        f"{args.device}:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    dataset, data, train_mask, val_mask, test_mask = load_data(args)
    _, data_utility, _, _, _ = load_data(args)
    if args.label_defense:
        new_labels = labels_defense(data.y, args.label_defense_budget)
        data.y = new_labels

    # add here lapgraph
    if args.lapgraph and args.epsilon != -1:
        adj = to_dense_adj(data.edge_index)[0]
        perturbed_adj = LapGraph(
            adj, epsilon=args.epsilon, device=device, eps_factor=args.espilon_factor
        )
        pertubed_edge_index = dense_to_sparse(perturbed_adj)[0]
        data.edge_index = pertubed_edge_index

    data = data.to(device)
    data_utility = data_utility.to(device)

    if args.train_ratio == 1.0:
        test_mask = train_mask

    # Setup logging
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_seed_{args.seed}_" + str(random.randint(0, 100))
    log_dir = f"logs/{args.experiment_name}/{run_id}"
    os.makedirs(log_dir, exist_ok=True)

    # Save config
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # Initialize CSV files for logging
    csv_files = {
        "metrics": f"{log_dir}/metrics.csv",
        "attack_results": f"{log_dir}/attack_results.csv"
    }
    for file in csv_files.values():
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "metric", "value"])

    if args.use_wandb:
        wandb.init(
            project="Link Inference In FL",
            config=vars(args),
            mode="offline"
        )
    else:
        wandb.init(project="Link Inference In FL", config=vars(args), mode="disabled")

    if args.multi_party:
        print("Multi-party setup")
        num_features_mlp_adv = int(data.num_features * 0.2)
        num_features_mlp_benign = int(data.num_features * 0.2)
        num_features_gcn = data.num_features - num_features_mlp_adv

        if args.n_parties > 2:
            num_features_gcn -= num_features_mlp_benign * (args.n_parties - 2)

        data_splits = [num_features_gcn, num_features_mlp_adv]
        if args.n_parties > 2:
            data_splits += [num_features_mlp_benign] * (args.n_parties - 2)

        features_split = torch.split(data.x, data_splits, dim=1)
        features_gcn = features_split[0].to(device)
        features_mlp_adv = features_split[1].to(device)
        features_mlp_benign_list = (
            [features_split[i + 2].to(device) for i in range(args.n_parties - 2)]
            if args.n_parties > 2
            else []
        )

        if args.architecture == "gcn":
            gnn_model = GCN(
                args, input_size=num_features_gcn, output_size=args.output_size
            ).to(device)
        elif args.architecture == "gat":
            gnn_model = GAT(
                args, input_size=num_features_gcn, output_size=args.output_size
            ).to(device)
        elif args.architecture == "sage":
            gnn_model = SAGE(
                args, input_size=num_features_gcn, output_size=args.output_size
            ).to(device)

        mlp_adv_model = MLPModel(
            args,
            input_size=num_features_mlp_adv,
            hidden_size=num_features_mlp_adv // 2,
            output_size=args.output_size,
        ).to(device)
        mlp_benign_models = [
            MLPModel(
                args,
                input_size=num_features_mlp_benign,
                hidden_size=num_features_mlp_benign // 2,
                output_size=args.output_size,
            ).to(device)
            for _ in range(args.n_parties - 2)
        ]

        input_size_active_model = (
            int(args.output_size * args.n_parties)
            if args.n_parties > 2
            else args.output_size * 2
        )
        active_model = ActiveModel(
            args, input_size=input_size_active_model, output_size=dataset.num_classes
        ).to(device)

        gnn_optimizer = optim.Adam(
            gnn_model.parameters(), lr=args.lr, weight_decay=1e-3
        )
        mlp_adv_optimizer = optim.Adam(
            mlp_adv_model.parameters(), lr=args.lr, weight_decay=1e-3
        )
        mlp_benign_optimizers = [
            optim.Adam(mlp_model.parameters(), lr=args.lr, weight_decay=1e-3)
            for mlp_model in mlp_benign_models
        ]
        active_optimizer = optim.Adam(
            active_model.parameters(), lr=args.lr, weight_decay=1e-3
        )

    else:
        print("Two-party setup")
        num_features_gcn = int(data.num_features * args.fraction_data_gcn)
        num_features_mlp = data.num_features - num_features_gcn
        features_gcn, features_mlp = torch.split(
            data.x, [num_features_gcn, num_features_mlp], dim=1
        )
        features_gcn = features_gcn.to(device)
        features_mlp = features_mlp.to(device)

        if args.architecture == "gcn":
            gnn_model = GCN(
                args, input_size=num_features_gcn, output_size=args.output_size
            ).to(device)
        elif args.architecture == "gat":
            gnn_model = GAT(
                args, input_size=num_features_gcn, output_size=args.output_size
            ).to(device)
        elif args.architecture == "sage":
            gnn_model = SAGE(
                args, input_size=num_features_gcn, output_size=args.output_size
            ).to(device)

        mlp_model = MLPModel(
            args,
            input_size=num_features_mlp,
            hidden_size=num_features_mlp // 2,
            output_size=args.output_size,
        ).to(device)
        active_model = ActiveModel(
            args, input_size=args.output_size * 2, output_size=dataset.num_classes
        ).to(device)

        gnn_optimizer = optim.Adam(
            gnn_model.parameters(), lr=args.lr, weight_decay=1e-3
        )
        mlp_optimizer = optim.Adam(
            mlp_model.parameters(), lr=args.lr, weight_decay=1e-3
        )
        active_optimizer = optim.Adam(
            active_model.parameters(), lr=args.lr, weight_decay=1e-3
        )

    if args.use_wandb:
        wandb.define_metric("auc", summary="max")

    if args.store_forward_mlp:
        mlp_forward_list = []
    if args.store_output_server:
        output_server = []

    if args.perform_attack_all_methods:
        args.attack_methods = args.attack_methods.split(",")
        data_train, node_index_mapping = get_training_graph(train_mask, args)
        data_attack, node_index_mapping_attack = get_attacked_nodes_data(
            features_mlp_adv if args.multi_party else features_mlp, train_mask, args
        )

    labels_attack_done, features_attack_done = False, False
    for epoch in tqdm(range(args.epochs)):
        if args.multi_party:
            gnn_output = gnn_model(features_gcn, data.edge_index)
            mlp_adv_output = mlp_adv_model(features_mlp_adv)
            mlp_benign_outputs = (
                [
                    mlp_model(features)
                    for mlp_model, features in zip(
                        mlp_benign_models, features_mlp_benign_list
                    )
                ]
                if args.n_parties > 2
                else []
            )

            combined_output = (
                torch.cat([gnn_output, mlp_adv_output] + mlp_benign_outputs, dim=1)
                if args.n_parties > 2
                else torch.cat([gnn_output, mlp_adv_output], dim=1)
            )
            final_output = active_model(combined_output)

            gnn_optimizer.zero_grad()
            mlp_adv_optimizer.zero_grad()
            for mlp_optimizer in mlp_benign_optimizers:
                mlp_optimizer.zero_grad()
            active_optimizer.zero_grad()

        else:
            gnn_output = gnn_model(features_gcn, data.edge_index)
            mlp_output = mlp_model(features_mlp)

            combined_output = torch.cat([gnn_output, mlp_output], dim=1)
            final_output = active_model(combined_output)

            gnn_optimizer.zero_grad()
            mlp_optimizer.zero_grad()
            active_optimizer.zero_grad()

        loss = nn.functional.nll_loss(final_output[train_mask], data.y[train_mask])

        active_model.eval()
        pred = final_output.argmax(dim=1)
        correct_test = (pred[test_mask] == data_utility.y[test_mask]).sum()
        acc_test = int(correct_test) / int(test_mask.sum())
        acc_train = int((pred[train_mask] == data_utility.y[train_mask]).sum()) / int(
            train_mask.sum()
        )

        metrics = {
            "epoch": epoch,
            "loss": loss.item(),
            "accuracy_test": acc_test,
            "accuracy_train": acc_train,
        }

        # Local logging
        if args.local_logging:
            with open(csv_files["metrics"], "a", newline="") as f:
                writer = csv.writer(f)
                for metric, value in metrics.items():
                    writer.writerow([epoch, metric, value])

        # Wandb logging (if enabled)
        if args.use_wandb:
            wandb.log(metrics, step=epoch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Test Accuracy: {acc_test:.4f}")
            print(f"  Train Accuracy: {acc_train:.4f}")

        loss.backward()

        if args.perform_attack:
            gradients = (
                mlp_adv_model.gradient_list[epoch]
                if args.multi_party
                else mlp_model.gradient_list[epoch]
            )
            pos_grad_sim, neg_grad_sim = perform_attack(
                args,
                gradients,
                data,
                args.attack_method,
                pos_samples,
                neg_samples,
                epoch,
            )
            auc, fpr, tpr = calculate_auc(pos_grad_sim, neg_grad_sim, epoch)
        if args.perform_attack_baseline:
            mlp_output_to_use = mlp_adv_output if args.multi_party else mlp_output
            pos_sim, neg_sim = perfrom_baseline_attack(
                args, mlp_output_to_use, pos_samples, neg_samples
            )
            auc, fpr, tpr = calculate_auc(pos_sim, neg_sim, epoch)
        if args.perform_attack_all_methods:
            for attack_method in args.attack_methods:
                if attack_method == "labels" and not labels_attack_done:
                    attack_data = process_attack_data_online(
                        gradients=None,
                        forward_values=None,
                        output_values=None,
                        data_attack=data_attack,
                        attack_methods=[attack_method],
                        args=args,
                    )
                    attack_results = compute_performance_metrics(
                        args,
                        inputs=attack_data[attack_method],
                        attack_method=attack_method,
                        labels=data_attack.adj,
                        indices=None,
                        ID=wandb.run.id,
                        attack_time=epoch,
                    )
                    attack_results = {
                        f"{k}-{attack_method}": v for k, v in attack_results.items()
                    }
                    attack_results["epoch"] = epoch
                    attack_results["attack_method"] = attack_method

                    # Local logging
                    if args.local_logging:
                        with open(csv_files["attack_results"], "a", newline="") as f:
                            writer = csv.writer(f)
                            for metric, value in attack_results.items():
                                writer.writerow([epoch, metric, value])

                    # Wandb logging (if enabled)
                    if args.use_wandb:
                        wandb.log(attack_results, step=epoch)

                    labels_attack_done = True
                if attack_method == "features" and not features_attack_done:
                    attack_data = process_attack_data_online(
                        gradients=None,
                        forward_values=None,
                        output_values=None,
                        data_attack=data_attack,
                        attack_methods=[attack_method],
                        args=args,
                    )
                    attack_results = compute_performance_metrics(
                        args,
                        inputs=attack_data[attack_method],
                        attack_method=attack_method,
                        labels=data_attack.adj,
                        indices=None,
                        ID=wandb.run.id,
                        attack_time=epoch,
                    )
                    attack_results = {
                        f"{k}-{attack_method}": v for k, v in attack_results.items()
                    }
                    attack_results["epoch"] = epoch
                    attack_results["attack_method"] = attack_method

                    # Local logging
                    if args.local_logging:
                        with open(csv_files["attack_results"], "a", newline="") as f:
                            writer = csv.writer(f)
                            for metric, value in attack_results.items():
                                writer.writerow([epoch, metric, value])

                    # Wandb logging (if enabled)
                    if args.use_wandb:
                        wandb.log(attack_results, step=epoch)

                    features_attack_done = True
                if attack_method == "gradients":
                    gradients = (
                        mlp_adv_model.gradient_list[epoch][
                            data_attack.attacked_nodes_mask
                        ].cpu()
                        if args.multi_party
                        else mlp_model.gradient_list[epoch][
                            data_attack.attacked_nodes_mask
                        ].cpu()
                    )
                    attack_data = process_attack_data_online(
                        gradients=gradients,
                        forward_values=None,
                        output_values=None,
                        data_attack=data_attack,
                        attack_methods=[attack_method],
                        args=args,
                    )
                    attack_results = compute_performance_metrics(
                        args,
                        inputs=attack_data[attack_method],
                        attack_method=attack_method,
                        labels=data_attack.adj,
                        indices=None,
                        ID=wandb.run.id,
                        attack_time=epoch,
                    )
                    attack_results = {
                        f"{k}-{attack_method}": v for k, v in attack_results.items()
                    }
                    attack_results["epoch"] = epoch
                    attack_results["attack_method"] = attack_method

                    # Local logging
                    if args.local_logging:
                        with open(csv_files["attack_results"], "a", newline="") as f:
                            writer = csv.writer(f)
                            for metric, value in attack_results.items():
                                writer.writerow([epoch, metric, value])

                    # Wandb logging (if enabled)
                    if args.use_wandb:
                        wandb.log(attack_results, step=epoch)

                if attack_method == "forward_values":
                    forward_values = (
                        mlp_adv_output.clone().detach().cpu()
                        if args.multi_party
                        else mlp_output.clone().detach().cpu()
                    )
                    attack_data = process_attack_data_online(
                        gradients=None,
                        forward_values=forward_values,
                        output_values=None,
                        data_attack=data_attack,
                        attack_methods=[attack_method],
                        args=args,
                    )
                    attack_results = compute_performance_metrics(
                        args,
                        inputs=attack_data[attack_method],
                        attack_method=attack_method,
                        labels=data_attack.adj,
                        indices=None,
                        ID=wandb.run.id,
                        attack_time=epoch,
                    )
                    attack_results = {
                        f"{k}-{attack_method}": v for k, v in attack_results.items()
                    }
                    attack_results["epoch"] = epoch
                    attack_results["attack_method"] = attack_method

                    # Local logging
                    if args.local_logging:
                        with open(csv_files["attack_results"], "a", newline="") as f:
                            writer = csv.writer(f)
                            for metric, value in attack_results.items():
                                writer.writerow([epoch, metric, value])

                    # Wandb logging (if enabled)
                    if args.use_wandb:
                        wandb.log(attack_results, step=epoch)

                if attack_method == "output_server":
                    output_values = final_output.clone().detach().cpu()
                    attack_data = process_attack_data_online(
                        gradients=None,
                        forward_values=None,
                        output_values=output_values,
                        data_attack=data_attack,
                        attack_methods=[attack_method],
                        args=args,
                    )
                    attack_results = compute_performance_metrics(
                        args,
                        inputs=attack_data[attack_method],
                        attack_method=attack_method,
                        labels=data_attack.adj,
                        indices=None,
                        ID=wandb.run.id,
                        attack_time=epoch,
                    )
                    attack_results = {
                        f"{k}-{attack_method}": v for k, v in attack_results.items()
                    }
                    attack_results["epoch"] = epoch
                    attack_results["attack_method"] = attack_method

                    # Local logging
                    if args.local_logging:
                        with open(csv_files["attack_results"], "a", newline="") as f:
                            writer = csv.writer(f)
                            for metric, value in attack_results.items():
                                writer.writerow([epoch, metric, value])

                    # Wandb logging (if enabled)
                    if args.use_wandb:
                        wandb.log(attack_results, step=epoch)

        gnn_optimizer.step()
        mlp_adv_optimizer.step() if args.multi_party else mlp_optimizer.step()
        if args.multi_party:
            for mlp_optimizer in mlp_benign_optimizers:
                mlp_optimizer.step()
        active_optimizer.step()

        if args.store_forward_mlp and epoch + 1 in args.attack_epochs:
            mlp_forward_list.append(
                (
                    mlp_adv_output.detach().cpu().numpy()
                    if args.multi_party
                    else mlp_output.detach().cpu().numpy()
                )
            )

        if args.store_output_server and epoch + 1 in args.attack_epochs:
            final_output_ = torch.clone(final_output)
            output_server.append(final_output_.detach().cpu().numpy())

    if args.store_gradients:
        if isinstance(args.attack_epochs, list):
            gradients_to_save = []
            for epoch in args.attack_epochs:
                epoch -= 1
                if (
                    0
                    <= epoch
                    < len(
                        (
                            mlp_adv_model.gradient_list
                            if args.multi_party
                            else mlp_model.gradient_list
                        )
                    )
                ):
                    gradients_to_save.append(
                        (
                            mlp_adv_model.gradient_list
                            if args.multi_party
                            else mlp_model.gradient_list
                        )[epoch].cpu()
                    )
                else:
                    print(f"Index out of bounds: {epoch}")
            with open(f"{log_dir}/gradients.pkl", "wb") as f:
                pickle.dump(gradients_to_save, f)
            with open(f"{log_dir}/train_mask.pkl", "wb") as f:
                pickle.dump(train_mask.tolist(), f)
        else:
            print("Error: attack_epochs is not a valid list of integers.")

    if args.store_forward_mlp:
        with open(f"{log_dir}/forward_mlp.pkl", "wb") as f:
            pickle.dump(mlp_forward_list, f)
    if args.store_output_server:
        with open(f"{log_dir}/output_server.pkl", "wb") as f:
            pickle.dump(output_server, f)


if __name__ == "__main__":
    main()